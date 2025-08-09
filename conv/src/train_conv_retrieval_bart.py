import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import transformers
import wandb
from accelerate import Accelerator
from accelerate.utils import set_seed
from loguru import logger
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer, AutoModel

from config_copy import gpt2_special_tokens_dict, prompt_special_tokens_dict
from dataset_conv_retrieval_prompt import CRSConvDataCollator, CRSConvDataset
from dataset_dbpedia import DBpedia
from evaluate_conv import ConvEvaluator

from utils import init_wandb_run, GENERATION, PROJECT_NAME, MODEL_NAME, wandb_logging, freeze_model_params, \
    count_parameters, save

from model_prompt import KGPrompt
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--output_dir", type=str, help="Where to store the final model.")
    parser.add_argument("--debug", action='store_true', help="Enable debug mode.")
    # Data
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name.")
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--context_max_length', type=int, help="Max encoder/decoder input length.")
    parser.add_argument('--resp_max_length', type=int, help="Max decoder input length.")
    parser.add_argument("--entity_max_length", type=int, help="Max entity length in dataset.")
    parser.add_argument("--prompt_max_length", type=int, default=50)
    parser.add_argument("--tokenizer", type=str)
    parser.add_argument("--ignore_pad_token_for_loss", action='store_true')
    parser.add_argument("--text_tokenizer", type=str)
    # Model
    parser.add_argument("--model", type=str)
    parser.add_argument("--max_gen_len", type=int, default=50)
    parser.add_argument("--text_encoder", type=str)
    parser.add_argument("--prompt_encoder", type=str)
    parser.add_argument("--n_prefix_conv", type=int)
    parser.add_argument("--num_bases", type=int, default=8, help="Number of RGCN bases.")
    parser.add_argument("--n_examples", type=int, default=5, help="Number of retrieved demonstrations.")
    # Optimizer
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--max_train_steps", type=int, default=None, help="Overrides num_train_epochs if set.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument('--max_grad_norm', type=float)
    parser.add_argument('--num_warmup_steps', type=int, default=10000)
    parser.add_argument('--fp16', action='store_true', help='Use mixed precision.')
    parser.add_argument('--mapping', action='store_true', help='Enable semantic mapping.')
    parser.add_argument('--bias_only', action='store_true', help='Train bias terms only.')
    # wandb
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--entity", type=str, help="wandb username")
    parser.add_argument("--project", type=str, help="wandb project name")
    parser.add_argument("--name", type=str, help="wandb run name")
    parser.add_argument("--log_all", action="store_true", help="Log on all processes.")
    parser.add_argument('--type_of_run', default='full', help='Experiment type (full, ablation, etc.)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    config = vars(args)

    accelerator = Accelerator(device_placement=False)
    device = accelerator.device

    local_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    logger.remove()
    logger.add(sys.stderr, level='DEBUG' if accelerator.is_local_main_process else 'ERROR')
    logger.add(f'log/{local_time}.log', level='DEBUG' if accelerator.is_local_main_process else 'ERROR')
    logger.info(accelerator.state)
    logger.info(config)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()

    # wandb initialization
    if args.use_wandb:
        name = args.name if args.name else local_time
        name += '_' + str(accelerator.process_index)
        if args.log_all:
            group = args.name if args.name else 'DDP_' + local_time
            run = wandb.init(entity=args.entity, project=args.project, group=group, config=config, name=name)
        else:
            run = wandb.init(entity=args.entity, project=args.project, config=config, name=name) if accelerator.is_local_main_process else None
    else:
        run = None

    if args.seed is not None:
        set_seed(args.seed)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    # Load KG and tokenizers/models
    kg = DBpedia(dataset=args.dataset, debug=args.debug).get_entity_kg_info()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.add_special_tokens(gpt2_special_tokens_dict)

    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model = model.to(device)

    text_tokenizer = AutoTokenizer.from_pretrained(args.text_tokenizer)
    text_tokenizer.add_special_tokens(prompt_special_tokens_dict)
    text_encoder = AutoModel.from_pretrained(args.text_encoder)
    text_encoder.resize_token_embeddings(len(text_tokenizer))
    text_encoder = text_encoder.to(device)

    # Prompt encoder initialization
    prompt_encoder = KGPrompt(
        model.config.n_embd, text_encoder.config.hidden_size, model.config.n_head, model.config.n_layer, 2,
        n_entity=kg['num_entities'], num_relations=kg['num_relations'], num_bases=args.num_bases,
        edge_index=kg['edge_index'], edge_type=kg['edge_type'],
        n_prefix_conv=args.n_prefix_conv,
        prompt_max_length=args.prompt_max_length,
        n_examples=args.n_examples
    ).to(device)

    init_wandb_run(PROJECT_NAME, args.dataset, GENERATION, MODEL_NAME, vars(args), args.type_of_run, "Method", None)

    freeze_model_params(model, text_encoder, bias_only=args.bias_only)

    modules = [model, prompt_encoder, text_encoder]
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for m in modules for n, p in m.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
         "weight_decay": args.weight_decay},
        {"params": [p for m in modules for n, p in m.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
         "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Dataset/Dataloader setup
    train_dataset = CRSConvDataset(args.dataset, 'train', tokenizer, debug=args.debug,
                                   context_max_length=args.context_max_length, resp_max_length=args.resp_max_length,
                                   entity_max_length=args.entity_max_length,
                                   prompt_tokenizer=text_tokenizer, prompt_max_length=args.prompt_max_length,
                                   n_examples=args.n_examples)
    valid_dataset = CRSConvDataset(args.dataset, 'valid', tokenizer, debug=args.debug,
                                   context_max_length=args.context_max_length, resp_max_length=args.resp_max_length,
                                   entity_max_length=args.entity_max_length,
                                   prompt_tokenizer=text_tokenizer, prompt_max_length=args.prompt_max_length,
                                   n_examples=args.n_examples)
    test_dataset = CRSConvDataset(args.dataset, 'test', tokenizer, debug=args.debug,
                                  context_max_length=args.context_max_length, resp_max_length=args.resp_max_length,
                                  entity_max_length=args.entity_max_length,
                                  prompt_tokenizer=text_tokenizer, prompt_max_length=args.prompt_max_length,
                                  n_examples=args.n_examples)

    data_collator_teacher = CRSConvDataCollator(tokenizer, device, use_amp=accelerator.use_fp16, debug=args.debug, gen=False,
                                                ignore_pad_token_for_loss=args.ignore_pad_token_for_loss,
                                                context_max_length=args.context_max_length + args.resp_max_length,
                                                entity_max_length=args.entity_max_length, pad_entity_id=kg['pad_entity_id'],
                                                prompt_tokenizer=text_tokenizer, n_examples=args.n_examples,
                                                prompt_max_length=args.prompt_max_length)

    train_dataloader = DataLoader(train_dataset, batch_size=args.per_device_train_batch_size,
                                  shuffle=True, collate_fn=data_collator_teacher)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.per_device_eval_batch_size,
                                  num_workers=args.num_workers, collate_fn=data_collator_teacher)
    test_dataloader = DataLoader(test_dataset, batch_size=args.per_device_eval_batch_size,
                                 num_workers=args.num_workers, collate_fn=data_collator_teacher)

    data_collator_generator = CRSConvDataCollator(tokenizer, device, gen=True, use_amp=accelerator.use_fp16, debug=args.debug,
                                                  ignore_pad_token_for_loss=args.ignore_pad_token_for_loss,
                                                  context_max_length=args.context_max_length, resp_max_length=args.resp_max_length,
                                                  entity_max_length=args.entity_max_length, pad_entity_id=kg['pad_entity_id'],
                                                  prompt_tokenizer=text_tokenizer, n_examples=args.n_examples,
                                                  prompt_max_length=args.prompt_max_length)
    valid_gen_dataloader = DataLoader(valid_dataset, batch_size=args.per_device_eval_batch_size,
                                      num_workers=args.num_workers, collate_fn=data_collator_generator)
    test_gen_dataloader = DataLoader(test_dataset, batch_size=args.per_device_eval_batch_size,
                                     num_workers=args.num_workers, collate_fn=data_collator_generator)

    evaluator = ConvEvaluator(tokenizer, os.path.join('log', f'gen_{local_time}.jsonl'))
    model, prompt_encoder, optimizer, train_dataloader = accelerator.prepare(model, prompt_encoder, optimizer, train_dataloader)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    lr_scheduler = get_linear_schedule_with_warmup(optimizer, args.num_warmup_steps, args.max_train_steps)
    lr_scheduler = accelerator.prepare(lr_scheduler)

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)

    metric, mode = 'dist@4', 1
    best_metric = 0 if mode == 1 else float('inf')
    best_metric_dir = os.path.join(args.output_dir, 'best')
    os.makedirs(best_metric_dir, exist_ok=True)

    # =======================
    #        TRAIN LOOP
    # =======================
    for epoch in range(args.num_train_epochs):
        train_loss = []
        model.train()
        for step, batch in enumerate(train_dataloader):
            with torch.no_grad():
                token_embeds = text_encoder(**batch['prompt']).last_hidden_state

            old_len = batch['context']['attention_mask'].shape[1]
            prompt_augmented_input_embeddings, new_attention_mask, _, _, ffloss = prompt_encoder(
                entity_ids=batch['entity'], token_embeds=token_embeds, output_entity=False,
                use_conv_prefix=True, mapping=args.mapping, context_str=batch['context_str'],
                word_embeddings=model.get_input_embeddings().weight,
                context_input_embeddings=model.get_input_embeddings()(batch['context']['input_ids']),
                attention_mask=batch['context']['attention_mask']
            )

            added_len = new_attention_mask.shape[1] - old_len
            if added_len > 0:
                pad_resp = torch.full((new_attention_mask.shape[0], added_len), -100,
                                      device=new_attention_mask.device, dtype=torch.long)
                batch['resp'] = torch.cat([pad_resp, batch['resp']], dim=1)

            batch['context']['input_ids'] = None
            batch['context']['inputs_embeds'] = prompt_augmented_input_embeddings
            batch['context']['attention_mask'] = new_attention_mask

            loss = model(inputs_embeds=batch['context']['inputs_embeds'], labels=batch['resp'], return_dict=True)['loss'] / args.gradient_accumulation_steps
            if ffloss is not None:
                loss = loss + 0.2 * ffloss / args.gradient_accumulation_steps

            accelerator.backward(loss)
            train_loss.append(float(loss))

            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                if args.max_grad_norm:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                if run:
                    run.log({'loss': np.mean(train_loss) * args.gradient_accumulation_steps})
            if progress_bar.n >= args.max_train_steps:
                break

        # Validation and Test loops would follow (loss + generation metrics)
        # [Omitted here for brevity — logic same as training loop, replacing optimizer steps with evaluation + metric logging]

    final_dir = os.path.join(args.output_dir, 'final')
    save(prompt_encoder, f"{final_dir}/prompt_encoder")
    save(model, f"{final_dir}/gen_model")
    logger.info(f'save final model')
    wandb.finish()
