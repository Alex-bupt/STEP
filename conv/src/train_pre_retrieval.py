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

from dataset_dbpedia import DBpedia
from dataset_pre_retrieval import CRSDataset, CRSDataCollator
from evaluate_rec import RecEvaluator
from config_copy import gpt2_special_tokens_dict, prompt_special_tokens_dict
from model_prompt import KGPrompt
from rec_model import DemonstrationRecModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--output_dir", type=str, default='save', help="Directory to store the final model.")
    parser.add_argument("--debug", action='store_true', help="Enable debug mode.")
    # Data
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name.")
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument("--max_length", type=int, help="Max input length in dataset.")
    parser.add_argument("--prompt_max_length", type=int)
    parser.add_argument("--entity_max_length", type=int, help="Max entity length in dataset.")
    parser.add_argument("--tokenizer", type=str)
    parser.add_argument("--text_tokenizer", type=str)
    # Model
    parser.add_argument("--model", type=str, required=True,
                        help="Path to pretrained model or Hugging Face model identifier.")
    parser.add_argument("--text_encoder", type=str)
    parser.add_argument("--num_bases", type=int, default=8, help="Number of bases in RGCN.")
    # Optimizer
    parser.add_argument("--num_train_epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--max_train_steps", type=int, default=None,
                        help="Total training steps; overrides num_train_epochs if set.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                        help="Batch size per device for training.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4,
                        help="Batch size per device for evaluation.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of steps to accumulate before backward/update.")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Initial learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument('--max_grad_norm', type=float)
    parser.add_argument('--num_warmup_steps', type=int, default=10000)
    parser.add_argument("--fp16", action='store_true')

    parser.add_argument("--mapping", action='store_true', help="Mapping flag (retained, but unused during pretraining).")

    # wandb
    parser.add_argument("--use_wandb", action="store_true", help="Enable wandb logging.")
    parser.add_argument("--entity", type=str, help="wandb username.")
    parser.add_argument("--project", type=str, help="wandb project name.")
    parser.add_argument("--name", type=str, help="wandb run name.")
    parser.add_argument("--log_all", action="store_true", help="Log in all processes, otherwise only in rank 0.")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    config = vars(args)

    accelerator = Accelerator(device_placement=False)
    device = accelerator.device

    local_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    logger.remove()
    logger.add(sys.stderr, level='DEBUG' if accelerator.is_local_main_process else 'ERROR')
    logger.add(f'log/{local_time}.log', level='DEBUG' if accelerator.is_local_main_process else 'ERROR')
    logger.info(config)
    logger.info(accelerator.state)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()

    # wandb setup
    if args.use_wandb:
        name = args.name if args.name else local_time
        name += '_' + str(accelerator.process_index)

        if args.log_all:
            group = args.name if args.name else 'DDP_' + local_time
            run = wandb.init(entity=args.entity, project=args.project, group=group, config=config, name=name)
        else:
            if accelerator.is_local_main_process:
                run = wandb.init(entity=args.entity, project=args.project, config=config, name=name)
            else:
                run = None
    else:
        run = None

    # Seed setting
    if args.seed is not None:
        set_seed(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # Load KG
    kg = DBpedia(dataset=args.dataset, debug=args.debug).get_entity_kg_info()

    # Tokenizers and models
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.add_special_tokens(gpt2_special_tokens_dict)
    model = DemonstrationRecModel(entity_dim=128).to(device)

    text_tokenizer = AutoTokenizer.from_pretrained(args.text_tokenizer)
    text_tokenizer.add_special_tokens(prompt_special_tokens_dict)
    text_encoder = AutoModel.from_pretrained(args.text_encoder)
    text_encoder.resize_token_embeddings(len(text_tokenizer))
    text_encoder = text_encoder.to(device)

    # Align RGCN output dimension to match DemonstrationRecModel
    prompt_encoder = KGPrompt(
        768, text_encoder.config.hidden_size, 8, 12, 2,
        n_entity=kg['num_entities'], num_relations=kg['num_relations'], num_bases=args.num_bases,
        edge_index=kg['edge_index'], edge_type=kg['edge_type'],
        entity_hidden_size=128
    ).to(device)

    # Optimizer
    modules = [prompt_encoder, model, text_encoder]
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for model_m in modules for n, p in model_m.named_parameters()
                       if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for model_m in modules for n, p in model_m.named_parameters()
                       if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Data
    train_dataset = CRSDataset(
        dataset=args.dataset, split='train', tokenizer=tokenizer, debug=args.debug, max_length=args.max_length,
        prompt_tokenizer=text_tokenizer, prompt_max_length=args.prompt_max_length,
        entity_max_length=args.entity_max_length
    )
    valid_dataset = CRSDataset(
        dataset=args.dataset, split='valid', tokenizer=tokenizer, debug=args.debug, max_length=args.max_length,
        prompt_tokenizer=text_tokenizer, prompt_max_length=args.prompt_max_length,
        entity_max_length=args.entity_max_length
    )
    test_dataset = CRSDataset(
        dataset=args.dataset, split='test', tokenizer=tokenizer, debug=args.debug, max_length=args.max_length,
        prompt_tokenizer=text_tokenizer, prompt_max_length=args.prompt_max_length,
        entity_max_length=args.entity_max_length
    )
    data_collator = CRSDataCollator(
        tokenizer=tokenizer, device=device, pad_entity_id=kg['pad_entity_id'],
        max_length=args.max_length, entity_max_length=args.entity_max_length,
        use_amp=accelerator.use_fp16, debug=args.debug,
        prompt_tokenizer=text_tokenizer, prompt_max_length=args.prompt_max_length
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        collate_fn=data_collator,
        shuffle=True
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.per_device_eval_batch_size,
        collate_fn=data_collator,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.per_device_eval_batch_size,
        collate_fn=data_collator,
    )

    evaluator = RecEvaluator()
    prompt_encoder, optimizer, train_dataloader, valid_dataloader, test_dataloader = accelerator.prepare(
        prompt_encoder, optimizer, train_dataloader, valid_dataloader, test_dataloader
    )

    # Steps & scheduler
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    completed_steps = 0

    lr_scheduler = get_linear_schedule_with_warmup(optimizer, args.num_warmup_steps, args.max_train_steps)
    lr_scheduler = accelerator.prepare(lr_scheduler)

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (parallel/distributed/accumulated) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)

    metric, mode = 'loss', -1
    best_metric = 0 if mode == 1 else float('inf')
    best_metric_dir = os.path.join(args.output_dir, 'best')
    os.makedirs(best_metric_dir, exist_ok=True)

    # Training loop
    for epoch in range(args.num_train_epochs):
        train_loss = []
        prompt_encoder.train()

        for step, batch in enumerate(train_dataloader):
            with torch.no_grad():
                token_embeds = text_encoder(**batch['prompt']).last_hidden_state

            entity_embeds, all_entity_embs, retrieved_entity_embeds = prompt_encoder(
                entity_ids=batch['entity'],
                token_embeds=None,
                output_entity=True,
                use_conv_prefix=False,
                mapping=False,
                context_str=batch['context_str'],
                use_fformer=False,
                rec_labels=batch['context']['rec_labels'],
                retrieved_entity_ids=batch['retrieved_entity']
            )

            rec_scores, loss = model(
                token_embeds,
                entity_embeds,
                all_entity_embs,
                retrieved_entity_embeds=retrieved_entity_embeds,
                retrieved_entity_ids=batch['retrieved_entity'],
                entity_ids=batch['entity'],
                rec_labels=batch['context']['rec_labels']
            )

            accelerator.backward(loss)
            train_loss.append(float(loss))

            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                if args.max_grad_norm is not None:
                    accelerator.clip_grad_norm_(prompt_encoder.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                progress_bar.update(1)
                completed_steps += 1
                if run:
                    run.log({'loss': np.mean(train_loss) * args.gradient_accumulation_steps})

            if completed_steps >= args.max_train_steps:
                break

        train_loss = np.mean(train_loss) * args.gradient_accumulation_steps
        logger.info(f'epoch {epoch} train loss {train_loss}')
        del train_loss, batch

        # Validation loop
        valid_loss = []
        prompt_encoder.eval()
        for batch in tqdm(valid_dataloader):
            with torch.no_grad():
                token_embeds = text_encoder(**batch['prompt']).last_hidden_state
                entity_embeds, all_entity_embs, retrieved_entity_embeds = prompt_encoder(
                    entity_ids=batch['entity'],
                    token_embeds=None,
                    output_entity=True,
                    use_conv_prefix=False,
                    mapping=False,
                    use_fformer=False,
                    context_str=batch['context_str'],
                    rec_labels=batch['context']['rec_labels'],
                    retrieved_entity_ids=batch['retrieved_entity']
                )
                rec_scores, loss = model(
                    token_embeds,
                    entity_embeds,
                    all_entity_embs,
                    retrieved_entity_embeds=retrieved_entity_embeds,
                    retrieved_entity_ids=batch['retrieved_entity'],
                    entity_ids=batch['entity'],
                    rec_labels=batch['context']['rec_labels']
                )
                valid_loss.append(float(loss))

                logits = rec_scores
                ranks = torch.topk(logits, k=50, dim=-1).indices.tolist()
                labels = batch['context']['rec_labels']
                evaluator.evaluate(ranks, labels)

        # Validation metrics
        report = accelerator.gather(evaluator.report())
        for k, v in report.items():
            report[k] = v.sum().item()

        valid_report = {f'valid/{k}': v / report['count'] for k, v in report.items() if k != 'count'}
        valid_report['valid/loss'] = np.mean(valid_loss)
        valid_report['epoch'] = epoch
        logger.info(f'{valid_report}')
        if run:
            run.log(valid_report)
        evaluator.reset_metric()

        if valid_report[f'valid/{metric}'] * mode > best_metric * mode:
            prompt_encoder.save(best_metric_dir)
            best_metric = valid_report[f'valid/{metric}']
            logger.info(f'new best model with {metric}')

        # Test loop
        test_loss = []
        prompt_encoder.eval()
        for batch in tqdm(test_dataloader):
            with torch.no_grad():
                token_embeds = text_encoder(**batch['prompt']).last_hidden_state
                entity_embeds, all_entity_embs, retrieved_entity_embeds = prompt_encoder(
                    entity_ids=batch['entity'],
                    token_embeds=None,
                    output_entity=True,
                    use_conv_prefix=False,
                    mapping=False,
                    use_fformer=False,
                    context_str=batch['context_str'],
                    rec_labels=batch['context']['rec_labels'],
                    retrieved_entity_ids=batch['retrieved_entity']
                )
                rec_scores, loss = model(
                    token_embeds,
                    entity_embeds,
                    all_entity_embs,
                    retrieved_entity_embeds=retrieved_entity_embeds,
                    retrieved_entity_ids=batch['retrieved_entity'],
                    entity_ids=batch['entity'],
                    rec_labels=batch['context']['rec_labels']
                )
                test_loss.append(float(loss))

                logits = rec_scores
                ranks = torch.topk(logits, k=50, dim=-1).indices.tolist()
                labels = batch['context']['rec_labels']
                evaluator.evaluate(ranks, labels)

        report = accelerator.gather(evaluator.report())
        for k, v in report.items():
            report[k] = v.sum().item()

        test_report = {f'test/{k}': v / report['count'] for k, v in report.items() if k != 'count'}
        test_report['test/loss'] = np.mean(test_loss)
        test_report['epoch'] = epoch
        logger.info(f'{test_report}')
        if run:
            run.log(test_report)
        evaluator.reset_metric()

    final_dir = os.path.join(args.output_dir, 'final')
    os.makedirs(final_dir, exist_ok=True)
    prompt_encoder.save(final_dir)
    logger.info(f'save final model')
