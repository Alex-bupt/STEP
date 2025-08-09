import argparse
import os
import sys
import time

import torch
import transformers
import wandb
from accelerate import Accelerator
from accelerate.utils import set_seed
from loguru import logger
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

from config_copy import gpt2_special_tokens_dict, prompt_special_tokens_dict
from dataset_conv_retrieval_prompt import CRSConvDataCollator, CRSConvDataset
from dataset_dbpedia import DBpedia
from evaluate_conv import ConvEvaluator
from model_prompt import KGPrompt

from utils import load, gpt2_special_tokens_dict as _g, prompt_special_tokens_dict as _p  # 避免误删引用


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--output_dir", type=str, help="Where to store the final model.")
    parser.add_argument("--debug", action='store_true', help="Debug mode.")
    # data
    parser.add_argument("--dataset", type=str, required=True, help="A file containing all data.")
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--context_max_length', type=int, help="max length of both encoder and decoder input.")
    parser.add_argument('--resp_max_length', type=int, help="max length of decoder input.")
    parser.add_argument("--entity_max_length", type=int, help="max entity length in dataset.")
    parser.add_argument("--prompt_max_length", type=int)
    parser.add_argument("--tokenizer", type=str)
    parser.add_argument("--ignore_pad_token_for_loss", action='store_true')
    parser.add_argument("--text_tokenizer", type=str)
    # model
    parser.add_argument("--model", type=str)
    parser.add_argument("--max_gen_len", type=int, default=50)
    parser.add_argument("--text_encoder", type=str)
    parser.add_argument("--prompt_encoder", type=str)
    parser.add_argument("--n_prefix_conv", type=int)
    parser.add_argument("--num_bases", type=int, default=8, help="num_bases in RGCN")
    # optim (unused here, keep for CLI compatibility)
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument('--max_grad_norm', type=float)
    parser.add_argument('--num_warmup_steps', type=int, default=10000)
    parser.add_argument('--fp16', action='store_true', help='use automatic mixed precision to speed up.')
    # wandb
    parser.add_argument("--use_wandb", action="store_true", help="whether to use wandb")
    parser.add_argument("--entity", type=str, help="wandb username")
    parser.add_argument("--project", type=str, help="wandb exp project")
    parser.add_argument("--name", type=str, help="wandb exp name")
    parser.add_argument("--log_all", action="store_true", help="log in all processes, otherwise only in rank0")

    parser.add_argument("--check_point", type=str, help="path to the check point")
    parser.add_argument("--n_examples", type=int, default=5, help="number of retrieved demonstrations")
    parser.add_argument('--mapping', action='store_true', help='if we use semantic mapping')
    parser.add_argument('--bias_only', action='store_true', help='if we use semantic mapping')
    parser.add_argument('--type_of_run', default='full', help='type of the experiment, eg: full, ablation, analysis')

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
    logger.info(accelerator.state)
    logger.info(config)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()

    # wandb
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

    # seed
    if args.seed is not None:
        set_seed(args.seed)

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

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

    print(tokenizer.pad_token_id)
    print(tokenizer.encode('<movie>'))

    prompt_encoder = KGPrompt(
        model.config.n_embd, text_encoder.config.hidden_size, model.config.n_head, model.config.n_layer, 2,
        n_entity=kg['num_entities'], num_relations=kg['num_relations'], num_bases=args.num_bases,
        edge_index=kg['edge_index'], edge_type=kg['edge_type'],
        n_prefix_rec=args.n_prefix_conv,
        prompt_max_length=args.prompt_max_length,
        n_examples=args.n_examples
    )

    if args.check_point is not None:
        prompt_encoder = load(prompt_encoder, args.check_point + "/prompt_encoder")
        model = load(model, args.check_point + "/gen_model")

    prompt_encoder = prompt_encoder.to(device)
    model = model.to(device)
    model, prompt_encoder = accelerator.prepare(model, prompt_encoder)

    # data
    dataset = CRSConvDataset(
        args.dataset, args.split, tokenizer, debug=args.debug,
        context_max_length=args.context_max_length, resp_max_length=args.resp_max_length,
        entity_max_length=args.entity_max_length,
        prompt_tokenizer=text_tokenizer, prompt_max_length=args.prompt_max_length,
        n_examples=args.n_examples
    )
    data_collator_generator = CRSConvDataCollator(
        tokenizer=tokenizer, device=device, gen=True, use_amp=accelerator.use_fp16, debug=args.debug,
        ignore_pad_token_for_loss=args.ignore_pad_token_for_loss,
        context_max_length=args.context_max_length, resp_max_length=args.resp_max_length,
        entity_max_length=args.entity_max_length, pad_entity_id=kg['pad_entity_id'],
        prompt_tokenizer=text_tokenizer,
        n_examples=args.n_examples,
        prompt_max_length=args.prompt_max_length
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.per_device_eval_batch_size,
        num_workers=args.num_workers,
        collate_fn=data_collator_generator,
    )

    gen_dir = os.path.join('save', args.dataset)
    os.makedirs(gen_dir, exist_ok=True)
    model_name = args.check_point.split('/')[-2] if args.check_point else "no_ckpt"
    gen_file_path = os.path.join(gen_dir, f'{model_name}_{args.split}.jsonl')
    evaluator = ConvEvaluator(tokenizer=tokenizer, log_file_path=gen_file_path)

    for batch in tqdm(dataloader, disable=not accelerator.is_local_main_process):
        with torch.no_grad():
            # 文本侧编码（RoBERTa 等）
            token_embeds = text_encoder(**batch['prompt']).last_hidden_state

            # 计算注入了 F-former 的输入 embedding
            # 注意：把 context_attention_mask 改为 attention_mask（KGPrompt 的签名里是 attention_mask）
            prompt_augmented_input_embeddings, new_attention_mask, _, _ = prompt_encoder(
                entity_ids=batch['entity'],
                token_embeds=token_embeds,
                output_entity=False,
                use_conv_prefix=True,
                mapping=args.mapping,
                word_embeddings=model.get_input_embeddings().weight,
                context_input_embeddings=model.get_input_embeddings()(batch['context']['input_ids']),
                attention_mask=batch['context']['attention_mask']
            )

            # 喂给生成模型
            batch['context']['inputs_embeds'] = prompt_augmented_input_embeddings
            batch['context']['attention_mask'] = new_attention_mask

            gen_seqs = accelerator.unwrap_model(model).generate(
                input_ids=None,
                inputs_embeds=batch['context']['inputs_embeds'],
                max_new_tokens=args.max_gen_len,
                no_repeat_ngram_size=3
            )

            gen_resp_ids = []
            for gen_seq, length in zip(gen_seqs, batch['context_len']):
                gen_seq = [token_id.item() for token_id in gen_seq if token_id != tokenizer.pad_token_id]
                gen_resp_ids.append(gen_seq)
            evaluator.evaluate(gen_resp_ids, batch['resp'], log=accelerator.is_local_main_process)

    # metric
    accelerator.wait_for_everyone()
    report = evaluator.report()
    test_report = {}
    for k, v in report.items():
        test_report[f'{args.split}/{k}'] = v
    logger.info(test_report)
    if run:
        run.log(test_report)