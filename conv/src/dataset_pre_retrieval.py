import json
import os
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from utils import padded_tensor


class CRSDataset(Dataset):
    def __init__(
        self, dataset, split, tokenizer, debug=False,
        max_length=None, entity_max_length=None,
        prompt_tokenizer=None, prompt_max_length=None
    ):
        super(CRSDataset, self).__init__()
        self.debug = debug
        self.tokenizer = tokenizer
        self.prompt_tokenizer = prompt_tokenizer

        # Max length for context tokenizer
        self.max_length = max_length or self.tokenizer.model_max_length

        # Max length for prompt tokenizer (reserve one position for CLS/SEP)
        self.prompt_max_length = prompt_max_length or self.prompt_tokenizer.model_max_length
        self.prompt_max_length -= 1

        # Max length for entity sequence
        self.entity_max_length = entity_max_length or self.tokenizer.model_max_length

        dataset_dir = os.path.join('data', dataset)
        data_file = os.path.join(dataset_dir, f'{split}_data_processed.jsonl')
        # Alternative retrieval-based file:
        # data_file = os.path.join(dataset_dir, f'{split}_data_processed_retrieval.jsonl')

        self.data = []
        self.prepare_data(data_file)

    def prepare_data(self, data_file):
        with open(data_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if self.debug:
                lines = lines[:1024]

            for line in tqdm(lines):
                dialog = json.loads(line)

                # Skip samples without recommendation targets
                if len(dialog.get('rec', [])) == 0:
                    continue

                context = ''
                prompt_context = ''
                # Build alternating User/System context
                for i, utt in enumerate(dialog['context']):
                    if utt == '':
                        continue
                    if i % 2 == 0:
                        context += 'User: '
                        prompt_context += 'User: '
                    else:
                        context += 'System: '
                        prompt_context += 'System: '
                    context += utt
                    context += self.tokenizer.eos_token
                    prompt_context += utt
                    prompt_context += self.prompt_tokenizer.sep_token

                # Append current response
                if len(dialog['context']) % 2 == 0:
                    resp_prefix = 'System: '
                else:
                    resp_prefix = 'User: '
                resp = resp_prefix + dialog['resp']

                context += resp + self.tokenizer.eos_token
                prompt_context += resp + self.prompt_tokenizer.sep_token

                # Keep raw context string for F-former
                context_str = context

                # Tokenize context
                context_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(context))
                context_ids = context_ids[-self.max_length:]

                # Tokenize prompt
                prompt_ids = self.prompt_tokenizer.convert_tokens_to_ids(self.prompt_tokenizer.tokenize(prompt_context))
                prompt_ids = prompt_ids[-self.prompt_max_length:]
                prompt_ids.insert(0, self.prompt_tokenizer.cls_token_id)

                # Create one sample per recommendation target
                for rec in dialog['rec']:
                    data = {
                        'context': context_ids,
                        'prompt': prompt_ids,
                        'entity': dialog['entity'][-self.entity_max_length:],
                        # Optionally add retrieved entities:
                        # 'retrieved_entity': list(set(dialog['retrieved_response_entity'] + dialog['retrieved_context_entity']))[-self.entity_max_length:],
                        'retrieved_entity': [],
                        'rec': rec,
                        'context_str': context_str,  # raw string for F-former
                    }
                    self.data.append(data)

    def __getitem__(self, ind):
        return self.data[ind]

    def __len__(self):
        return len(self.data)


class CRSDataCollator:
    def __init__(
        self, tokenizer, device, pad_entity_id, debug=False,
        max_length=None, entity_max_length=None,
        prompt_tokenizer=None, prompt_max_length=None,
        use_amp=False
    ):
        self.debug = debug
        self.device = device
        self.tokenizer = tokenizer
        self.prompt_tokenizer = prompt_tokenizer

        self.padding = 'max_length' if self.debug else True
        self.pad_to_multiple_of = 8 if use_amp else None

        self.max_length = max_length or self.tokenizer.model_max_length
        self.prompt_max_length = prompt_max_length or self.prompt_tokenizer.model_max_length

        self.pad_entity_id = pad_entity_id
        self.entity_max_length = entity_max_length or self.tokenizer.model_max_length

    def __call__(self, data_batch):
        context_batch = defaultdict(list)
        prompt_batch = defaultdict(list)
        entity_batch = []
        label_batch = []
        retrieved_entity = []
        context_str_list = []  # keep raw string contexts

        for data in data_batch:
            context_batch['input_ids'].append(data['context'])
            prompt_batch['input_ids'].append(data['prompt'])
            entity_batch.append(data['entity'])
            label_batch.append(data['rec'])
            retrieved_entity.append(data['retrieved_entity'])
            context_str_list.append(data['context_str'])

        input_batch = {}

        # Pad context sequences
        context_batch = self.tokenizer.pad(
            context_batch, padding=self.padding, max_length=self.max_length, pad_to_multiple_of=self.pad_to_multiple_of
        )
        context_batch['rec_labels'] = label_batch
        for k, v in context_batch.items():
            if not isinstance(v, torch.Tensor):
                context_batch[k] = torch.as_tensor(v, device=self.device)
        input_batch['context'] = context_batch

        # Pad prompt sequences
        prompt_batch = self.prompt_tokenizer.pad(
            prompt_batch, padding=self.padding, max_length=self.prompt_max_length, pad_to_multiple_of=self.pad_to_multiple_of
        )
        for k, v in prompt_batch.items():
            if not isinstance(v, torch.Tensor):
                prompt_batch[k] = torch.as_tensor(v, device=self.device)
        input_batch['prompt'] = prompt_batch

        # Pad entity sequences
        entity_batch = padded_tensor(entity_batch, pad_idx=self.pad_entity_id, pad_tail=True, device=self.device)
        retrieved_entity_batch = padded_tensor(retrieved_entity, pad_idx=self.pad_entity_id, pad_tail=True, device=self.device)

        input_batch['entity'] = entity_batch
        input_batch['retrieved_entity'] = retrieved_entity_batch

        # Add raw string context list for F-former
        input_batch['context_str'] = context_str_list

        return input_batch


if __name__ == '__main__':
    from dataset_dbpedia import DBpedia
    from config import gpt2_special_tokens_dict
    from pprint import pprint

    debug = True
    device = torch.device('cpu')
    dataset_name = 'inspired'

    kg = DBpedia(dataset_name, debug=debug).get_entity_kg_info()

    model_name_or_path = '../utils/tokenizer/dialogpt-small'
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.add_special_tokens(gpt2_special_tokens_dict)
    prompt_tokenizer = AutoTokenizer.from_pretrained('../utils/tokenizer/roberta-base')

    dataset = CRSDataset(
        dataset=dataset_name, split='test', tokenizer=tokenizer, debug=debug,
        prompt_tokenizer=prompt_tokenizer
    )

    # Sample inspection
    for i in range(min(3, len(dataset))):
        data = dataset[i]
        print('--- sample', i, '---')
        print('context_str:', data['context_str'][:120].replace('\n', ' '), '...')
        print('context_ids decode:', tokenizer.decode(data['context']))
        print('prompt_ids decode:', prompt_tokenizer.decode(data['prompt']))
        print()

    data_collator = CRSDataCollator(
        tokenizer=tokenizer, device=device, pad_entity_id=kg['pad_entity_id'],
        prompt_tokenizer=prompt_tokenizer
    )
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        collate_fn=data_collator,
    )

    input_max_len = 0
    entity_max_len = 0
    for batch in tqdm(dataloader):
        if debug:
            pprint({k: (v if isinstance(v, torch.Tensor) else {kk: vv.shape if isinstance(vv, torch.Tensor) else type(vv) for kk, vv in v.items()})
                    for k, v in batch.items()})
            print('decoded context[1]:', tokenizer.decode(batch['context']['input_ids'][1]))
            print('decoded prompt[1]:', prompt_tokenizer.decode(batch['prompt']['input_ids'][1]))
            print('context_str (list[str]) len:', len(batch['context_str']), '; example:', batch['context_str'][0][:120], '...')
            break

        input_max_len = max(input_max_len, batch['context']['input_ids'].shape[1])
        entity_max_len = max(entity_max_len, batch['entity'].shape[1])

    print('input_max_len =', input_max_len)
    print('entity_max_len =', entity_max_len)
    # Example lengths:
    # redial: (1024, 31), (688, 29), (585, 19) -> (1024, 31)
    # inspired: (1024, 30), (902, 23), (945, 32) -> (1024, 32)
