import random
import json
from retriv import SparseRetriever
import os
import pickle

def create_query(list_entity):
    if len(list_entity) == 0:
        return '-9999'
    text = ''
    for en in list_entity:
        text += str(en)
        text += ' '
    return text

def filtering_examples(conv_id, sampled_examples, top_k = 3):
    new_examples = []
    for example in sampled_examples:
        if example['conv_id'] == conv_id:
            continue
        new_examples.append(example)
    return new_examples[:top_k]

def create_entity_collection(train_data):
    collection = []
    for id, data in enumerate(train_data):
        data = json.loads(data)
        if len(data['entity']) == 0:
            text = '-9999'
        else:
            text = ''
            for en in data['entity']:
                text += str(en)
                text += ' '
        dic = {"id": str(id), "conv_id": str(data['conv_id']), "text": text}
        collection.append(dic)
    return collection


def read_file(file_path):
    with open(file_path, 'r') as  f:
        lines = f.readlines()
        return lines

def merge_utts(list_utt):
    s = ''
    for utt in list_utt:
        s += utt
        s += '. '
    return s

def write_file(data, file_path):
    with open(file_path, 'w') as f:
        for line in data:
            f.write(json.dumps(line) + '\n') 


def create_new_example(example, sampled_examples, is_train = True):
    retrieved_contexts = []
    retrieved_resps = []
    un_mask_retrieved_resp = []
    retrieved_context_entity = []
    retrieved_response_entity = []
    new_example = example.copy()
    for ex in sampled_examples:
        if is_train:
            if ex['conv_id'] != example['conv_id']:
                retrieved_contexts.append(merge_utts(ex['context']))
                retrieved_resps.append(ex['resp'])
                un_mask_retrieved_resp.append(ex['un_mask_utt'])
                retrieved_context_entity.extend(ex['entity'])
                retrieved_response_entity.extend(ex['rec'])
        else:
            retrieved_contexts.append(merge_utts(ex['context']))
            retrieved_resps.append(ex['resp'])
            un_mask_retrieved_resp.append(ex['un_mask_utt'])
            retrieved_context_entity.extend(ex['entity'])
            retrieved_response_entity.extend(ex['rec'])

    new_example['retrieved_resp'] = retrieved_resps
    new_example['retrieved_contexts'] = retrieved_contexts
    new_example['un_mask_retrieved_resp'] = un_mask_retrieved_resp
    new_example['retrieved_context_entity'] = retrieved_context_entity
    new_example['retrieved_response_entity'] = retrieved_response_entity
    
    return new_example

def random_retrieval(train_data, valid_data, test_data, n = 5):
    num_train_examples = len(train_data)
    list_indexes = list(range(num_train_examples))
    new_train_data, new_valid_data, new_test_data = [], [], []
    for example in train_data:
        example = json.loads(example)
        sampled_indexes = random.choices(list_indexes, k = n)
        sampled_examples = [json.loads(train_data[x]) for x in sampled_indexes]
        new_example = create_new_example(example, sampled_examples, is_train=True)
        new_train_data.append(new_example)

    for example in valid_data:
        example = json.loads(example)
        sampled_indexes = random.choices(list_indexes, k = n)
        sampled_examples = [json.loads(train_data[x]) for x in sampled_indexes]
        new_example = create_new_example(example, sampled_examples, is_train=False)
        new_valid_data.append(new_example)

    for example in test_data:
        example = json.loads(example)
        sampled_indexes = random.choices(list_indexes, k = n)
        sampled_examples = [json.loads(train_data[x]) for x in sampled_indexes]
        new_example = create_new_example(example, sampled_examples, is_train=False)
        new_test_data.append(new_example)
    
    return new_train_data, new_valid_data, new_test_data

def bm25_retrieval(train_data, valid_data, test_data, n = 5):

    index_name = 'bm25_index_inspired'
    if not os.path.exists(index_name):
        sr = SparseRetriever(
            index_name=index_name,
            model="bm25",
            min_df=1,
            tokenizer="whitespace",
            stemmer="english",
            stopwords="english",
            do_lowercasing=True,
            do_ampersand_normalization=True,
            do_special_chars_normalization=True,
            do_acronyms_normalization=True,
            do_punctuation_removal=True,
        )

        collection = create_entity_collection(train_data)
        sr.index(collection)
        with open(index_name, 'wb') as f:
            pickle.dump(sr, f)
    else:
        print("Loading index ......")
        with open(index_name, 'rb') as f:
            sr = pickle.load(f)
        
    
    idx = 0
    new_train_data, new_valid_data, new_test_data = [], [], []

    for example in train_data:
        example = json.loads(example)
        query = create_query(example['entity'])
        sampled_examples = sr.search(query, cutoff =50)
        sampled_examples = [json.loads(train_data[int(x['id'])]) for x in sampled_examples if x['text'] != '-9999']
        
        sampled_examples = filtering_examples(example['conv_id'], sampled_examples)

        new_example = create_new_example(example, sampled_examples, is_train=True)
        new_train_data.append(new_example)

    for example in valid_data:
        example = json.loads(example)
        query = create_query(example['entity'])
        sampled_examples = sr.search(query, cutoff =50)
        sampled_examples = [json.loads(train_data[int(x['id'])]) for x in sampled_examples if x['text'] != '-9999']
        
        sampled_examples = filtering_examples(example['conv_id'], sampled_examples)

        new_example = create_new_example(example, sampled_examples, is_train=False)
        new_valid_data.append(new_example)

    for example in test_data:
        example = json.loads(example)
        query = create_query(example['entity'])
        sampled_examples = sr.search(query, cutoff =50)
        sampled_examples = [json.loads(train_data[int(x['id'])]) for x in sampled_examples if x['text'] != '-9999']
        sampled_examples = filtering_examples(example['conv_id'], sampled_examples)

        new_example = create_new_example(example, sampled_examples, is_train=False)
        new_test_data.append(new_example)
    
    return new_train_data, new_valid_data, new_test_data
    



if __name__ == '__main__':

    k = 3
    train_rec_path = 'data/inspired/train_data_processed_retrieval.jsonl'
    valid_rec_path = 'data/inspired/valid_data_processed_retrieval.jsonl'
    test_rec_path = 'data/inspired/test_data_processed_retrieval.jsonl'

    # train_conv_path = 'data/redial/train_data_processed_retrieval.jsonl'
    # valid_conv_path = 'data/redial/valid_data_processed_retrieval.jsonl'
    # test_conv_path = 'data/redial/test_data_processed_retrieval.jsonl'

    train_rec_data = read_file(train_rec_path)
    valid_rec_data = read_file(valid_rec_path)
    test_rec_data = read_file(test_rec_path)

    new_train_rec_data, new_valid_rec_data, new_test_rec_data = random_retrieval(train_rec_data, valid_rec_data, test_rec_data, n = k)

    # assert len(new_train_conv_data) == len(train_conv_data)
    # assert len(new_valid_conv_data) == len(valid_conv_data)
    # assert len(new_test_conv_data) == len(test_conv_data)

    new_train_rec_path = 'data/inspired/train_data_processed_retrieval_random.jsonl'
    new_valid_rec_path = 'data/inspired/valid_data_processed_retrieval_random.jsonl'
    new_test_rec_path = 'data/inspired/test_data_processed_retrieval_random.jsonl'

    # new_train_rec_path = 'data/redial/train_data_processed_retrieval_bm25.jsonl'
    # new_valid_rec_path = 'data/redial/valid_data_processed_retrieval_bm25.jsonl'
    # new_test_rec_path = 'data/redial/test_data_processed_retrieval_bm25.jsonl'

    # write_file(new_train_conv_data, new_train_conv_path)
    # write_file(new_valid_conv_data, new_valid_conv_path)
    # write_file(new_test_conv_data, new_test_conv_path)

    # new_train_rec_data, new_valid_rec_data, new_test_rec_data = bm25_retrieval(train_rec_data, valid_rec_data, test_rec_data)

    write_file(new_train_rec_data, new_train_rec_path)
    write_file(new_valid_rec_data, new_valid_rec_path)
    write_file(new_test_rec_data, new_test_rec_path)
