import json

from tqdm import tqdm

with open('/root/autodl-tmp/STEP-main/data/inspired/entity2id.json', encoding='utf-8') as f:
    entity2id = json.load(f)


def remove(src_file, tgt_file):
    tgt = open(tgt_file, 'w', encoding='utf-8')
    with open(src_file, encoding='utf-8') as f:
        for line in tqdm(f):
            line = json.loads(line)
            for i, message in enumerate(line):
                new_entity, new_entity_name = [], []
                for j, entity in enumerate(message['entity_STEP']):
                    if entity in entity2id:
                        new_entity.append(entity)
                        new_entity_name.append(message['entity_name'][j])
                line[i]['entity_STEP'] = new_entity
                line[i]['entity_name'] = new_entity_name

                new_movie, new_movie_name = [], []
                for j, movie in enumerate(message['movie_STEP']):
                    if movie in entity2id:
                        new_movie.append(movie)
                        new_movie_name.append(message['movie_name'][j])
                line[i]['movie_STEP'] = new_movie
                line[i]['movie_name'] = new_movie_name

            tgt.write(json.dumps(line, ensure_ascii=False) + '\n')
    tgt.close()


src_files = ['/root/autodl-tmp/STEP-main/data/inspired/test.jsonl', '/root/autodl-tmp/STEP-main/data/inspired/dev.jsonl', '/root/autodl-tmp/STEP-main/data/inspired/train.jsonl']
tgt_files = ['/root/autodl-tmp/STEP-main/data/inspired/test_data_dbpedia.jsonl', '/root/autodl-tmp/STEP-main/data/inspired/valid_data_dbpedia.jsonl', '/root/autodl-tmp/STEP-main/data/inspired/train_data_dbpedia.jsonl']
for src_file, tgt_file in zip(src_files, tgt_files):
    remove(src_file, tgt_file)
