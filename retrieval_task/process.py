import json
import os
from pathlib import Path

import pandas as pd
from datasets import load_dataset
import statcan_dialogue_dataset as sdd

def convert_to_chat_records(conversation: list):
    # convert to {'role': <actor>, 'content': <text>} format similar to huggingface
    conversation = json.loads(conversation)
    conv_recs = []

    for c in conversation:
        conv_recs.append({
            'role': c['actor'],
            'content': c['text']
        })
    
    return json.dumps(conv_recs)

language_to_data_dir = {
    'english': 'retrieval',
    'french': 'retrieval_fr'
}
language_to_doc_col_name = {
    'french': 'basic_and_member_fr',
    'english': 'basic_and_member',
}

language_to_title_col_name = {
    'french': 'title_fr',
    'english': 'title',
}

# see readme.md for how to use this download zip file and extract, or uncomment the following lines:
# saved_path = sdd.download_huggingface(api_token=os.environ['HF_TOKEN'], data_dir="./")
# sdd.extract_task_data_zip(saved_path, remove_zip=True, load_dir='./', data_dir="./retrieval_task/data_original")

metadata = pd.read_csv('retrieval_task/data_original/retrieval/metadata.csv.zip')

for lang in ['english', 'french']:
    ds_ret = load_dataset("McGill-NLP/statcan-dialogue-dataset", data_dir=language_to_data_dir[lang])
    for split in ['train', 'test', 'validation']:
        data = ds_ret[split]
        data = pd.DataFrame(data)

        queries = {}

        queries['query'] = list(map(convert_to_chat_records, data['conversation']))
        queries['query_id'] = [f"Q{i}" for i in data['conversation_index']]
        queries['doc_id'] = [f"D{i}" for i in data['target_pid']]


        queries_df = pd.DataFrame(queries)
        # rename validation to dev to match with the other datasets
        if split == 'validation':
            split = 'dev'
        
        # save as data/<language>/<split>.csv
        save_dir = Path(f"retrieval_task/data_out/{lang}")
        save_dir.mkdir(parents=True, exist_ok=True)
        queries_df.to_csv(f"retrieval_task/data_out/{lang}/{split}.csv", index=False)

    corpus = {}
    corpus['doc_id'] = [f"D{i}" for i in metadata['pid']]
    corpus['title'] = metadata[language_to_title_col_name[lang]]
    corpus['doc'] = metadata[language_to_doc_col_name[lang]]
    # save corpus as data/<language>/corpus.csv
    corpus_df = pd.DataFrame(corpus)
    corpus_df.to_csv(f"retrieval_task/data_out/{lang}/corpus.csv", index=False)
