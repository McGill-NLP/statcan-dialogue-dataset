import os
import json

import pandas as pd
import transformers
from tqdm import tqdm

from eacl_code.utils import get_raw_data_dir, get_large_data_dir, get_data_dir

data_dir = get_large_data_dir() / 'tables'
raw_data_dir = get_raw_data_dir()
retrieval_dir = get_data_dir() / 'retrieval'

if __name__ == '__main__':
    cube_list = json.load(open(os.path.join(raw_data_dir, 'cube_list.json')))
    pid_to_title = {str(cube['productId']): cube['cubeTitleEn'] for cube in cube_list}

    tokenizer = transformers.TapasTokenizer.from_pretrained('google/tapas-base')
    tapas_tokens_path = os.path.join(retrieval_dir, 'tapas_tokens.json')

    if os.path.exists(tapas_tokens_path):
        tapas_tokens = json.load(open(tapas_tokens_path))
    else:
        tapas_tokens = {}

    for c in tqdm(cube_list):
        pid = str(c['productId'])
        title = c['cubeTitleEn']

        if pid in tapas_tokens:
            continue
        
        table_path = os.path.join(data_dir, f'{pid}.csv.zip')

        if not os.path.exists(table_path):
            print(f"Table {pid} does not exist, skipping.")
            continue

        df = pd.read_csv(table_path, nrows=250, error_bad_lines=False, index_col=False).astype(str)

        input_ids = tokenizer(table=df, queries=title, padding='max_length', truncation=True)['input_ids']
        tapas_tokens[pid] = input_ids

        json.dump(tapas_tokens, open(tapas_tokens_path, 'w'))