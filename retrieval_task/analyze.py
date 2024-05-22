# load retrieval_task/data_out/*.csv into a pandas dataframe, get the average character length for each split
from collections import defaultdict
import pandas as pd

splits = [
    'train', 'dev', 'test', 'corpus'
]

avg_char_len = defaultdict(dict)
counts = defaultdict(dict)

for lang in ['english', 'french']:
    for split in splits:
        df = pd.read_csv(f'retrieval_task/data_out/{lang}/{split}.csv')
        if split == 'corpus':
            col = 'doc'
        else:
            col = 'query'
        avg_len = df[col].str.len().mean()
        print(f'Average character length for {split}: {avg_len} ({lang})')
        avg_char_len[lang][split] = round(avg_len, 2)
        counts[lang][split] = len(df)

# save as json
import json
with open('retrieval_task/average_character_length.json', 'w') as f:
    json.dump(avg_char_len, f)

with open('retrieval_task/counts.json', 'w') as f:
    json.dump(counts, f)