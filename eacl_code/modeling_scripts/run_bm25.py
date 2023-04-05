import argparse
import os
import itertools

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer

from eacl_code.models.bm25 import BM25Retriever
from eacl_code.utils import get_data_dir


def main(args):
    data_dir = args.data_dir
    lang = args.lang

    # cur_dir = os.path.dirname(os.path.abspath(__file__))
    train = pd.read_csv(os.path.join(data_dir, f"train_{lang}.csv"))
    valid = pd.read_csv(os.path.join(data_dir, f"valid_{lang}.csv"))
    test = pd.read_csv(os.path.join(data_dir, f"test_{lang}.csv"))
    meta = pd.read_csv(os.path.join(data_dir, "metadata.csv.zip"))

    all_dataframes = {"valid": valid, "test": test}

    all_tokenizers = {"sklearn": TfidfVectorizer().build_tokenizer()}

    passages_cols = [
        "basic_info",
        "basic_and_member",
        "basic_and_footnote",
        "title",
        "title_and_member",
        "title_and_footnote",
        "full_info",
    ]
    splits = list(all_dataframes.keys())
    tokenizers = list(all_tokenizers.keys())

    records = []

    for passage_type, split, tokenizer in tqdm(
        itertools.product(passages_cols, splits, tokenizers)
    ):
        df = all_dataframes[split]
        queries_train = train["conversation_processed"].to_list()
        queries = df["conversation_processed"].to_list()
        passages = meta[passage_type].to_list()

        all_labels = meta["pid"].values
        target_labels = df["target_pid"].values[..., np.newaxis]

        ret = BM25Retriever(tokenizer=all_tokenizers[tokenizer])

        ret.build_vocab(queries_train + passages)

        queries_enc = ret.encode_queries(queries)
        passages_enc = ret.encode_passages(passages)

        indices, scores = ret.retrieve(queries_enc, passages_enc, k=20, verbose=True)

        k_list = [1, 5, 10, 20]

        for k in k_list:
            retrieved_labels = all_labels[indices[:, :k]]
            matches = np.any(target_labels == retrieved_labels, axis=1)
            acc = matches.mean()

            records.append(
                {
                    "acc": acc,
                    "split": split,
                    "passages": passage_type,
                    "k": k,
                    "tokenizer": tokenizer,
                }
            )

    records_df = pd.DataFrame.from_records(records)
    print(records_df.head())

    os.makedirs(args.output_dir, exist_ok=True)
    records_df.to_csv(
        os.path.join(args.output_dir, f"bm25_results_{lang}.csv"), index=False
    )


if __name__ == "__main__":
    data_dir = get_data_dir()

    parser = argparse.ArgumentParser(
        description="Runs BM25 retrieval",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=data_dir / "retrieval",
        help="Path to data directory containing train, valid, test, and metadata.csv.zip",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=data_dir / "results",
        help="Path to output directory for results",
    )
    parser.add_argument(
        "--lang",
        default="en",
        type=str,
        choices=["en", "fr"],
        help="Language of data. The file will be determined based on this.",
    )

    args = parser.parse_args()
    main(args)
