import argparse
import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from eacl_code.models.bm25 import BM25Retriever
from eacl_code.utils import get_data_dir


def generate_hard_negatives(target_pids, retrieved_pids):
    hard_negs = []

    for retrieved, target in zip(retrieved_pids, target_pids):
        if retrieved[0] == target:
            ix = 1
        else:
            ix = 0
        hard_negs.append(retrieved[ix])

    return hard_negs


def compute_indices(queries, passages):
    tokenizer = TfidfVectorizer().build_tokenizer()
    ret = BM25Retriever(tokenizer=tokenizer)
    ret.build_vocab(passages + queries)

    queries_enc = ret.encode_queries(queries)
    passages_enc = ret.encode_passages(passages)

    indices, _ = ret.retrieve(queries_enc, passages_enc, k=2, verbose=True)

    return indices


def main(args):
    data_dir = args.data_dir
    lang = args.lang
    cols = args.cols

    # Load data
    train = pd.read_csv(os.path.join(data_dir, f"train_{lang}.csv"))
    meta = pd.read_csv(os.path.join(data_dir, "metadata.csv.zip"))

    # Convert pd.Series -> list
    all_labels = meta["pid"].values
    queries_train = train["conversation_processed"].to_list()

    for col in cols.split(","):
        passages = meta[col].to_list()

        # Create tokenizer and models
        title_indices = compute_indices(queries=queries_train, passages=passages)

        # Create hard negatives using the retrieved pids
        target_pids = train["target_pid"].values

        train[f"hard_negative_pid_{col}"] = generate_hard_negatives(
            target_pids, retrieved_pids=all_labels[title_indices]
        )

    train.to_csv(os.path.join(data_dir, f"train_{lang}_bm25.csv"), index=False)


if __name__ == "__main__":
    data_dir = get_data_dir()

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="add hard negative examples to train data via BM25 retrieval.",
    )
    parser.add_argument(
        "--data_dir",
        default=data_dir / "retrieval/",
        help="The directory containing the data.",
    )
    parser.add_argument(
        "--lang",
        default="en",
        type=str,
        choices=["en", "fr"],
        help="Language of data. The file will be determined based on this.",
    )
    parser.add_argument(
        "--cols",
        default="title,basic_info,full_info",
        type=str,
        help="Columns to use, separated by commas with no space.",
    )
    args = parser.parse_args()
    main(args)
