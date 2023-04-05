"""
This script is used to cache DPR index for the DPR demo app in apps/dpr_demo.py
"""
import argparse
import os
import transformers
import torch
from functools import partial
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader
import transformers
from tqdm.auto import tqdm

from eacl_code.utils import get_data_dir


def main(args):
    data_dir = args.data_dir
    apps_cache_dir = args.apps_cache_dir
    lang = args.lang

    TOKEN = os.getenv(args.token_env_var)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ctx_encoder = transformers.DPRContextEncoder.from_pretrained(
        args.pretrained_path, use_auth_token=TOKEN
    ).to(device)
    ctx_encoder.eval()

    ctx_tokenizer = transformers.DPRContextEncoderTokenizerFast.from_pretrained(
        args.tokenizer_path
    )
    ctx_tokenize = partial(
        ctx_tokenizer, padding="max_length", truncation=True, max_length=256
    )

    meta = pd.read_csv(os.path.join(data_dir, "metadata.csv.zip"))
    all_contexts = meta[args.ctx_colname].to_list()
    all_ctx_ids = ctx_tokenize(all_contexts, return_tensors="pt")["input_ids"]

    ctx_loader = DataLoader(all_ctx_ids, batch_size=64, shuffle=False, num_workers=0)

    with torch.no_grad():
        P = torch.cat(
            [
                ctx_encoder(docs.to(ctx_encoder.device)).pooler_output.cpu().detach()
                for docs in tqdm(ctx_loader)
            ]
        )

    torch.save(P, os.path.join(apps_cache_dir, "dpr_passages_cached.pt"))


if __name__ == "__main__":
    data_dir = get_data_dir()
    parser = argparse.ArgumentParser(
        description="Cache DPR index for the DPR demo app in apps/dpr_demo.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=data_dir / "retrieval",
        help="Path to the data directory containing the CSV files.",
    )
    parser.add_argument(
        "--apps_cache_dir",
        type=str,
        default=data_dir / "apps/dpr_demo/cache",
        help="Path to the directory where the cached DPR passages will be stored.",
    )
    parser.add_argument(
        "--lang",
        default="en",
        type=str,
        choices=["en", "fr"],
        help="Language of data. The file will be determined based on this.",
    )
    parser.add_argument(
        "--ctx_colname",
        type=str,
        default="basic_info",
        help="Column name of the context text (table metadata).",
    )
    parser.add_argument(
        "--pretrained_path",
        type=str,
        required=True,
        help="Path (or huggingface repo name) to the DPR pretrained model.",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="facebook/dpr-ctx_encoder-single-nq-base",
        help="Path (or huggingface repo name) to the DPR tokenizer.",
    )
    parser.add_argument(
        "--token_env_var",
        type=str,
        default="HUGGINGFACE_API_TOKEN",
        help="Environment variable containing the auth token for the DPR model.",
    )
    args = parser.parse_args()
    main(args)
