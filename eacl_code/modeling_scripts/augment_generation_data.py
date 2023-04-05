"""
This augments the generation data with results retrieved by DPR
"""
import argparse
import os
import json
from pathlib import Path

import pandas as pd
import transformers as hft
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from eacl_code.models import CustomDPRModel
from eacl_code.utils import get_checkpoint_dir, get_data_dir


def main(args):
    data_dir = args.data_dir
    model_ckpt = args.model_ckpt
    cache_dir = args.cache_dir
    generation_data_dir = args.generation_data_dir

    meta = pd.read_csv(os.path.join(data_dir, "metadata.csv.zip"))

    train = pd.read_csv(os.path.join(generation_data_dir, f"train_{args.lang}.csv"))
    valid = pd.read_csv(os.path.join(generation_data_dir, f"valid_{args.lang}.csv"))
    test = pd.read_csv(os.path.join(generation_data_dir, f"test_{args.lang}.csv"))

    # Auto-select GPu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # cache encoded passages if it doesn't exist yet

    passage_cached_path = os.path.join(cache_dir, f"dpr_{args.ctx_colname}_passages_cached.pt")
    if not os.path.exists(passage_cached_path):
        os.makedirs(cache_dir, exist_ok=True)

        # load saved retriever
        CtxEncoderClass = hft.DPRContextEncoder if args.lang == "en" else hft.AutoModel
        
        ctx_encoder = (
            CtxEncoderClass.from_pretrained(
                os.path.join(model_ckpt, "ctx-encoder")
            )
            .to(device)
            .eval()
        )

        if args.lang == "en":
            dpr_name = "facebook/dpr-ctx_encoder-single-nq-base"
        else:
            dpr_name = "etalab-ia/dpr-ctx_encoder-fr_qa-camembert"

        ctx_tokenizer = hft.AutoTokenizer.from_pretrained(dpr_name)

        all_contexts = meta[args.ctx_colname].to_list()
        all_ctx_ids = ctx_tokenizer(
            all_contexts,
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )["input_ids"]

        ctx_loader = DataLoader(
            all_ctx_ids, batch_size=64, shuffle=False, num_workers=0
        )

        with torch.no_grad():
            P = torch.cat(
                [
                    ctx_encoder(docs.to(ctx_encoder.device))
                    .pooler_output
                    .cpu()
                    .detach()
                    for docs in tqdm(ctx_loader)
                ]
            )

        torch.save(P, passage_cached_path)

    if args.lang == "en":
        q_dpr_name = "facebook/dpr-question_encoder-single-nq-base"
        ctx_dpr_name = "facebook/dpr-ctx_encoder-single-nq-base"
    elif args.lang == "fr":
        ctx_dpr_name = "etalab-ia/dpr-ctx_encoder-fr_qa-camembert"
        q_dpr_name = "etalab-ia/dpr-question_encoder-fr_qa-camembert"

    model = CustomDPRModel(
        cache_path=passage_cached_path,
        q_enc_path=os.path.join(model_ckpt, "q-encoder"),
        q_dpr_name=q_dpr_name,
        ctx_dpr_name=ctx_dpr_name,        
    )

    meta = meta.set_index("pid")

    if args.lang == "en":
        t5_name = "t5-large"
    else:
        t5_name = "google/mt5-large"

    t5_tok = hft.AutoTokenizer.from_pretrained(t5_name)

    def retrieve_titles(df):
        source_augmented = []
        source_augmented_1 = []

        for source, source_processed in tqdm(df[["source", "source_processed"]].values):
            sep = f" {t5_tok.eos_token} "

            sorted_indices = model.retrieve_table_indices(json.loads(source), k=5)
            if args.lang == "fr":
                title = "title_fr"
            else:
                title = "title"
            
            retrieved_titles = meta.iloc[sorted_indices][title].values
            title_text = sep.join(retrieved_titles)

            source_augmented.append(source_processed + sep + title_text)
            source_augmented_1.append(source_processed + sep + title_text[0])

        return source_augmented, source_augmented_1

    train_retrieved, train_retrieved_1 = retrieve_titles(train)
    valid_retrieved, valid_retrieved_1 = retrieve_titles(valid)
    test_retrieved, test_retrieved_1 = retrieve_titles(test)

    train["source_augmented"] = train_retrieved
    valid["source_augmented"] = valid_retrieved
    test["source_augmented"] = test_retrieved

    train["source_augmented_1"] = train_retrieved_1
    valid["source_augmented_1"] = valid_retrieved_1
    test["source_augmented_1"] = test_retrieved_1

    train.to_csv(
        os.path.join(generation_data_dir, f"train_{args.lang}_augmented.csv.zip")
    )
    valid.to_csv(
        os.path.join(generation_data_dir, f"valid_{args.lang}_augmented.csv.zip")
    )
    test.to_csv(
        os.path.join(generation_data_dir, f"test_{args.lang}_augmented.csv.zip")
    )


if __name__ == "__main__":
    data_dir = get_data_dir()
    checkpoint_dir = get_checkpoint_dir()

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Augments the generation data with results retrieved by DPR",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=data_dir / "retrieval",
        help="Path to the data directory for retrieval task",
    )
    parser.add_argument(
        "--model_ckpt",
        type=str,
        default=checkpoint_dir / "statscan-dpr" / "unique-star-3994" / "epoch-29",
        # For french, use: checkpoint_dir / "statscan-dpr" / "eager-forest-4035" / "epoch-29"
        help="Path to the model checkpoint. Default value is the best performing model.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=data_dir / "modeling" / "cache",
        help="Path to the cache directory",
    )
    parser.add_argument(
        "--generation_data_dir",
        type=str,
        default=data_dir / "generation/",
        help="Path to the generation data directory",
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
        default="basic_and_member",
        type=str,
        help="Column name of the context to be used for DPR",
    )

    args = parser.parse_args()
    main(args)
