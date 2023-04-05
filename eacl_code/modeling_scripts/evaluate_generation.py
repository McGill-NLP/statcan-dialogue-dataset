import argparse
from collections import defaultdict
from functools import partial
import json
import os
import re
from collections import OrderedDict
from urllib.parse import urlparse, parse_qs

import pandas as pd
import torch
import transformers as hft
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from datasets import load_metric
import nltk
import numpy as np

from eacl_code.models import moverscore
from eacl_code.utils import find_urls, encode_right_truncated_old, get_checkpoint_dir, get_data_dir


def get_pid(target):
    urls = find_urls(target)
    if len(urls) < 1:
        return None
    url = urls[0]
    if "action?pid=" not in url:
        return None
    parsed = parse_qs(urlparse(url).query)
    if "pid" not in parsed:
        return None
    pid = parsed["pid"][0]
    if len(pid) < 8:
        return None
    if pid.startswith("http"):
        return None

    return pid[:8]


def run_generation(args, source_text, target_text, ckpt_load_dir):
    # Load model
    torch.manual_seed(args.random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.lang == "en":
        ConditionaGenerationClass = hft.T5ForConditionalGeneration
    else:
        ConditionaGenerationClass = hft.MT5ForConditionalGeneration

    model = (
        ConditionaGenerationClass.from_pretrained(
            os.path.join(ckpt_load_dir, "model")
        )
        .to(device)
        .eval()
    )
    tokenizer = hft.AutoTokenizer.from_pretrained(args.model)

    tokenize_target = partial(
        tokenizer,
        max_length=args.max_target_length,
        truncation=True,
    )

    # when generating, we will use the logits of right-most token to predict the next token
    # so the padding should be on the left
    # source: https://huggingface.co/docs/transformers/model_doc/t5
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token  # to avoid an error

    source_ids, attention_masks = encode_right_truncated_old(
        tokenizer, source_text, args.max_source_length
    )

    target_ids = tokenize_target(target_text)["input_ids"]

    loader = DataLoader(
        list(zip(source_ids, attention_masks)),
        batch_size=args.batch_size,
        shuffle=False,
    )

    generated_ids = []

    for source_ids, attention_mask in tqdm(loader, disable=not args.show_progress):
        g = model.generate(
            input_ids=source_ids.to(model.device),
            attention_mask=attention_mask.to(model.device),
            max_length=args.max_target_length,
            num_beams=args.num_beams,
            length_penalty=args.length_penalty,
        )
        generated_ids.extend(g.tolist())

    generated_text = [
        tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids
    ]
    generated_ids = [
        [t for t in sample if t != 0] for sample in generated_ids
    ]  # strip padding

    return generated_ids, target_ids, generated_text


def parse_args():
    data_dir = get_data_dir()
    checkpoint_dir = get_checkpoint_dir()
    
    parser = argparse.ArgumentParser(
        description="Evaluate a generation model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default=data_dir,
        help="Directory containing the dataset.",
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        default=checkpoint_dir,
        help="Directory to store the model checkpoints.",
    )

    parser.add_argument(
        "--show_progress",
        type=bool,
        default=True,
        help="Show progress bar while generating.",
    )

    parser.add_argument(
        "--max_source_length",
        type=int,
        default=512,
        help="Maximum length of the source sequence.",
    )

    parser.add_argument(
        "--max_target_length",
        type=int,
        default=256,
        help="Maximum length of the target sequence.",
    )

    parser.add_argument(
        "--wandb_name", 
        type=str, 
        help="Name of the experiment in wandb."
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        help="Name of the project in wandb.",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=9,
        help="Epoch to load the model from.",
    )
    parser.add_argument(
        "--source_colname",
        type=str,
        default="source_processed",
        help="Column name of the source sequences (conversation generated so far).",
    )
    parser.add_argument(
        "--target_colname",
        type=str,
        default="target_processed",
        help="Column name of the target sequences (message to generate).",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for evaluation."
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=4,
        help="Number of beams for beam search decoding.",
    )
    parser.add_argument(
        "--length_penalty",
        type=float,
        default=0.6,
        help="Length penalty for beam search.",
    )
    parser.add_argument(
        "--no_wandb_log",
        action="store_false",
        dest="wandb_log",
        help="Whether to log to wandb.",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="valid",
        help="Which split (train, valid, test) to evaluate on.",
    )

    parser.add_argument(
        "--lang",
        default="en",
        type=str,
        choices=["en", "fr"],
        help="Language of data. The file will be determined based on this.",
    )
    parser.add_argument('--overwrite-cache', dest='overwrite_cache', default=False, action='store_true')
    
    parser.set_defaults(wandb_log=True)

    args = parser.parse_args()

    return args


def main(args):
    # Download nltk package
    nltk.download("omw-1.4")

    print("Args:", args)

    if args.lang == "en":
        title = "title"
    else:
        title = "title_" + args.lang

    ckpt_load_dir = os.path.join(
        args.model_dir, args.wandb_project, args.wandb_name, f"epoch-{args.epoch}"
    )

    # Get the name of the file we want
    save_name = f"generated_text--split-{args.split}--source_colname-{args.source_colname}--num_beams-{args.num_beams}--length_penalty-{args.length_penalty}.csv"
    fname = os.path.join(ckpt_load_dir, save_name)

    if os.path.exists(fname) and args.overwrite_cache is False:
        print("File already exists. Skipping generation and loading from disk.")
        df = pd.read_csv(fname)
    else:
        print("Generating data...")
        # Load data
        df = pd.read_csv(
            os.path.join(args.data_dir, "generation", f"{args.split}_{args.lang}_augmented.csv.zip")
        )
        source_text = df[args.source_colname].to_list()
        target_text = df[args.target_colname].to_list()
        generated_ids, target_ids, generated_text = run_generation(
            args, source_text, target_text, ckpt_load_dir
        )
        df["generated_ids"] = generated_ids
        df["generated"] = generated_text
        df["target_ids"] = target_ids

        df.to_csv(fname, index=False)

    # Start evaluation here
    var_args = vars(args)


    # Retrieval accuracy
    print("Computing retrieval accuracy...")
    meta = pd.read_csv(os.path.join(args.data_dir, "retrieval", "metadata.csv.zip"))
    pid_to_title = meta.set_index("pid")[title].to_dict()

    df["pid"] = df["target"].apply(get_pid)
    df_filt = df[df["pid"].notnull()].copy()  # None are targets without pids


    df_filt["title"] = df_filt["pid"].apply(lambda pid: pid_to_title.get(int(pid)))
    df_filt = df_filt.dropna()
    aug_mask = df_filt.apply(lambda row: row["title"] in row["source_augmented"], axis=1)
    tgt_mask = df_filt.apply(lambda row: row["title"] in row["target"], axis=1)

    num = (
        df_filt[tgt_mask].apply(lambda row: row["title"] in row["generated"], axis=1).sum()
    )
    denom = df_filt[tgt_mask].shape[0]
    var_args["title_accuracy"] = num / denom
    
    num_aug = (
        df_filt[aug_mask & tgt_mask]
        .apply(lambda row: row["title"] in row["generated"], axis=1)
        .sum()
    )
    denom_aug = df_filt[aug_mask & tgt_mask].shape[0]
    var_args["rerank_accuracy"] = num_aug / denom_aug
    
    # predictions and references for metrics
    predictions = df["generated"].tolist()
    references = df[["target_processed"]].values.tolist()
    references_flat = [df["target_processed"].values.tolist()]
    
    # MoverScore
    print("Computing MoverScore...")
    model_name = 'camembert-base' if args.lang == 'fr' else 'bert-base-cased'
    wordmover = moverscore.load_metric(model_name)
    scores = wordmover(references=references_flat, predictions=predictions)
    var_args['moverscore'] = scores
    

    # BERT Score
    print("Computing BERT score...")
    bertscore = load_metric('bertscore')
    res = bertscore.compute(predictions=predictions, references=references, lang=args.lang)
    for key in ['precision', 'recall', 'f1']:
        var_args[f'bert-score-{key}'] = np.mean(res[key])

    # Sacrebleu
    print("Computing sacrebleu...")
    sacrebleu = load_metric("sacrebleu")
    res = sacrebleu.compute(predictions=predictions, references=references)
    var_args["sacrebleu"] = res["score"]

    # Rouge
    print("Computing Rouge...")
    rouge = load_metric("rouge")
    result = rouge.compute(predictions=predictions, references=references)
    for key in result:
        for bound in ["low", "mid", "high"]:
            var_args[f"{key}-{bound}"] = getattr(result[key], bound).fmeasure

    # Meteor
    print("Computing meteor...")
    meteor = load_metric("meteor")
    res = meteor.compute(predictions=predictions, references=references)
    var_args["meteor"] = res["meteor"]

    results_name = f"eval_results--split-{args.split}--source_colname-{args.source_colname}--num_beams-{args.num_beams}--length_penalty-{args.length_penalty}.json"
    json.dump(var_args, open(os.path.join(ckpt_load_dir, results_name), "w"))

    print("=" * 80)
    print("FINAL RESULTS:")
    print("-" * 80)
    for k, v in var_args.items():
        print(f"{k}: {v}")


    # Log into wandb
    if args.wandb_log:
        import wandb

        wandb.init(project=args.wandb_project + "-results", name=args.wandb_name)
        wandb.log(var_args)
        wandb.finish()

if __name__ == "__main__":
    args = parse_args()

    # Hardcoded
    args.model = "t5-large" if args.lang == "en" else "google/mt5-large"

    
    main(args)