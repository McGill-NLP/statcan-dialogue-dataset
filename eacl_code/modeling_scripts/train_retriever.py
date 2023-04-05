import argparse
from functools import partial
import os
import json

import pandas as pd
import torch
from torch.utils.data import DataLoader
import transformers as hft
from tqdm.auto import tqdm
import wandb

from eacl_code.models import dpr, tapas
from eacl_code.utils import get_data_dir, get_checkpoint_dir


def main(args):
    # Define controllable variables
    data_dir = args.data_dir
    model_dir = args.model_dir
    show_progress = args.show_progress
    show_eval = args.show_eval
    ctx_colname = args.ctx_colname
    q_colname = args.q_colname
    lang = args.lang

    batch_size = args.batch_size
    n_epochs = args.n_epochs
    qmaxlen = args.qmaxlen
    cmaxlen = args.cmaxlen
    lr = args.lr
    weight_decay = args.weight_decay
    neg_col = args.hn_colname

    # Non-controllable variables
    top_k_list = [1, 5, 10, 20, 30]

    # Load dataframes
    train = pd.read_csv(os.path.join(data_dir, f"train_{lang}_bm25.csv"))
    valid = pd.read_csv(os.path.join(data_dir, f"valid_{lang}.csv"))
    test = pd.read_csv(os.path.join(data_dir, f"test_{lang}.csv"))

    meta = pd.read_csv(os.path.join(data_dir, "metadata.csv.zip"))

    meta_dict = meta.set_index("pid").to_dict()
    all_pids = meta["pid"].values

    # Load table tokens for tapas, convert PIDs to integers
    tapas_tokens = json.load(open(os.path.join(data_dir, "tapas_tokens.json")))
    tapas_tokens = {int(k): v[:cmaxlen] for k, v in tapas_tokens.items()}

    # Set up wandb
    wandb.init(
        project=args.wandb_project,
        entity="statscan-nlp",
        name=args.wandb_name,
        reinit=True,
    )
    wandb_config = vars(args)
    wandb_config["top_k_list"] = top_k_list
    wandb_config["name"] = wandb.run.name
    print("Training Configuration:")
    print(wandb_config)
    wandb.config.update(wandb_config)

    # Load tokenizer
    if lang == "en":
        ctx_dpr_name = "facebook/dpr-ctx_encoder-single-nq-base"
        q_dpr_name = "facebook/dpr-question_encoder-single-nq-base"
    elif lang == "fr":
        ctx_dpr_name = "etalab-ia/dpr-ctx_encoder-fr_qa-camembert"
        q_dpr_name = "etalab-ia/dpr-question_encoder-fr_qa-camembert"

    ctx_tokenizer = hft.AutoTokenizer.from_pretrained(ctx_dpr_name)
    q_tokenizer = hft.AutoTokenizer.from_pretrained(q_dpr_name)

    ctx_tokenize = partial(
        ctx_tokenizer, padding="max_length", truncation=True, max_length=cmaxlen
    )

    train_queries_ids, train_queries_masks = dpr.process_queries(
        q_tokenizer, df=train, max_length=qmaxlen, q_col=q_colname
    )

    # Create the data loaders
    if args.tokenization == "dpr":
        (
            train_pos_ids,
            train_pos_masks,
            train_neg_ids,
            train_neg_masks,
        ) = dpr.process_ctx(
            ctx_tokenize, df=train, neg_col=neg_col, pid_dict=meta_dict[ctx_colname]
        )
    else:
        (
            train_pos_ids,
            train_pos_masks,
            train_neg_ids,
            train_neg_masks,
        ) = tapas.process_ctx(df=train, neg_col=neg_col, pid_dict=tapas_tokens)

    train_loader = DataLoader(
        list(
            zip(
                train_queries_ids,
                train_queries_masks,
                train_pos_ids,
                train_pos_masks,
                train_neg_ids,
                train_neg_masks,
            )
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    train_query_only_loader = DataLoader(
        list(zip(train_queries_ids, train_queries_masks)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    valid_queries_ids, valid_queries_masks = dpr.process_queries(
        q_tokenizer, df=valid, max_length=qmaxlen, q_col=q_colname
    )
    valid_loader = DataLoader(
        list(zip(valid_queries_ids, valid_queries_masks)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    test_queries_ids, test_queries_masks = dpr.process_queries(
        q_tokenizer, df=test, max_length=qmaxlen, q_col=q_colname
    )
    test_loader = DataLoader(
        list(zip(test_queries_ids, test_queries_masks)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    if args.tokenization == "dpr":
        all_contexts = meta[ctx_colname].to_list()
        tokenized = ctx_tokenize(all_contexts, return_tensors="pt")
        all_ctx_ids = tokenized["input_ids"]
        all_ctx_masks = tokenized["attention_mask"]
    elif args.tokenization == "tapas":
        all_ctx_ids = torch.tensor([tapas_tokens[pid] for pid in all_pids])
        all_ctx_masks = torch.ones_like(all_ctx_ids)

    ctx_loader = DataLoader(
        list(zip(all_ctx_ids, all_ctx_masks)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Load the models and optimizers
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model == "dpr":
        # Caveat: DPR encoder does not support AutoModel
        CtxEncoderClass = hft.DPRContextEncoder if lang == "en" else hft.AutoModel
        
        ctx_encoder = CtxEncoderClass.from_pretrained(ctx_dpr_name).to(device)
        q_encoder = hft.AutoModel.from_pretrained(q_dpr_name).to(device)
        
        if args.gradient_checkpointing:
            dpr.enable_gradient_checkpointing(ctx_encoder)
            dpr.enable_gradient_checkpointing(q_encoder)

    elif args.model.startswith("tapas"):
        if args.model == "tapas":
            ctx_name = q_name = "tapas-base"

        elif args.model == "tapas-nq-medium":
            ctx_name = "xhluca/tapas-nq-hn-retriever-medium-0"
            q_name = "xhluca/tapas-nq-hn-retriever-medium-1"

        elif args.model == "tapas-nq-large":
            ctx_name = "xhluca/tapas-nq-hn-retriever-large-0"
            q_name = "xhluca/tapas-nq-hn-retriever-large-1"

        elif args.model == "tapas-nq-medium-reversed":
            ctx_name = "xhluca/tapas-nq-hn-retriever-medium-1"
            q_name = "xhluca/tapas-nq-hn-retriever-medium-0"

        elif args.model == "tapas-nq-large-reversed":
            ctx_name = "xhluca/tapas-nq-hn-retriever-large-1"
            q_name = "xhluca/tapas-nq-hn-retriever-large-0"

        else:
            raise ValueError(f"Model {args.model} not supported")

        ctx_encoder = tapas.TapasRetriever(
            hft.TapasModel.from_pretrained(ctx_name),
            gradient_checkpointing=args.gradient_checkpointing,
        ).to(device)
        q_encoder = tapas.TapasRetriever(
            hft.TapasModel.from_pretrained(q_name),
            gradient_checkpointing=args.gradient_checkpointing,
        ).to(device)

    else:
        raise ValueError(f"Model {args.model} not supported")

    optimizer = torch.optim.AdamW(
        [{"params": ctx_encoder.parameters()}, {"params": q_encoder.parameters()}],
        lr=lr,
        weight_decay=weight_decay,
    )

    ckpt_save_dir = os.path.join(model_dir, wandb.run.project, wandb.run.name)
    os.makedirs(ckpt_save_dir, exist_ok=True)

    for epoch in range(n_epochs):
        total_loss = 0.0
        step = 0

        ctx_encoder.train()
        q_encoder.train()

        pbar = tqdm(train_loader, disable=not show_progress)
        for query_ids, query_masks, pos_ids, pos_masks, neg_ids, neg_masks in pbar:
            loss = dpr.train_step(
                ctx_encoder,
                q_encoder,
                optimizer,
                query_ids,
                query_masks,
                pos_ids,
                pos_masks,
                neg_ids,
                neg_masks,
            )

            total_loss += loss.item()
            step += 1
            running_loss = total_loss / step

            pbar.set_postfix({"running_loss": running_loss})

            if step % 20 == 1:
                wandb.log({"running_loss": running_loss, "epoch": epoch, "step": step})

        # Save the model checkpoints and tokenizers
        ctx_encoder.save_pretrained(
            os.path.join(ckpt_save_dir, f"epoch-{epoch}", "ctx-encoder")
        )
        q_encoder.save_pretrained(
            os.path.join(ckpt_save_dir, f"epoch-{epoch}", "q-encoder")
        )

        ctx_tokenizer.save_pretrained(
            os.path.join(ckpt_save_dir, f"epoch-{epoch}", "ctx-encoder")
        )
        q_tokenizer.save_pretrained(
            os.path.join(ckpt_save_dir, f"epoch-{epoch}", "q-encoder")
        )

        # Generate the context embeddings
        ctx_encoder.eval()
        with torch.no_grad():
            P = dpr.batched_encode(ctx_encoder, ctx_loader, verbose=show_eval)
        torch.save(
            P, os.path.join(ckpt_save_dir, f"epoch-{epoch}", "dpr_passages_cached.pt")
        )

        # Evaluate top-k accuracy
        train_results = dpr.evaluate(
            net=q_encoder,
            loader=train_query_only_loader,
            P=P,
            all_pids=all_pids,
            target=train["target_pid"].values,
            top_k_list=top_k_list,
            prefix="train_",
            show_eval=show_eval,
        )

        valid_results = dpr.evaluate(
            net=q_encoder,
            loader=valid_loader,
            P=P,
            all_pids=all_pids,
            target=valid["target_pid"].values,
            top_k_list=top_k_list,
            prefix="valid_",
            show_eval=show_eval,
        )

        results = {**train_results, **valid_results, "epoch": epoch}

        wandb.log(results)

    if n_epochs == 0:
        with torch.no_grad():
            P = dpr.batched_encode(ctx_encoder, ctx_loader, verbose=show_eval)

    # Last epoch, run on test set
    test_results = dpr.evaluate(
        net=q_encoder,
        loader=test_loader,
        P=P,
        all_pids=all_pids,
        target=test["target_pid"].values,
        top_k_list=top_k_list,
        prefix="test_",
        show_eval=show_eval,
    )

    wandb.log(test_results)

    test_acc_table = test_results["test_acc_table"]
    for _, row in test_acc_table.iterrows():
        wandb.log({"k": row["k"], "acc": row["acc"]})

    # terminate the job
    wandb.finish()


if __name__ == "__main__":
    data_dir = get_data_dir()
    checkpoint_dir = get_checkpoint_dir()
    
    parser = argparse.ArgumentParser(
        description="Trains a retriever model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=data_dir / "retrieval",
        help="Directory containing the dataset.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=checkpoint_dir,
        help="Directory to store the model checkpoints.",
    )
    parser.add_argument(
        "--ctx_colname",
        type=str,
        default="title",
        help="Column name of the context text (table metadata). This is ignored if --model=tapas or --model=tapas-nq since the context is always the table tokens.",
    )
    parser.add_argument(
        "--q_colname",
        type=str,
        default="conversation_processed",
        help="Column name of the query text (conversations).",
    )
    parser.add_argument(
        "--hn_colname",
        type=str,
        default="hard_negative_pid_title",
        help="Column name of the hard negative PIDs.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Number of samples in a batch."
    )
    parser.add_argument(
        "--n_epochs", type=int, default=30, help="Number of epochs to train the model."
    )
    parser.add_argument(
        "--qmaxlen", type=int, default=512, help="Maximum sequence length of queries."
    )
    parser.add_argument(
        "--cmaxlen",
        type=int,
        default=128,
        help="Maximum sequence length of contexts (table metadata).",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-5, help="Learning rate of AdamW optimizer."
    )
    parser.add_argument(
        "--wandb_name", type=str, help="Name of the experiment in wandb."
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        help="Name of the project in wandb.",
        required=True,
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Weight decay of AdamW optimizer.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="dpr",
        choices=[
            "dpr",
            "tapas",
            "tapas-nq-medium",
            "tapas-nq-large",
            "tapas-nq-medium-reversed",
            "tapas-nq-large-reversed",
        ],
        help="""
            Which model to train. For TAPAS, 'medium'/'large' is the size, 'nq' is the task 
            of open-domain retrieval, 'reversed' means we are reversing the question and 
            context encoders.
        """,
    )
    parser.add_argument(
        "--tokenization",
        type=str,
        default=None,
        help="Whether to use 'dpr' or 'tapas' style tokenization for context.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        dest="gradient_checkpointing",
        action="store_true",
        help="Use --gradient_checkpointing to save memory by using checkpoints.",
    )
    parser.add_argument(
        "--no_show_progress",
        dest="show_progress",
        action="store_false",
        help="Use --no_show_progress to disable progress bar.",
    )
    parser.add_argument(
        "--no_show_eval",
        dest="show_eval",
        action="store_false",
        help="Use --no_show_eval to disable progress bar for evaluation.",
    )
    parser.add_argument(
        "--lang",
        default="en",
        type=str,
        choices=["en", "fr"],
        help="Language of data. The file will be determined based on this.",
    )
    parser.set_defaults(
        gradient_checkpointing=False, show_progress=True, show_eval=True
    )

    args = parser.parse_args()

    # Update default args in some conditions
    if args.tokenization is None:
        args.tokenization = args.model

    if args.tokenization.startswith("tapas"):
        args.ctx_colname = "tapas_tokens"

    if args.tokenization not in ["dpr", "tapas", "tapas-nq"]:
        raise ValueError(
            f"Invalid tokenization option selected. Must be either 'dpr' or 'tapas'. Got {args.model}."
        )

    if args.model == "dpr" and args.tokenization == "tapas":
        raise ValueError(
            f"You cannot use a {args.model} model with {args.tokenization} tokenization scheme."
        )

    if args.lang == "fr" and args.model.startswith("tapas"):
        raise ValueError(
            f"{args.model} model is not currently supported with {args.lang} language."
        )

    # Warnings
    if args.lang == "fr" and not args.hn_colname.endswith("fr"):
        print(
            f"WARNING: You are using the French dataset with {args.model} model. "
            "You may want to use the French column name for hard negative PIDs."
        )

    main(args)
