import argparse
from functools import partial
import os

import pandas as pd
import torch
import transformers as hft
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import wandb

from eacl_code.utils import encode_right_truncated_old, get_checkpoint_dir, get_data_dir

def main(args):
    # Set up wandb
    wandb.init(
        project=args.wandb_project,
        entity="statcan-models",
        name=args.wandb_name,
        reinit=True,
        dir=args.wandb_dir,
    )
    wandb_config = vars(args)
    wandb_config["name"] = wandb.run.name
    print("Training Configuration:")
    print(wandb_config)
    wandb.config.update(wandb_config)

    # Create a folder to store the model checkpoints
    ckpt_save_dir = os.path.join(args.model_dir, wandb.run.project, wandb.run.name)
    os.makedirs(ckpt_save_dir, exist_ok=True)

    # Load the models and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ClassForCondGeneration = hft.T5ForConditionalGeneration if args.lang == "en" else hft.MT5ForConditionalGeneration
    model = ClassForCondGeneration.from_pretrained(args.model).to(device)
    tokenizer = hft.AutoTokenizer.from_pretrained(args.model)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        use_cache = False
    else:
        use_cache = True

    tokenize_target = partial(
        tokenizer,
        max_length=args.max_target_length,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
    )

    optimizer: torch.optim.Optimizer = hft.Adafactor(
        [{"params": model.parameters()}], lr=args.lr, relative_step=args.lr is None
    )

    # Load dataframes
    train = pd.read_csv(os.path.join(args.data_dir, f"train_{args.lang}_augmented.csv.zip"))
    valid = pd.read_csv(os.path.join(args.data_dir, f"valid_{args.lang}_augmented.csv.zip"))

    # Preprocess source text
    train_source_ids, train_attention_masks = encode_right_truncated_old(
        tokenizer, train[args.source_colname].to_list(), args.max_source_length
    )
    valid_source_ids, valid_attention_masks = encode_right_truncated_old(
        tokenizer, valid[args.source_colname].to_list(), args.max_source_length
    )

    # Preprocess target text
    train_target_ids = tokenize_target(train[args.target_colname].to_list())[
        "input_ids"
    ]
    valid_target_ids = tokenize_target(valid[args.target_colname].to_list())[
        "input_ids"
    ]

    train_target_ids[train_target_ids == tokenizer.pad_token_id] = -100
    valid_target_ids[valid_target_ids == tokenizer.pad_token_id] = -100

    # Create DataLoaders
    train_loader = DataLoader(
        list(zip(train_source_ids, train_target_ids, train_attention_masks)),
        batch_size=args.batch_size,
        shuffle=True,
    )

    valid_loader = DataLoader(
        list(zip(valid_source_ids, valid_target_ids, valid_attention_masks)),
        batch_size=args.batch_size,
        shuffle=False,
    )

    for epoch in range(args.n_epochs):
        total_loss = 0.0
        step = 0

        model.train()
        optimizer.zero_grad()

        pbar = tqdm(train_loader, disable=not args.show_progress)
        for source_ids, target_ids, attention_mask in pbar:
            loss = model(
                input_ids=source_ids.to(model.device),
                attention_mask=attention_mask.to(model.device),
                labels=target_ids.to(model.device),
                use_cache=use_cache,
            ).loss
            loss.backward()
            total_loss += loss.item()
            step += 1
            running_loss = total_loss / step

            pbar.set_postfix({"running_loss": running_loss})

            # Gradient accumulation:
            if (step + 1) % args.accumulate_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            if step % 20 == 1:
                wandb.log({"running_loss": running_loss, "epoch": epoch, "step": step})

        # Evaluation
        model.eval()
        valid_loss = 0.0

        for source_ids, target_ids, attention_mask in tqdm(
            valid_loader, disable=not args.show_progress
        ):
            with torch.no_grad():
                loss = model(
                    input_ids=source_ids.to(model.device),
                    attention_mask=attention_mask.to(model.device),
                    labels=target_ids.to(model.device),
                ).loss
                valid_loss += loss.item()

        wandb.log({"valid_loss": valid_loss / len(valid_loader), "epoch": epoch})

        model.save_pretrained(os.path.join(ckpt_save_dir, f"epoch-{epoch}", "model"))

    # terminate the job
    wandb.finish()


if __name__ == "__main__":
    data_dir = get_data_dir()
    checkpoint_dir = get_checkpoint_dir()

    parser = argparse.ArgumentParser(
        description="Trains a generation model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=data_dir / "generation",
        help="Directory containing the dataset.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate for the Adafactor optimizer.",
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
        help="Show progress bar during training.",
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
        "--model",
        type=str,
        default="t5-small",
        help="Which variant of T5 to use (must be input input to `model.from_pretrained`).",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for training."
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
        "--wandb_name", type=str, help="Name of the experiment in wandb."
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        help="Name of the project in wandb.",
    )
    parser.add_argument(
        "--wandb_dir",
        type=str,
        help="Directory to store the wandb logs.",
        default=os.getenv("WANDB_DIR", "./wandb_logs"),
    )

    parser.add_argument(
        "--n_epochs", type=int, default=5, help="Number of epochs to train the model."
    )
    parser.add_argument(
        "--gradient_checkpointing",
        dest="gradient_checkpointing",
        action="store_true",
        help="Use --gradient_checkpointing to save memory by using checkpoints.",
    )
    parser.add_argument(
        "--accumulate_steps",
        type=int,
        default=1,
        help="Number of steps to accumulate gradient before backprop. If set to 1, it means gradient is backproped at every step.",
    )
    parser.add_argument(
        "--lang",
        default="en",
        type=str,
        choices=["en", "fr"],
        help="Language of data. The file will be determined based on this.",
    )

    parser.set_defaults(gradient_checkpointing=False)

    args = parser.parse_args()

    main(args)
