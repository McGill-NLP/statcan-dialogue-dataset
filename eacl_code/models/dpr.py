import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import Tensor as T
from transformers import PreTrainedTokenizerFast
import transformers as hft

from ..utils import encode_right_truncated_old

# ############################ MODELING ############################
def enable_gradient_checkpointing(model):
    if model.config.model_type == "dpr":
        encoder = list(model.children())[0]
        encoder.bert_model.gradient_checkpointing_enable()
    else:
        model.gradient_checkpointing_enable()



def criterion(S, target=None):
    softS = F.log_softmax(S, dim=1)
    if target is None:
        target = torch.arange(0, S.shape[0])

    target = target.to(softS.device)
    loss = F.nll_loss(softS, target, reduction="mean")

    return loss


def get_sorted_indices(S):
    S_indices_sorted = S.T.argsort(0).flip(0).cpu()
    return S_indices_sorted


def compute_accuracy(all_pids, target, topk_ix):
    topk_pids = all_pids[topk_ix]
    matches = np.any(topk_pids == target, axis=0)
    return matches.sum() / matches.shape[0]


def batched_encode(net: hft.PreTrainedModel, loader: DataLoader, verbose: bool = False):
    return torch.cat(
        [
            net(input_ids=ids.to(net.device), attention_mask=masks.to(net.device)).pooler_output
            for ids, masks in tqdm(loader, disable=not verbose, leave=False)
        ]
    )


def evaluate(
    net: hft.PreTrainedModel,
    loader: DataLoader,
    P: T,
    all_pids: list,
    target: list,
    top_k_list: list,
    prefix: str = "",
    show_eval: bool = False,
):
    net.eval()

    pid_to_ix = {pid: ix for ix, pid in enumerate(all_pids)}

    with torch.no_grad():
        target_ix = torch.tensor([pid_to_ix[pid] for pid in target])

        Q = batched_encode(net, loader, verbose=show_eval)

        P = P.to(Q.device)

        S = torch.mm(Q, P.T)
        loss = criterion(S, target_ix).item()
        sorted_indices = get_sorted_indices(S)

    accuracies = {
        k: compute_accuracy(
            all_pids=all_pids, target=target, topk_ix=sorted_indices[:k]
        )
        for k in range(min(top_k_list), max(top_k_list) + 1)
    }

    results = {f"{prefix}acc_at_{k}": accuracies[k] for k in top_k_list}

    results[f"{prefix}loss"] = loss
    results[f"{prefix}acc_table"] = pd.DataFrame(
        accuracies.items(), columns=["k", "acc"]
    )

    return results


def train_step(
    ctx_encoder: hft.PreTrainedModel,
    q_encoder: hft.PreTrainedModel,
    optimizer: torch.optim.Optimizer,
    query_ids: T,
    query_masks: T,
    pos_ids: T,
    pos_masks: T,
    neg_ids: T,
    neg_masks: T,
):
    optimizer.zero_grad()

    ids = torch.cat([pos_ids, neg_ids]).to(ctx_encoder.device)
    masks = torch.cat([pos_masks, neg_masks]).to(ctx_encoder.device)

    P = ctx_encoder(input_ids=ids, attention_mask=masks).pooler_output
    Q = q_encoder(
        input_ids=query_ids.to(q_encoder.device),
        attention_mask=query_masks.to(q_encoder.device),
    ).pooler_output
    P = P.to(Q.device)
    S = torch.mm(Q, P.T)

    loss = criterion(S)

    loss.backward()
    optimizer.step()

    return loss


# ############################ DATA PROCESSING ############################
def process_ctx(
    tokenizer: PreTrainedTokenizerFast,
    df: pd.DataFrame,
    pid_dict: dict,
    neg_col: str,
    pos_col: str = "target_pid",
):
    pos = df[pos_col].apply(lambda pid: pid_dict[pid]).to_list()
    pos_tokenized = tokenizer(pos, return_tensors="pt")
    pos_ids = pos_tokenized["input_ids"]
    pos_masks = pos_tokenized["attention_mask"]

    hard_neg = df[neg_col].apply(lambda pid: pid_dict[pid]).to_list()
    hn_tokenized = tokenizer(hard_neg, return_tensors="pt")
    neg_ids = hn_tokenized["input_ids"]
    neg_masks = hn_tokenized["attention_mask"]

    return pos_ids, pos_masks, neg_ids, neg_masks


def process_queries(
    tokenizer: PreTrainedTokenizerFast,
    df: pd.DataFrame,
    q_col: str,
    max_length: int = 512,
    padding: str = "max_length",
):
    queries = df[q_col].to_list()

    ids, masks = encode_right_truncated_old(
        tokenizer, queries, max_length=max_length, padding=padding
    )

    return ids, masks
