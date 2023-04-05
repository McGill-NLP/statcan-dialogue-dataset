import os

import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor as T


class TapasRetriever(nn.Module):
    def __init__(self, tapas: nn.Module, gradient_checkpointing: bool = False):
        super().__init__()
        self.tapas = tapas
        if gradient_checkpointing:
            self.tapas.gradient_checkpointing_enable()

        self.projection = nn.Linear(tapas.config.hidden_size, 256, bias=False).to(
            tapas.device
        )

    @property
    def device(self):
        return self.tapas.device

    def forward(self, input_ids, attention_mask):
        x = self.tapas(input_ids).pooler_output
        x = self.projection(x)
        return x

    def save_pretrained(self, path):
        self.tapas.save_pretrained(path)
        torch.save(self.projection.state_dict(), os.path.join(path, "projection.pt"))


def process_ctx(
    df: pd.DataFrame,
    pid_dict: dict,
    neg_col: str,
    pos_col: str = "target_pid",
):
    pos = df[pos_col].apply(lambda pid: pid_dict[pid]).to_list()
    pos_ids = torch.tensor(pos)

    hard_neg = df[neg_col].apply(lambda pid: pid_dict[pid]).to_list()
    neg_ids = torch.tensor(hard_neg)

    return pos_ids, neg_ids, torch.ones_like(pos_ids), torch.ones_like(neg_ids)
