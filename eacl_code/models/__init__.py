import torch
import transformers as hft
import numpy as np
import fasttext

from ..utils import get_data_dir, prepare_for_training, encode_right_truncated


class CustomDPRModel:
    def __init__(
        self,
        q_enc_path,
        cache_path=get_data_dir() / "apps/cache/dpr_passages_cached.pt",
        q_dpr_name="facebook/dpr-question_encoder-single-nq-base",
        ctx_dpr_name="facebook/dpr-ctx_encoder-single-nq-base",
        token=None,
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        q_enc = hft.AutoModel.from_pretrained(
            q_enc_path, use_auth_token=token
        ).to(device)
        q_enc.eval()

        q_tokenizer = hft.AutoTokenizer.from_pretrained(q_dpr_name)
        ctx_tokenizer = hft.AutoTokenizer.from_pretrained(ctx_dpr_name)

        self.pid_token = ctx_tokenizer.convert_ids_to_tokens(1)

        self.q_enc = q_enc
        self.q_tokenizer = q_tokenizer
        self.ctx_tokenizer = ctx_tokenizer

        self.P = torch.load(cache_path)

    def retrieve_table_indices(self, conversation_data, k=20):
        """
        Retrieve table indices from the query.
        """
        conversation_processed = prepare_for_training(
            conversation_data, pid_token=self.pid_token
        )
        x = torch.tensor(
            [encode_right_truncated(self.q_tokenizer, conversation_processed)]
        )

        with torch.no_grad():
            Q = self.q_enc(x.to(self.q_enc.device)).pooler_output
            S = torch.mm(Q, self.P.T.to(Q.device))

        sorted_indices = self.get_sorted_indices(S)[:k].numpy().squeeze()

        return sorted_indices

    # copied from dpr.py
    @staticmethod
    def get_sorted_indices(S):
        S_indices_sorted = S.T.argsort(0).flip(0).cpu()
        return S_indices_sorted


class FTDetector:
    def __init__(self, path=get_data_dir() / "weights" / "lid.176.ftz"):
        self.path = str(path)
        self.model = fasttext.load_model(self.path)

    def detect(self, text: list, label_no_content=True):
        """
        text: A list of strings
        label_no_content: If True, return "no_content" if the text is empty
        """
        if type(text) not in [str, list]:
            text = list(text)

        if type(text) is list:
            text = [t.replace("\n", " ") for t in text]
        else:
            text = [text.replace("\n", " ")]

        labels, probs = self.model.predict(text)
        probs = np.concatenate(probs)

        new_labels = []
        new_probs = []

        for l, t, p in zip(labels, text, probs):
            if t == "":
                if label_no_content:
                    l = "no_content"
                    p = 0
            else:
                l = l[0].replace("__label__", "")

            new_labels.append(l)
            new_probs.append(p)

        return new_labels, new_probs
