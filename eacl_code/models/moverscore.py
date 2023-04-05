"""
Custom implementation of the MoverScore metric, modified from:
https://github.com/AIPHES/emnlp19-moverscore/blob/master/moverscore_v2.py
"""
from __future__ import absolute_import, division, print_function
from typing import List, Union, Iterable
from itertools import zip_longest
import numpy as np
import torch
import string
from collections import defaultdict, Counter
from multiprocessing import Pool
from functools import partial
from math import log
from itertools import chain


from pyemd import emd_with_flow


def get_idf_dict(tokenizer, arr):
    idf_count = Counter()
    num_docs = len(arr)

    input_ids = tokenizer(arr, truncation=True).input_ids
    for sent_ids in input_ids:
        idf_count.update(set(sent_ids))

    idf_dict = defaultdict(lambda: log((num_docs + 1) / (1)))
    idf_dict.update(
        {idx: log((num_docs + 1) / (c + 1)) for (idx, c) in idf_count.items()}
    )
    return idf_dict


def load_metric(model_name=None, model=None, tokenizer=None, device="auto"):
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model is not None and model_name is not None:
        raise ValueError("Only provide `model` or `model_name`, not both.")

    if model is not None and tokenizer is None:
        raise ValueError(
            "If model is provided, tokenizer must be provided as well and vice versa"
        )

    if model is None and tokenizer is None:
        from transformers import AutoTokenizer, AutoModel

        if model_name is None:
            model_name = "distilbert-base-uncased"

        model = AutoModel.from_pretrained(
            model_name, output_hidden_states=True, output_attentions=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True)

    model.eval()
    model.to(device)

    def padding(arr, pad_token, dtype=torch.long):
        lens = torch.LongTensor([len(a) for a in arr])
        max_len = lens.max().item()
        padded = torch.ones(len(arr), max_len, dtype=dtype) * pad_token
        mask = torch.zeros(len(arr), max_len, dtype=torch.long)
        for i, a in enumerate(arr):
            padded[i, : lens[i]] = torch.tensor(a, dtype=dtype)
            mask[i, : lens[i]] = 1
        return padded, lens, mask

    def bert_encode(model, x, attention_mask):
        x = x.to(device)
        attention_mask = attention_mask.to(device)
        with torch.no_grad():
            result = model(x, attention_mask=attention_mask)
        if model_name == "distilbert-base-uncased":
            return result[1]
        else:
            return result[2]

    def truncate(tokens):
        if len(tokens) > tokenizer.model_max_length - 2:
            tokens = tokens[0:(tokenizer.model_max_length - 2)]
        return tokens

    def collate_idf(arr, tokenize, numericalize, idf_dict):
        pad = tokenizer.pad_token
        
        tokens = [[tokenizer.cls_token] + truncate(tokenize(a, truncation=True)) + [tokenizer.sep_token] for a in arr]
        arr = [numericalize(a) for a in tokens]

        idf_weights = [[idf_dict[i] for i in a] for a in arr]

        pad_token = numericalize([pad])[0]

        padded, lens, mask = padding(arr, pad_token, dtype=torch.long)
        padded_idf, _, _ = padding(idf_weights, pad_token, dtype=torch.float)

        return padded, padded_idf, lens, mask, tokens
    
    @torch.no_grad()
    def get_bert_embedding(all_sens, model, tokenizer, idf_dict, batch_size=-1):

        padded_sens, padded_idf, lens, mask, tokens = collate_idf(
            all_sens, tokenizer.tokenize, tokenizer.convert_tokens_to_ids, idf_dict
        )

        if batch_size == -1:
            batch_size = len(all_sens)

        embeddings = []
        model.eval()
        for i in range(0, len(all_sens), batch_size):
            batch_embedding = bert_encode(
                model,
                padded_sens[i : i + batch_size],
                attention_mask=mask[i : i + batch_size],
            )
            batch_embedding = torch.stack(batch_embedding)
            embeddings.append(batch_embedding)

        total_embedding = torch.cat(embeddings, dim=-3)
        return total_embedding, lens, mask, padded_idf, tokens

    def _safe_divide(numerator, denominator):
        return numerator / (denominator + 1e-30)

    def batched_cdist_l2(x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = (
            torch.baddbmm(x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2)
            .add_(x1_norm)
            .clamp_min_(1e-30)
            .sqrt_()
        )
        return res

    def word_mover_score(
        refs,
        hyps,
        idf_dict_ref,
        idf_dict_hyp,
        stop_words=[],
        n_gram=1,
        remove_subwords=True,
        batch_size=32,
    ):
        preds = []
        for batch_start in range(0, len(refs), batch_size):
            batch_refs = refs[batch_start : batch_start + batch_size]
            batch_hyps = hyps[batch_start : batch_start + batch_size]

            (
                ref_embedding,
                ref_lens,
                ref_masks,
                ref_idf,
                ref_tokens,
            ) = get_bert_embedding(batch_refs, model, tokenizer, idf_dict_ref)
            (
                hyp_embedding,
                hyp_lens,
                hyp_masks,
                hyp_idf,
                hyp_tokens,
            ) = get_bert_embedding(batch_hyps, model, tokenizer, idf_dict_hyp)

            ref_embedding = ref_embedding[-1]
            hyp_embedding = hyp_embedding[-1]

            batch_size = len(ref_tokens)
            for i in range(batch_size):
                ref_ids = [
                    k
                    for k, w in enumerate(ref_tokens[i])
                    if w in stop_words or "##" in w or w in set(string.punctuation)
                ]
                hyp_ids = [
                    k
                    for k, w in enumerate(hyp_tokens[i])
                    if w in stop_words or "##" in w or w in set(string.punctuation)
                ]

                ref_embedding[i, ref_ids, :] = 0
                hyp_embedding[i, hyp_ids, :] = 0

                ref_idf[i, ref_ids] = 0
                hyp_idf[i, hyp_ids] = 0

            raw = torch.cat([ref_embedding, hyp_embedding], 1).to(device)

            raw.div_(torch.norm(raw, dim=-1).unsqueeze(-1) + 1e-30)

            distance_matrix = batched_cdist_l2(raw, raw).double().cpu().numpy()

            for i in range(batch_size):
                c1 = np.zeros(raw.shape[1], dtype=np.float)
                c2 = np.zeros(raw.shape[1], dtype=np.float)
                c1[: len(ref_idf[i])] = ref_idf[i]
                c2[len(ref_idf[i]) :] = hyp_idf[i]

                c1 = _safe_divide(c1, np.sum(c1))
                c2 = _safe_divide(c2, np.sum(c2))

                dst = distance_matrix[i]
                _, flow = emd_with_flow(c1, c2, dst)
                flow = np.array(flow, dtype=np.float32)
                score = 1.0 / (1.0 + np.sum(flow * dst))  # 1 - np.sum(flow * dst)
                preds.append(score)

        return preds

    def word_score(
        references,
        predictions,
        stop_words=None,
        n_gram=1,
        remove_subwords=True,
        batch_size=32,
    ):
        if stop_words is None:
            stop_words = []

        idf_dict_hyp = get_idf_dict(
            tokenizer, predictions
        )  
        idf_dict_ref = get_idf_dict(
            tokenizer, references
        )

        return word_mover_score(
            references,
            predictions,
            idf_dict_ref,
            idf_dict_hyp,
            stop_words,
            n_gram,
            remove_subwords,
            batch_size,
        )

    def sentence_score(hypothesis: str, references: List[str], trace=0):    
        idf_dict_hyp = defaultdict(lambda: 1.)
        idf_dict_ref = defaultdict(lambda: 1.)
        
        hypothesis = [hypothesis] * len(references)
        
        sentence_score = 0 

        scores = word_mover_score(references, hypothesis, idf_dict_ref, idf_dict_hyp, stop_words=[], n_gram=1, remove_subwords=False)
        
        sentence_score = np.mean(scores)
        
        if trace > 0:
            print(hypothesis, references, sentence_score)
                
        return sentence_score

    def corpus_score(sys_stream: List[str],
                     ref_streams:Union[str, List[Iterable[str]]], trace=0):

        if isinstance(sys_stream, str):
            sys_stream = [sys_stream]

        if isinstance(ref_streams, str):
            ref_streams = [[ref_streams]]

        fhs = [sys_stream] + ref_streams

        corpus_score = 0
        for lines in zip_longest(*fhs):
            if None in lines:
                raise EOFError("Source and reference streams have different lengths!")
                
            hypo, *refs = lines
            corpus_score += sentence_score(hypo, refs, trace=0)
            
        corpus_score /= len(sys_stream)

        return corpus_score
    
    def metric(references, predictions):
        return corpus_score(sys_stream=predictions, ref_streams=references)

    return metric