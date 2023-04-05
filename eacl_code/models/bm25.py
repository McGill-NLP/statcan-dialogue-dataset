import numpy as np
from gensim import corpora
from gensim.summarization import bm25
from tqdm.auto import tqdm

from .retriever_base import RetrieverBase


class BM25Retriever(RetrieverBase):
    def __init__(self, tokenizer=None):
        if tokenizer is None:
            tokenizer = lambda x: x.split()

        self.tokenizer = tokenizer

    def build_vocab(self, text):
        tokenized = [self.tokenizer(doc) for doc in text]
        self.dictionary = corpora.Dictionary(tokenized)

    def encode_queries(self, text):
        return [self.dictionary.doc2bow(self.tokenizer(t)) for t in text]

    def encode_passages(self, text):
        return self.encode_queries(text)

    def build_model(self, passages_encoded: list):
        """
        Parameters
        ----------

        passages_encoded: list or array_like
            The encoded representation of the passages.

        Returns
        -------

        model
            The BM25 model that can be used to compute the score from a query.
        """
        self.model = bm25.BM25(passages_encoded)
        return self.model

    def retrieve(
        self,
        queries_encoded: list,
        passages_encoded: list = None,
        k: int = None,
        verbose: bool = False,
    ):
        """
        Parameters
        ----------
        queries_encoded: list or array_like
            The encoded representation of the queries.
        passages_encoded: list or array_like, optional
            The encoded representation of the passages. If provided, then
            `self.build_model(passage_encoded)` will be run. If not provided, it will fall back
            to the model that has been previously built with `self.build_model`. If no model
            was previously created, this will throw an error.
        k: int, optional
            The number of passages we want to retrieve for each query. If not provided, it will
            return all the passages.
        verbose: boolean, default True
            Whether to print the progress of the retrieval.

        Returns
        -------
        indices: ndarray
            An array of dimension (N, k) containing the indices of the top k retrieved
            passages for all N queries.
        scores: ndarray
            An array dimension (N, k) of relevance scores corresponding to each index retrieved.
        """

        if passages_encoded is not None:
            self.build_model(passages_encoded)

        all_indices = []
        all_scores = []

        for query in tqdm(queries_encoded, disable=not verbose):
            scores = np.array(self.model.get_scores(query))

            topk_ix = np.flip(scores.argsort())
            if k is not None:
                topk_ix = topk_ix[:k]

            topk_scores = scores[topk_ix]

            all_indices.append(topk_ix)
            all_scores.append(topk_scores)

        all_indices = np.stack(all_indices)
        all_scores = np.stack(all_scores)

        return all_indices, all_scores
