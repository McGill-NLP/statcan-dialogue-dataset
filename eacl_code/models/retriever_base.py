class RetrieverBase:
    def build_vocab(self, text: "list[str]"):
        """
        If necessary, creates an internal representation of the vocabulary

        Parameters
        ----------

        text: list of string
            Corpus used to build the vocabulary.
        """
        pass

    def encode_queries(self, text: "list[str]") -> list:
        """
        Parameters
        ----------

        text: list of string
            The queries to be encoded.

        Returns
        ------
        list or array_like
            The encoded representation of the queries.
        """
        pass

    def encode_passages(self, text: "list[str]") -> list:
        """
        Parameters
        ----------

        text: list of string
            The passages you want to encode.

        Returns
        ------
        list or array_like
            The encoded representation of the passages.
        """
        pass

    def build_model(self):
        pass

    def retrieve(
        self, queries_encoded: list, passages_encoded: list = None, k: int = None
    ):
        pass
