from __future__ import annotations

import math
from typing import Iterable, List

from ..utils import tokenize


class EmbeddingBackendError(RuntimeError):
    pass


class EmbeddingBackend:
    name = "base"

    def embed(self, texts: Iterable[str]) -> List[List[float]]:
        raise NotImplementedError


class SentenceTransformersBackend(EmbeddingBackend):
    name = "sentence_transformers"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingBackendError("sentence_transformers not installed") from exc
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: Iterable[str]) -> List[List[float]]:
        embeddings = self._model.encode(list(texts))
        return [emb.tolist() for emb in embeddings]


class TfidfBackend(EmbeddingBackend):
    name = "tfidf"

    def __init__(self) -> None:
        self._vocab = {}
        self._idf = {}

    def fit(self, texts: List[str]) -> None:
        doc_count = len(texts)
        df = {}
        for text in texts:
            tokens = set(tokenize(text))
            for token in tokens:
                df[token] = df.get(token, 0) + 1
        self._vocab = {token: idx for idx, token in enumerate(sorted(df.keys()))}
        self._idf = {token: math.log((doc_count + 1) / (count + 1)) + 1.0 for token, count in df.items()}

    def embed(self, texts: Iterable[str]) -> List[List[float]]:
        texts = list(texts)
        if not self._vocab:
            self.fit(texts)
        embeddings: List[List[float]] = []
        vocab_size = len(self._vocab)
        for text in texts:
            vec = [0.0] * vocab_size
            tokens = tokenize(text)
            if not tokens:
                embeddings.append(vec)
                continue
            tf = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
            for token, count in tf.items():
                if token not in self._vocab:
                    continue
                idx = self._vocab[token]
                vec[idx] = (count / len(tokens)) * self._idf.get(token, 0.0)
            embeddings.append(vec)
        return embeddings


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for a, b in zip(vec_a, vec_b):
        dot += a * b
        norm_a += a * a
        norm_b += b * b
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))
