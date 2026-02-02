from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Optional

from ..types import RAGChunk, RagConfig
from ..utils import jaccard_similarity, tokenize
from .embeddings import EmbeddingBackendError, SentenceTransformersBackend, TfidfBackend, cosine_similarity
from .index import RagIndex, build_chunks, load_index, save_index, RagIndexMeta


@dataclass
class BaseRetriever:
    def search(self, query: str, top_k: int = 5) -> List[RAGChunk]:
        raise NotImplementedError


class JaccardRetriever(BaseRetriever):
    def __init__(self, chunks: List[RAGChunk]):
        self.chunks = chunks

    def search(self, query: str, top_k: int = 5) -> List[RAGChunk]:
        query_tokens = tokenize(query)
        scored = []
        for chunk in self.chunks:
            score = jaccard_similarity(query_tokens, tokenize(chunk.text))
            scored.append((score, chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        results = []
        for score, chunk in scored[:top_k]:
            if score <= 0:
                continue
            results.append(RAGChunk(
                doc_id=chunk.doc_id,
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                score=score,
                meta=chunk.meta,
            ))
        return results


class EmbeddingRetriever(BaseRetriever):
    def __init__(self, index: RagIndex, backend_name: str = "tfidf"):
        self.index = index
        self.backend_name = backend_name
        if backend_name == "sentence_transformers":
            try:
                self.backend = SentenceTransformersBackend()
            except EmbeddingBackendError:
                self.backend = TfidfBackend()
        else:
            self.backend = TfidfBackend()
        if self.backend_name == "tfidf" and hasattr(self.backend, "fit"):
            self.backend.fit([chunk.text for chunk in self.index.chunks])

    def search(self, query: str, top_k: int = 5) -> List[RAGChunk]:
        query_vec = self.backend.embed([query])[0]
        scored = []
        for chunk, emb in zip(self.index.chunks, self.index.embeddings):
            score = cosine_similarity(query_vec, emb)
            scored.append((score, chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        results = []
        for score, chunk in scored[:top_k]:
            if score <= 0:
                continue
            results.append(RAGChunk(
                doc_id=chunk.doc_id,
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                score=score,
                meta=chunk.meta,
            ))
        return results


class RagStore:
    def __init__(self, config: RagConfig):
        self.config = config
        self.retriever: Optional[BaseRetriever] = None

    def load_documents(self, config: Optional[RagConfig] = None) -> None:
        cfg = config or self.config
        if cfg.mode == "embeddings":
            index = load_index(cfg.index_path)
            if index is None or cfg.rebuild_index:
                chunks = build_chunks(cfg.docs_paths)
                backend_name = "sentence_transformers" if cfg.mode == "embeddings" else "tfidf"
                backend = None
                try:
                    backend = SentenceTransformersBackend()
                except EmbeddingBackendError as exc:
                    if not cfg.allow_fallback:
                        raise
                    backend = TfidfBackend()
                    backend_name = "tfidf"
                embeddings = backend.embed([chunk.text for chunk in chunks])
                index = RagIndex(
                    chunks=chunks,
                    embeddings=embeddings,
                    meta=RagIndexMeta(
                        created_at=datetime.utcnow().isoformat() + "Z",
                        backend=backend_name,
                        docs=cfg.docs_paths,
                        chunk_count=len(chunks),
                        embedding_dim=len(embeddings[0]) if embeddings else 0,
                    ),
                )
                save_index(index, cfg.index_path)
            self.retriever = EmbeddingRetriever(index, backend_name=index.meta.backend if index.meta else "tfidf")
            return

        chunks = build_chunks(cfg.docs_paths)
        self.retriever = JaccardRetriever(chunks)

    def search(self, query: str, top_k: int = 5) -> List[RAGChunk]:
        if self.retriever is None:
            self.load_documents()
        if self.retriever is None:
            return []
        return self.retriever.search(query, top_k=top_k)

    def summarize(self, docs: Iterable[RAGChunk], max_chars: int = 800) -> str:
        chunks = []
        total = 0
        for doc in docs:
            snippet = doc.text.strip().replace("\n", " ")
            if len(snippet) > 200:
                snippet = snippet[:200] + "..."
            chunks.append(f"[{doc.doc_id}] {snippet}")
            total += len(chunks[-1])
            if total >= max_chars:
                break
        return "\n".join(chunks)
