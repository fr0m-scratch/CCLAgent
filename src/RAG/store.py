from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from ..utils import jaccard_similarity, tokenize


@dataclass
class Document:
    doc_id: str
    text: str
    source: str


class RagStore:
    def __init__(self) -> None:
        self.documents: List[Document] = []

    def add_document(self, doc: Document) -> None:
        self.documents.append(doc)

    def add_documents_from_path(self, path: str) -> None:
        base = Path(path)
        if not base.exists():
            return
        for file_path in base.rglob("*"):
            if file_path.is_dir():
                continue
            if file_path.suffix.lower() not in {".md", ".txt", ".json"}:
                continue
            try:
                text = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            self.add_document(
                Document(
                    doc_id=str(file_path),
                    text=text,
                    source=str(file_path),
                )
            )

    def search(self, query: str, top_k: int = 5) -> List[Document]:
        query_tokens = tokenize(query)
        scored = []
        for doc in self.documents:
            score = jaccard_similarity(query_tokens, tokenize(doc.text))
            scored.append((score, doc))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [doc for score, doc in scored[:top_k] if score > 0]

    def summarize(self, docs: Iterable[Document], max_chars: int = 800) -> str:
        chunks = []
        total = 0
        for doc in docs:
            snippet = doc.text.strip().replace("\n", " ")
            if len(snippet) > 200:
                snippet = snippet[:200] + "..."
            chunks.append(f"[{doc.source}] {snippet}")
            total += len(chunks[-1])
            if total >= max_chars:
                break
        return "\n".join(chunks)
