from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

from ..types import RAGChunk
from ..utils import tokenize


ALLOWED_SUFFIXES = {".md", ".txt", ".json", ".pdf"}


@dataclass
class RagIndexMeta:
    created_at: str
    backend: str
    docs: List[str]
    chunk_count: int
    embedding_dim: int


@dataclass
class RagIndex:
    chunks: List[RAGChunk] = field(default_factory=list)
    embeddings: List[List[float]] = field(default_factory=list)
    meta: Optional[RagIndexMeta] = None


def gather_documents(paths: Iterable[str]) -> List[Path]:
    files: List[Path] = []
    for path_str in paths:
        path = Path(path_str)
        if not path.exists():
            continue
        if path.is_file():
            files.append(path)
            continue
        for file_path in path.rglob("*"):
            if file_path.is_dir():
                continue
            if file_path.suffix.lower() not in ALLOWED_SUFFIXES:
                continue
            files.append(file_path)
    return files


def chunk_text(text: str, max_tokens: int = 200, overlap: int = 20) -> List[str]:
    tokens = tokenize(text)
    if not tokens:
        return []
    chunks: List[str] = []
    start = 0
    while start < len(tokens):
        end = min(len(tokens), start + max_tokens)
        chunk_tokens = tokens[start:end]
        chunks.append(" ".join(chunk_tokens))
        if end >= len(tokens):
            break
        start = max(0, end - overlap)
    return chunks


def build_chunks(paths: Iterable[str]) -> List[RAGChunk]:
    chunks: List[RAGChunk] = []
    for file_path in gather_documents(paths):
        text = _read_document_text(file_path)
        if not text:
            continue
        for idx, chunk in enumerate(chunk_text(text)):
            chunks.append(
                RAGChunk(
                    doc_id=str(file_path),
                    chunk_id=f"{file_path.name}:{idx}",
                    text=chunk,
                    score=0.0,
                    meta={"source": str(file_path)},
                )
            )
    return chunks


def _read_document_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf_text(path)
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return ""
    except Exception:
        return ""


def _extract_pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except Exception:
        return ""
    try:
        reader = PdfReader(str(path))
    except Exception:
        return ""
    pages: List[str] = []
    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        if text.strip():
            pages.append(text)
    return "\n".join(pages)


def save_index(index: RagIndex, index_path: str) -> None:
    base = Path(index_path)
    base.mkdir(parents=True, exist_ok=True)
    chunks_path = base / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as handle:
        for chunk in index.chunks:
            handle.write(json.dumps({
                "doc_id": chunk.doc_id,
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "meta": chunk.meta,
            }) + "\n")

    embeddings_path = base / "embeddings.json"
    with embeddings_path.open("w", encoding="utf-8") as handle:
        json.dump(index.embeddings, handle)
    try:
        import numpy as np
    except Exception:
        np = None
    if np is not None:
        np.save(base / "embeddings.npy", np.array(index.embeddings, dtype=float))

    meta = index.meta
    if meta is None:
        meta = RagIndexMeta(
            created_at=datetime.utcnow().isoformat() + "Z",
            backend="unknown",
            docs=[],
            chunk_count=len(index.chunks),
            embedding_dim=len(index.embeddings[0]) if index.embeddings else 0,
        )
    meta_path = base / "index_meta.json"
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(meta.__dict__, handle, indent=2)


def load_index(index_path: str) -> Optional[RagIndex]:
    base = Path(index_path)
    chunks_path = base / "chunks.jsonl"
    embeddings_path = base / "embeddings.json"
    npy_path = base / "embeddings.npy"
    meta_path = base / "index_meta.json"
    if not chunks_path.exists():
        return None

    chunks: List[RAGChunk] = []
    with chunks_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            chunks.append(
                RAGChunk(
                    doc_id=payload.get("doc_id", ""),
                    chunk_id=payload.get("chunk_id", ""),
                    text=payload.get("text", ""),
                    score=0.0,
                    meta=payload.get("meta", {}),
                )
            )
    embeddings = []
    if npy_path.exists():
        try:
            import numpy as np
            embeddings = np.load(npy_path).tolist()
        except Exception:
            embeddings = []
    if not embeddings and embeddings_path.exists():
        with embeddings_path.open("r", encoding="utf-8") as handle:
            embeddings = json.load(handle)

    meta = None
    if meta_path.exists():
        meta_payload = json.loads(meta_path.read_text(encoding="utf-8"))
        meta = RagIndexMeta(**meta_payload)
    return RagIndex(chunks=chunks, embeddings=embeddings, meta=meta)
