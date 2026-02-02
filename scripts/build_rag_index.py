#!/usr/bin/env python3
from __future__ import annotations

import argparse

from src.RAG.store import RagStore
from src.types import RagConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RAG index")
    parser.add_argument("--mode", default="embeddings", choices=["embeddings", "jaccard"], help="RAG mode")
    parser.add_argument("--index-path", default="rag_index", help="Index output directory")
    parser.add_argument("--docs", nargs="*", default=["doc/Design", "doc/Knowledge", "README", "workload"], help="Docs paths")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild")
    args = parser.parse_args()

    config = RagConfig(
        mode=args.mode,
        index_path=args.index_path,
        docs_paths=args.docs,
        rebuild_index=args.rebuild,
    )
    store = RagStore(config)
    store.load_documents(config)
    print(f"RAG index ready at {args.index_path}")


if __name__ == "__main__":
    main()
