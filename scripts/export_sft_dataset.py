#!/usr/bin/env python3
from __future__ import annotations

import argparse

from src.data.export import export_sft_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Export SFT dataset from artifacts")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out", default="memory/datasets/sft.jsonl")
    args = parser.parse_args()
    export_sft_dataset(args.run_dir, args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
