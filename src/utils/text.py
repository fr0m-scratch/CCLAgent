import re
from typing import Iterable


_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


def normalize(text: str) -> str:
    return " ".join(_TOKEN_RE.findall(text.lower()))


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def jaccard_similarity(tokens_a: Iterable[str], tokens_b: Iterable[str]) -> float:
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    if not set_a and not set_b:
        return 0.0
    return len(set_a & set_b) / max(1, len(set_a | set_b))
