from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    # rough heuristic: 4 chars per token
    return max(1, int(len(text) / 4))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    trimmed = text[:max_chars].rstrip()
    return trimmed + "\n...[truncated]"


@dataclass
class PromptSection:
    name: str
    content: str
    priority: int = 1  # lower = more important
    max_tokens: Optional[int] = None
    min_tokens: int = 0


class ContextWindowManager:
    def __init__(self, max_tokens: int, reserve_tokens: int = 0) -> None:
        self.max_tokens = max(0, int(max_tokens))
        self.reserve_tokens = max(0, int(reserve_tokens))

    def build(self, sections: List[PromptSection]) -> tuple[str, Dict[str, Any]]:
        budget = max(0, self.max_tokens - self.reserve_tokens)
        enriched = []
        for idx, section in enumerate(sections):
            original = section.content or ""
            original_tokens = estimate_tokens(original)
            content = original
            if section.max_tokens is not None and original_tokens > section.max_tokens:
                content = truncate_to_tokens(content, section.max_tokens)
            tokens = estimate_tokens(content)
            enriched.append(
                {
                    "index": idx,
                    "name": section.name,
                    "priority": section.priority,
                    "min_tokens": max(0, int(section.min_tokens)),
                    "max_tokens": section.max_tokens,
                    "tokens_before": original_tokens,
                    "tokens_after": tokens,
                    "content": content,
                    "truncated": content != original,
                }
            )

        total_tokens = sum(item["tokens_after"] for item in enriched)
        overflow = max(0, total_tokens - budget)
        if overflow > 0:
            # truncate lowest-priority sections first
            for item in sorted(enriched, key=lambda s: (s["priority"], s["tokens_after"]), reverse=True):
                if overflow <= 0:
                    break
                available = max(0, item["tokens_after"] - item["min_tokens"])
                if available <= 0:
                    continue
                reduce_by = min(available, overflow)
                target_tokens = max(item["min_tokens"], item["tokens_after"] - reduce_by)
                item["content"] = truncate_to_tokens(item["content"], target_tokens)
                item["tokens_after"] = estimate_tokens(item["content"])
                item["truncated"] = True
                overflow -= reduce_by

        # rebuild in original order
        ordered = sorted(enriched, key=lambda s: s["index"])
        parts = []
        for item in ordered:
            if not item["content"].strip():
                continue
            parts.append(f"[{item['name']}]\n{item['content']}")
        prompt_text = "\n\n".join(parts)
        meta = {
            "schema_version": "1.0",
            "max_tokens": self.max_tokens,
            "reserve_tokens": self.reserve_tokens,
            "budget_tokens": budget,
            "total_tokens": sum(item["tokens_after"] for item in ordered),
            "sections": [
                {
                    "name": item["name"],
                    "priority": item["priority"],
                    "min_tokens": item["min_tokens"],
                    "max_tokens": item["max_tokens"],
                    "tokens_before": item["tokens_before"],
                    "tokens_after": item["tokens_after"],
                    "truncated": item["truncated"],
                }
                for item in ordered
            ],
        }
        return prompt_text, meta
