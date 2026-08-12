from __future__ import annotations

from typing import Sequence

from src.budget.tokenizer import BudgetResult, Tokenizer


def atomic_budget(blocks: Sequence[str], tokenizer: Tokenizer, limit: int) -> BudgetResult:
    """Deterministically include whole content blocks until the token budget is reached.

    Emits valid, human-readable text and reports pre/post truncation token counts
    (spec §10). Used for Raw exemplar, Summary, and Guideline writing-condition content.
    """
    if limit < 0:
        raise ValueError("Token limit must be non-negative")
    pre_tokens = sum(len(tokenizer.encode(block)) for block in blocks)
    included: list[str] = []
    tokens = 0
    for block in blocks:
        block_tokens = len(tokenizer.encode(block))
        if tokens > 0 and tokens + block_tokens > limit:
            break
        included.append(block)
        tokens += block_tokens
    content = "\n\n".join(included)
    post_tokens = len(tokenizer.encode(content))
    return BudgetResult(
        content=content,
        pre_truncation_tokens=pre_tokens,
        post_truncation_tokens=post_tokens,
        truncated=len(included) < len(blocks),
        tokenizer_version=tokenizer.version,
    )
