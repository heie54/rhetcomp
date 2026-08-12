from __future__ import annotations

import statistics
from typing import Iterable

from src.budget.tokenizer import Tokenizer


def derive_desired_introduction_length(
    gold_introductions: Iterable[str],
    tokenizer: Tokenizer,
) -> dict:
    """Derive the frozen target Introduction length from the target dataset distribution.

    Spec §11: the desired length is derived once from the target dataset distribution and
    frozen before the full run. Uses the gold Introductions (evaluation-side input).
    """
    lengths = sorted(
        len(tokenizer.encode(text)) for text in gold_introductions if text and text.strip()
    )
    if not lengths:
        return {
            "derived": False,
            "count": 0,
            "mean_tokens": None,
            "median_tokens": None,
            "min_tokens": None,
            "max_tokens": None,
        }
    return {
        "derived": True,
        "count": len(lengths),
        "mean_tokens": round(statistics.fmean(lengths), 2),
        "median_tokens": float(statistics.median(lengths)),
        "min_tokens": lengths[0],
        "max_tokens": lengths[-1],
    }
