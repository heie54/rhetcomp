from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from src.budget.tokenizer import Tokenizer
from src.common.jsonio import canonical_json, sha256_text
from src.compilers.experience.canonicalize import CanonicalExperience


@dataclass(frozen=True)
class LibraryBudgetResult:
    content: str
    content_tokens: int
    content_hash: str
    included_experience_ids: tuple[str, ...]
    excluded_experience_ids: tuple[str, ...]


def _library_entry(canonical: CanonicalExperience) -> dict[str, Any]:
    evidence = [
        {
            "source_id": item.source_id,
            "paragraph": item.location.paragraph,
            "sentence_start": item.location.sentence_start,
            "sentence_end": item.location.sentence_end,
        }
        for item in canonical.experience.evidence
    ]
    return {
        "experience_id": canonical.experience.experience_id,
        "tier": canonical.meta.tier,
        "distinct_source_count": canonical.meta.distinct_source_count,
        "observed_pattern": canonical.experience.observed_pattern,
        "strategy": canonical.experience.strategy,
        "applicable_when": canonical.experience.applicable_when,
        "evidence": evidence,
    }


def build_library(
    canonical: Sequence[CanonicalExperience],
    tokenizer: Tokenizer,
    limit: int,
    tier_order: Sequence[str],
) -> LibraryBudgetResult:
    """Serialize the fixed-budget Experience Library (spec §8.7 → Writer representation).

    Entries are atomic; deterministic selection keeps as many as fit within the writing
    condition token budget, always emitting valid canonical JSON.
    """
    tier_rank = {tier: index for index, tier in enumerate(tier_order)}
    ordered = sorted(
        canonical,
        key=lambda item: (
            tier_rank.get(item.meta.tier, len(tier_order)),
            item.experience.experience_id,
        ),
    )

    included: list[dict[str, Any]] = []
    excluded: list[str] = []
    tokens = 0
    for item in ordered:
        entry = _library_entry(item)
        entry_text = canonical_json(entry)
        entry_tokens = len(tokenizer.encode(entry_text))
        if tokens > 0 and tokens + entry_tokens > limit:
            excluded.append(item.experience.experience_id)
            continue
        included.append(entry)
        tokens += entry_tokens

    content = canonical_json(included)
    content_tokens = len(tokenizer.encode(content))
    return LibraryBudgetResult(
        content=content,
        content_tokens=content_tokens,
        content_hash=sha256_text(content),
        included_experience_ids=tuple(item["experience_id"] for item in included),
        excluded_experience_ids=tuple(excluded),
    )
