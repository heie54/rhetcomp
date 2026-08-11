from __future__ import annotations

from typing import Any

from src.domain.models import TargetEvidence, TargetGold, TargetVisible
from src.ingest.text import normalize_plain_text


def normalize_target_record(
    record: dict[str, Any],
) -> tuple[TargetVisible, TargetEvidence, TargetGold]:
    required = (
        "target_id",
        "title",
        "abstract",
        "non_intro_sections",
        "reference_metadata",
        "gold_introduction",
    )
    missing = [field for field in required if field not in record]
    if missing:
        raise ValueError(f"Target record missing fields: {', '.join(missing)}")
    sections = record["non_intro_sections"]
    if not isinstance(sections, dict) or any(
        "intro" in str(name).lower() for name in sections
    ):
        raise ValueError("non_intro_sections must be a map without Introduction")
    visible = TargetVisible(
        target_id=record["target_id"],
        title=normalize_plain_text(record["title"]),
        abstract=normalize_plain_text(record["abstract"]),
    )
    evidence = TargetEvidence(
        target_id=record["target_id"],
        non_intro_sections={
            str(name): normalize_plain_text(text)
            for name, text in sorted(sections.items())
        },
        reference_metadata=tuple(record["reference_metadata"]),
    )
    gold = TargetGold(
        target_id=record["target_id"],
        introduction=normalize_plain_text(record["gold_introduction"]),
    )
    return visible, evidence, gold
