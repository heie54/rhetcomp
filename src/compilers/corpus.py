from __future__ import annotations

from typing import Iterable

from src.common.jsonio import sha256_json
from src.domain.models import SourcePaper


def source_corpus_hash(sources: Iterable[SourcePaper]) -> str:
    """Deterministic hash of the normalized source corpus, shared by every compiler."""
    ordered = sorted((source.to_dict() for source in sources), key=lambda item: item["source_id"])
    return sha256_json(ordered)
