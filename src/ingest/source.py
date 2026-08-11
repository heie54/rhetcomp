from __future__ import annotations

from typing import Any

from src.common.jsonio import sha256_json
from src.domain.models import SourcePaper
from src.ingest.text import normalize_introduction


def normalize_source_record(record: dict[str, Any]) -> SourcePaper:
    required = ("source_id", "title", "authors", "introduction")
    missing = [field for field in required if field not in record]
    if missing:
        raise ValueError(f"Source record missing fields: {', '.join(missing)}")
    introduction = normalize_introduction(record["introduction"])
    document_identity = {
        "source_id": record["source_id"],
        "title": record["title"],
        "authors": record["authors"],
        "venue": record.get("venue", "ACL 2024"),
        "track": record.get("track", "main"),
        "introduction": introduction.normalized_text,
    }
    return SourcePaper(
        source_id=record["source_id"],
        title=record["title"],
        authors=tuple(record["authors"]),
        venue=record.get("venue", "ACL 2024"),
        track=record.get("track", "main"),
        introduction=introduction,
        document_hash=sha256_json(document_identity),
    )
