from __future__ import annotations

from typing import Any, Callable

from src.domain.models import RepresentationArtifact, SourcePaper, TargetEvidencePack


SCHEMA_LOADERS: dict[str, Callable[[dict[str, Any]], Any]] = {
    "SourcePaper": SourcePaper.from_dict,
    "TargetEvidencePack": TargetEvidencePack.from_dict,
    "RepresentationArtifact": RepresentationArtifact.from_dict,
}


def validate_schema(schema_name: str, payload: dict[str, Any]) -> None:
    try:
        loader = SCHEMA_LOADERS[schema_name]
    except KeyError as exc:
        raise ValueError(f"Unknown schema: {schema_name}") from exc
    loader(payload)
