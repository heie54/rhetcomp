from __future__ import annotations

from pathlib import Path
from typing import Any

from src.common.jsonio import sha256_json, write_json


def artifact_hash(payload: Any) -> str:
    return sha256_json(payload)


def artifact_id(prefix: str, payload: Any) -> str:
    if not prefix or not prefix.replace("_", "").isalnum():
        raise ValueError("Artifact prefix must be alphanumeric with optional underscores")
    return f"{prefix}_{artifact_hash(payload)[:20]}"


class ArtifactStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def put(self, namespace: str, identifier: str, payload: Any) -> Path:
        if Path(namespace).is_absolute() or ".." in Path(namespace).parts:
            raise ValueError("Artifact namespace must stay under the store root")
        if Path(identifier).name != identifier:
            raise ValueError("Artifact identifier must be a filename-safe stem")
        destination = self.root / namespace / f"{identifier}.json"
        write_json(destination, payload)
        return destination
