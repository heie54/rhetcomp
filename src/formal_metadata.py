from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from src.common.jsonio import sha256_json


@dataclass(frozen=True, slots=True)
class FormalArtifactMetadata:
    run_id: str
    config_hash: str
    data_manifest_hash: str

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("Formal metadata requires run_id")
        for name in ("config_hash", "data_manifest_hash"):
            value = getattr(self, name)
            if len(value) != 64:
                raise ValueError(f"Formal metadata requires SHA-256 {name}")


def build_formal_metadata(
    run_id: str,
    *,
    configs: Mapping[str, Any],
    manifests: Mapping[str, Any],
) -> FormalArtifactMetadata:
    if not configs or not manifests:
        raise ValueError("Formal metadata requires configs and manifests")
    return FormalArtifactMetadata(
        run_id=run_id,
        config_hash=sha256_json(dict(configs)),
        data_manifest_hash=sha256_json(dict(manifests)),
    )
