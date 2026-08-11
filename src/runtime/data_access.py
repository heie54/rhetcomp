from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from src.runtime.config import (
    CompilerRuntimeConfig,
    EvaluatorRuntimeConfig,
    WriterRuntimeConfig,
)


def _record_path(root: Path, record_id: str) -> Path:
    if (
        not isinstance(record_id, str)
        or not record_id
        or record_id in {".", ".."}
        or Path(record_id).name != record_id
    ):
        raise ValueError("record_id must be a single non-empty path component")
    return root / f"{record_id}.json"


@dataclass(frozen=True, slots=True)
class CompilerDataAccess:
    _config: CompilerRuntimeConfig

    def __post_init__(self) -> None:
        if type(self._config) is not CompilerRuntimeConfig:
            raise TypeError("CompilerDataAccess requires CompilerRuntimeConfig")

    def configured_roots(self) -> Mapping[str, Path]:
        return {
            "source_normalized": self._config.source_normalized_root,
            "target_visible": self._config.target_visible_root,
            "target_evidence": self._config.target_evidence_root,
            "evidence_packs": self._config.evidence_packs_root,
        }

    def source_paper_path(self, source_id: str) -> Path:
        return _record_path(self._config.source_normalized_root, source_id)

    def target_visible_path(self, target_id: str) -> Path:
        return _record_path(self._config.target_visible_root, target_id)

    def target_evidence_path(self, target_id: str) -> Path:
        return _record_path(self._config.target_evidence_root, target_id)

    def evidence_pack_path(self, target_id: str) -> Path:
        return _record_path(self._config.evidence_packs_root, target_id)


@dataclass(frozen=True, slots=True)
class WriterDataAccess:
    _config: WriterRuntimeConfig

    def __post_init__(self) -> None:
        if type(self._config) is not WriterRuntimeConfig:
            raise TypeError("WriterDataAccess requires WriterRuntimeConfig")

    def configured_roots(self) -> Mapping[str, Path]:
        return {
            "target_visible": self._config.target_visible_root,
            "evidence_packs": self._config.evidence_packs_root,
        }

    def target_visible_path(self, target_id: str) -> Path:
        return _record_path(self._config.target_visible_root, target_id)

    def evidence_pack_path(self, target_id: str) -> Path:
        return _record_path(self._config.evidence_packs_root, target_id)


@dataclass(frozen=True, slots=True)
class EvaluatorDataAccess:
    _config: EvaluatorRuntimeConfig

    def __post_init__(self) -> None:
        if type(self._config) is not EvaluatorRuntimeConfig:
            raise TypeError("EvaluatorDataAccess requires EvaluatorRuntimeConfig")

    def configured_roots(self) -> Mapping[str, Path]:
        return {"target_gold": self._config.target_gold_root}

    def target_gold_path(self, target_id: str) -> Path:
        return _record_path(self._config.target_gold_root, target_id)
