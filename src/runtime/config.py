from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from src.config import load_config


@dataclass(frozen=True, slots=True)
class CompilerRuntimeConfig:
    source_normalized_root: Path
    target_visible_root: Path
    target_evidence_root: Path
    evidence_packs_root: Path


@dataclass(frozen=True, slots=True)
class WriterRuntimeConfig:
    target_visible_root: Path
    evidence_packs_root: Path


@dataclass(frozen=True, slots=True)
class EvaluatorRuntimeConfig:
    target_gold_root: Path


def _resolve(root: Path, configured_path: str) -> Path:
    path = Path(configured_path)
    return (path if path.is_absolute() else root / path).resolve()


def _role_paths(
    dataset_config_path: str | Path,
    project_root: str | Path | None,
    allowed_names: Iterable[str],
) -> dict[str, Path]:
    config_path = Path(dataset_config_path).resolve()
    root = Path(project_root).resolve() if project_root else config_path.parent.parent
    dataset = load_config(config_path)
    paths = dataset.get("paths")
    if not isinstance(paths, dict):
        raise ValueError("Dataset config paths must be an object")

    selected: dict[str, Path] = {}
    for name in allowed_names:
        configured_path = paths.get(name)
        if not isinstance(configured_path, str) or not configured_path.strip():
            raise ValueError(f"Dataset config is missing paths.{name}")
        selected[name] = _resolve(root, configured_path)
    return selected


def load_compiler_runtime_config(
    dataset_config_path: str | Path,
    project_root: str | Path | None = None,
) -> CompilerRuntimeConfig:
    paths = _role_paths(
        dataset_config_path,
        project_root,
        ("source_normalized", "target_visible", "target_evidence", "evidence_packs"),
    )
    return CompilerRuntimeConfig(
        source_normalized_root=paths["source_normalized"],
        target_visible_root=paths["target_visible"],
        target_evidence_root=paths["target_evidence"],
        evidence_packs_root=paths["evidence_packs"],
    )


def load_writer_runtime_config(
    dataset_config_path: str | Path,
    project_root: str | Path | None = None,
) -> WriterRuntimeConfig:
    paths = _role_paths(
        dataset_config_path,
        project_root,
        ("target_visible", "evidence_packs"),
    )
    return WriterRuntimeConfig(
        target_visible_root=paths["target_visible"],
        evidence_packs_root=paths["evidence_packs"],
    )


def load_evaluator_runtime_config(
    dataset_config_path: str | Path,
    project_root: str | Path | None = None,
) -> EvaluatorRuntimeConfig:
    paths = _role_paths(dataset_config_path, project_root, ("target_gold",))
    return EvaluatorRuntimeConfig(target_gold_root=paths["target_gold"])
