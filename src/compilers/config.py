from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.config import load_config


@dataclass(frozen=True, slots=True)
class CompilerSettings:
    config_version: str
    model: str | None
    experience_compiler_enabled: bool
    extraction_prompt_version: str
    max_candidates_per_source: int
    verifier_prompt_version: str
    admission_observation_support: tuple[str, ...]
    admission_strategy_generalization: tuple[str, ...]
    retrieval_backend: str
    retrieval_dimensions: int
    retrieval_top_k: int
    retrieval_min_cosine: float
    adjudication_prompt_version: str
    compatible_relations: tuple[str, ...]
    require_nonconflicting_applicable_when: bool
    summary_prompt_version: str
    guideline_prompt_version: str
    library_tier_order: tuple[str, ...]


def _require_mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Compiler config {path} must be an object")
    return value


def load_compiler_settings(compiler_config_path: str) -> CompilerSettings:
    config = load_config(compiler_config_path)
    extraction = _require_mapping(config.get("extraction"), "extraction")
    verifier = _require_mapping(config.get("verifier"), "verifier")
    retrieval = _require_mapping(config.get("retrieval"), "retrieval")
    adjudication = _require_mapping(config.get("adjudication"), "adjudication")
    summary = _require_mapping(config.get("summary"), "summary")
    guideline = _require_mapping(config.get("guideline"), "guideline")
    library = _require_mapping(config.get("library"), "library")

    def admission(values: Any, name: str) -> tuple[str, ...]:
        if not isinstance(values, list) or not values or any(
            not isinstance(item, str) or not item for item in values
        ):
            raise ValueError(f"Compiler config {name} must be a non-empty string list")
        return tuple(values)

    return CompilerSettings(
        config_version=config["config_version"],
        model=config.get("model"),
        experience_compiler_enabled=bool(config.get("experience_compiler_enabled", False)),
        extraction_prompt_version=extraction["prompt_version"],
        max_candidates_per_source=int(extraction.get("max_candidates_per_source", 1)),
        verifier_prompt_version=verifier["prompt_version"],
        admission_observation_support=admission(
            verifier.get("admission_observation_support"), "verifier.admission_observation_support"
        ),
        admission_strategy_generalization=admission(
            verifier.get("admission_strategy_generalization"),
            "verifier.admission_strategy_generalization",
        ),
        retrieval_backend=str(retrieval.get("pair_retrieval", "deterministic_feature_hash")),
        retrieval_dimensions=int(
            retrieval.get("dimensions", retrieval.get("feature_hash_dimensions", 128))
        ),
        retrieval_top_k=int(retrieval.get("top_k", 10)),
        retrieval_min_cosine=float(retrieval.get("min_cosine", 0.0)),
        adjudication_prompt_version=adjudication["prompt_version"],
        compatible_relations=admission(
            adjudication.get("compatible_relations"), "adjudication.compatible_relations"
        ),
        require_nonconflicting_applicable_when=bool(
            adjudication.get("require_nonconflicting_applicable_when", True)
        ),
        summary_prompt_version=summary["prompt_version"],
        guideline_prompt_version=guideline["prompt_version"],
        library_tier_order=admission(library.get("tier_order"), "library.tier_order"),
    )
