from __future__ import annotations

from typing import Any, Mapping

from src.domain.models import RepresentationArtifact

REPRESENTATION_TYPES = ("raw", "summary", "guideline", "experience")
COST_LOGGED_TYPES = ("summary", "guideline", "experience")


def build_cost_record(representation: RepresentationArtifact, mode: str) -> dict[str, Any]:
    return {
        "representation_id": representation.representation_id,
        "type": representation.type,
        "model": representation.compiler_model,
        "prompt_version": representation.compiler_prompt_version,
        "input_tokens": representation.compiler_input_tokens,
        "output_tokens": representation.compiler_output_tokens,
        "calls": representation.compiler_calls,
        "mode": mode,
        "logged": True,
    }


def build_gate3_report(
    representations: Mapping[str, RepresentationArtifact],
    costs: Mapping[str, dict[str, Any]],
    writing_condition_tokens: int,
) -> dict[str, Any]:
    missing = [name for name in REPRESENTATION_TYPES if name not in representations]
    corpus_hashes = {
        name: representations[name].source_corpus_hash
        for name in REPRESENTATION_TYPES
        if name in representations
    }
    same_corpus_hash = len(set(corpus_hashes.values())) == 1

    costs_logged = all(
        costs.get(name, {}).get("logged", False) for name in COST_LOGGED_TYPES
    )
    guideline = costs.get("guideline", {})
    experience = costs.get("experience", {})
    compute_comparable = all(
        guideline.get(key) is not None and experience.get(key) is not None
        for key in ("input_tokens", "output_tokens", "calls")
    )
    budget_respected = all(
        representations[name].content_tokens <= writing_condition_tokens
        for name in REPRESENTATION_TYPES
        if name in representations
    )

    checks = {
        "shared_source_corpus_hash": {
            "passed": same_corpus_hash and not missing,
            "missing_types": missing,
            "corpus_hashes": {name: corpus_hashes[name][:12] for name in corpus_hashes},
            "distinct_hash_count": len(set(corpus_hashes.values())),
        },
        "compiler_costs_logged": {
            "passed": costs_logged,
            "cost_logged_types": [
                name for name in COST_LOGGED_TYPES if costs.get(name, {}).get("logged")
            ],
        },
        "guideline_experience_compute_comparable": {
            "passed": compute_comparable,
            "guideline": {
                key: guideline.get(key) for key in ("input_tokens", "output_tokens", "calls")
            },
            "experience": {
                key: experience.get(key) for key in ("input_tokens", "output_tokens", "calls")
            },
        },
        "writing_budget_respected": {
            "passed": budget_respected,
            "budget_tokens": writing_condition_tokens,
            "content_tokens": {
                name: representations[name].content_tokens
                for name in REPRESENTATION_TYPES
                if name in representations
            },
        },
    }
    return {
        "audit_version": "gate3-v1",
        "gate": "Gate 3",
        "passed": all(check["passed"] for check in checks.values()),
        "checks": checks,
    }
