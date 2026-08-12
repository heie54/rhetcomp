from __future__ import annotations

from typing import Any, Sequence

from src.writer.config import WRITER_CONDITIONS
from src.writer.writer import GenerationArtifact


def build_gate4_report(
    generations: Sequence[GenerationArtifact],
    costs: dict[str, Any],
) -> dict[str, Any]:
    """Gate 4: for one target, all 5 conditions share prompts/evidence; only representation differs."""
    by_condition = {generation.condition: generation for generation in generations}
    evidence_hash_values = {generation.target_evidence_hash for generation in generations}
    template_hash_values = {generation.prompt_template_hash for generation in generations}
    base_hash_values = {generation.base_prompt_hash for generation in generations}
    representation_hashes = {
        generation.representation_hash for generation in generations
    }
    prompt_hash_values = {generation.writer_prompt_hash for generation in generations}

    all_conditions = set(by_condition) == set(WRITER_CONDITIONS)
    evidence_only = by_condition.get("evidence_only")
    representation_differs = (
        evidence_only is not None
        and evidence_only.representation_hash is None
        and len(representation_hashes) == len(WRITER_CONDITIONS)
    )
    output_complete = all(
        bool(generation.text.strip())
        and generation.output_tokens > 0
        and bool(generation.target_id)
        and bool(generation.writer_model)
        and bool(generation.generation_id)
        and bool(generation.writer_prompt_hash)
        for generation in generations
    )
    cost_complete = all(costs.get(condition, {}).get("logged") for condition in WRITER_CONDITIONS)

    checks = {
        "all_five_conditions_ran": {
            "passed": all_conditions,
            "conditions": sorted(by_condition),
        },
        "target_evidence_hash_identical": {
            "passed": len(evidence_hash_values) == 1,
            "distinct_hashes": len(evidence_hash_values),
        },
        "prompt_template_hash_identical": {
            "passed": len(template_hash_values) == 1,
            "distinct_hashes": len(template_hash_values),
        },
        "base_prompt_hash_identical": {
            "passed": len(base_hash_values) == 1,
            "distinct_hashes": len(base_hash_values),
        },
        "representation_hash_differs": {
            "passed": representation_differs,
            "evidence_only_representation_hash": (
                evidence_only.representation_hash if evidence_only else None
            ),
            "representation_hash_count": len(representation_hashes),
        },
        "full_prompt_hash_differs": {
            "passed": len(prompt_hash_values) == len(WRITER_CONDITIONS),
            "distinct_hashes": len(prompt_hash_values),
        },
        "output_artifacts_complete": {"passed": output_complete, "generation_count": len(generations)},
        "cost_artifacts_complete": {"passed": cost_complete, "conditions_logged": sorted(costs)},
    }
    return {
        "audit_version": "gate4-v1",
        "gate": "Gate 4",
        "target_id": by_condition["evidence_only"].target_id if evidence_only else None,
        "passed": all(check["passed"] for check in checks.values()),
        "checks": checks,
    }
