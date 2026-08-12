from __future__ import annotations

from typing import Any

FORMAL_RUN_NOTE = (
    "Mechanics-mode pilot freeze on synthetic contract fixtures. The formal 100-target "
    "v0.1 run requires real ACL 2024 source Introductions, real NC_Physics test targets, "
    "and explicit model/credential authorization."
)


def build_gate5_report(
    review: dict[str, Any],
    config_versions: dict[str, str],
    prompt_versions: dict[str, str],
    expected_generations: int,
    desired_length_derivation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    generation_count = review["diagnostics"].get("generation_count", 0)
    checks = {
        "pilot_review_passed": {
            "passed": bool(review.get("passed")),
            "review_checks": {name: check["passed"] for name, check in review.get("checks", {}).items()},
        },
        "pilot_scale_complete": {
            "passed": generation_count == expected_generations,
            "generation_count": generation_count,
            "expected_generations": expected_generations,
        },
        "configs_frozen": {
            "passed": all(bool(value) for value in config_versions.values()),
            "config_versions": config_versions,
        },
        "prompt_versions_frozen": {
            "passed": all(bool(value) for value in prompt_versions.values()),
            "prompt_versions": prompt_versions,
        },
    }
    passed = all(check["passed"] for check in checks.values())
    return {
        "audit_version": "gate5-v1",
        "gate": "Gate 5",
        "passed": passed,
        "checks": checks,
        "freeze_status": "frozen" if passed else "needs_correction",
        "frozen_configs": config_versions,
        "frozen_prompt_versions": prompt_versions,
        "desired_introduction_length_derivation": desired_length_derivation,
        "note": FORMAL_RUN_NOTE,
    }
