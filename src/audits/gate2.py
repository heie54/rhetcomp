from __future__ import annotations

from typing import Any

from src.compilers.config import CompilerSettings
from src.compilers.experience.pipeline import ExperienceLibraryResult

VERIFIER_KEYS = {"observation_support", "strategy_generalization", "notes"}
OBSERVATION_VALUES = {"supported", "partial", "unsupported"}
GENERALIZATION_VALUES = {"reasonable", "overgeneralized", "unsupported"}


def build_gate2_report(
    result: ExperienceLibraryResult,
    settings: CompilerSettings,
    writing_condition_tokens: int,
) -> dict[str, Any]:
    span_checked = result.span_verified_count + result.span_rejected_count
    span_exactness = (
        result.span_verified_count / span_checked if span_checked else 0.0
    )

    span_reject_trace = [
        item for item in result.trace
        if item.get("stage") == "span_validate" and item.get("level") == "error"
    ]
    verifier_reject_trace = [
        item for item in result.trace
        if item.get("stage") == "verify"
        and item.get("grounding_status") == "rejected"
    ]
    verifier_entries = [
        item for item in result.trace if item.get("stage") == "verify"
    ]
    verifier_structured = all(
        isinstance(item.get("verifier_result"), dict)
        and VERIFIER_KEYS.issubset(item["verifier_result"])
        and item["verifier_result"]["observation_support"] in OBSERVATION_VALUES
        and item["verifier_result"]["strategy_generalization"] in GENERALIZATION_VALUES
        for item in verifier_entries
    )

    checks = {
        "atomic_candidates": {
            "passed": result.candidate_count > 0,
            "candidate_count": result.candidate_count,
            "adapter_mode": result.adapter_mode,
        },
        "exact_span_check": {
            "passed": span_exactness >= 0.95,
            "span_verified": result.span_verified_count,
            "span_rejected": result.span_rejected_count,
            "span_exactness_ratio": round(span_exactness, 6),
        },
        "rejected_spans_traceable": {
            "passed": len(span_reject_trace) == result.span_rejected_count
            and len(verifier_reject_trace) == result.verifier_rejected_count,
            "span_rejections_traced": len(span_reject_trace),
            "verifier_rejections_traced": len(verifier_reject_trace),
            "trace_span_rejections": [item["reason"] for item in span_reject_trace],
        },
        "structured_verifier_output": {
            "passed": verifier_structured,
            "verifier_entries": len(verifier_entries),
        },
        "merge_only_via_adjudication": {
            "passed": not result.merged_without_adjudication,
            "merged_without_adjudication": result.merged_without_adjudication,
            "adjudicated_pair_count": result.adjudicated_pair_count,
            "merge_edge_count": result.merge_edge_count,
        },
        "library_within_budget": {
            "passed": result.library.content_tokens <= writing_condition_tokens,
            "content_tokens": result.library.content_tokens,
            "budget_tokens": writing_condition_tokens,
            "included_experiences": len(result.library.included_experience_ids),
            "excluded_experiences": list(result.library.excluded_experience_ids),
            "stable_core": result.stable_core_count,
            "supported_rare": result.supported_rare_count,
        },
    }
    return {
        "audit_version": "gate2-v1",
        "gate": "Gate 2",
        "passed": all(check["passed"] for check in checks.values()),
        "checks": checks,
        "summary": {
            "adapter_mode": result.adapter_mode,
            "candidates": result.candidate_count,
            "span_verified": result.span_verified_count,
            "span_rejected": result.span_rejected_count,
            "support_verified": result.support_verified_count,
            "verifier_rejected": result.verifier_rejected_count,
            "canonical_experiences": result.canonical_count,
            "stable_core": result.stable_core_count,
            "supported_rare": result.supported_rare_count,
            "library_tokens": result.library.content_tokens,
            "writing_budget_tokens": writing_condition_tokens,
            "source_corpus_hash": result.source_corpus_hash,
        },
    }
