from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from src.adapters.records import ProviderCallArtifact
from src.compilers.formal_representations import RepresentationBudgetMetrics
from src.domain.models import RepresentationArtifact


TYPES = ("raw", "summary", "guideline", "experience")


def build_gate3_formal_report(
    representations: Mapping[str, RepresentationArtifact],
    budget_metrics: Mapping[str, RepresentationBudgetMetrics],
    call_artifacts: Sequence[ProviderCallArtifact],
    *,
    gate2_report: dict[str, Any],
    writing_condition_tokens: int,
    run_id: str,
) -> dict[str, Any]:
    missing = [name for name in TYPES if name not in representations]
    hashes = {
        name: representations[name].source_corpus_hash
        for name in TYPES
        if name in representations
    }
    models = {
        name: representations[name].compiler_model
        for name in ("summary", "guideline", "experience")
        if name in representations
    }
    guideline = representations.get("guideline")
    experience = representations.get("experience")
    if guideline and experience and experience.compiler_calls > 0:
        call_diff = abs(guideline.compiler_calls - experience.compiler_calls)
        call_tolerance = max(1, math.ceil(experience.compiler_calls * 0.10))
        input_diff_ratio = abs(
            guideline.compiler_input_tokens - experience.compiler_input_tokens
        ) / experience.compiler_input_tokens if experience.compiler_input_tokens else math.inf
    else:
        call_diff = math.inf
        call_tolerance = 0
        input_diff_ratio = math.inf
    successful = [call for call in call_artifacts if call.status == "success"]
    deepseek = [call for call in successful if call.provider == "deepseek"]
    qwen = [call for call in successful if call.provider == "qwen"]
    required_roles = {
        "experience_extractor",
        "experience_verifier",
        "summary_compiler",
        "guideline_compiler",
    }
    if experience and any(call.role == "experience_adjudicator" for call in deepseek):
        required_roles.add("experience_adjudicator")
    actual_roles = {call.role for call in deepseek}
    representation_config_hashes = {
        item.config_hash for item in representations.values()
    }
    representation_data_hashes = {
        item.data_manifest_hash for item in representations.values()
    }
    representation_profile_hashes = {
        item.provider_profile_hash for item in representations.values()
    }
    metrics_complete = not missing and all(
        name in budget_metrics
        and budget_metrics[name].pre_budget_tokens >= budget_metrics[name].post_budget_tokens
        and budget_metrics[name].post_budget_tokens
        == representations[name].content_tokens
        and budget_metrics[name].tokenizer_version.startswith("deepseek_formal:")
        for name in TYPES
    )
    checks = {
        "gate2r_precondition": {
            "passed": gate2_report.get("status") == "PASS"
            and len(representation_config_hashes) == 1
            and gate2_report.get("config_hash") in representation_config_hashes
            and len(representation_data_hashes) == 1
            and gate2_report.get("data_manifest_hash") in representation_data_hashes,
            "status": gate2_report.get("status"),
            "upstream_config_hash": gate2_report.get("config_hash"),
            "upstream_data_manifest_hash": gate2_report.get("data_manifest_hash"),
        },
        "shared_real_source_corpus": {
            "passed": not missing and len(set(hashes.values())) == 1,
            "missing": missing,
            "source_corpus_hashes": hashes,
        },
        "formal_compiler_models": {
            "passed": len(models) == 3
            and all(model == "deepseek-v4-flash" for model in models.values()),
            "models": models,
        },
        "real_provider_calls_nonzero": {
            "passed": bool(guideline)
            and bool(experience)
            and representations["summary"].compiler_calls > 0
            and guideline.compiler_calls > 0
            and experience.compiler_calls > 0,
            "calls": {
                name: representations[name].compiler_calls
                for name in ("summary", "guideline", "experience")
                if name in representations
            },
        },
        "guideline_experience_compute_envelope": {
            "passed": call_diff <= call_tolerance and input_diff_ratio <= 0.15,
            "call_difference": call_diff,
            "call_tolerance": call_tolerance,
            "input_token_difference_ratio": input_diff_ratio,
            "input_token_tolerance": 0.15,
        },
        "formal_writing_budgets": {
            "passed": writing_condition_tokens == 4000
            and not missing
            and all(
                representations[name].content_tokens <= writing_condition_tokens
                for name in TYPES
            )
            and metrics_complete,
            "budget_tokens": writing_condition_tokens,
            "metrics": {
                name: budget_metrics[name].to_dict()
                for name in TYPES
                if name in budget_metrics
            },
        },
        "guideline_not_trivially_experience": {
            "passed": bool(guideline)
            and bool(experience)
            and guideline.content_hash != experience.content_hash,
            "guideline_hash": guideline.content_hash if guideline else None,
            "experience_hash": experience.content_hash if experience else None,
        },
        "provider_call_artifacts_complete": {
            "passed": required_roles.issubset(actual_roles)
            and bool(qwen)
            and all(
                call.run_id == run_id
                and call.gateway == "opencode_go"
                and call.requested_model == "deepseek-v4-flash"
                and call.returned_model == "deepseek-v4-flash"
                and call.provider_request_id
                and call.thinking_mode == "enabled"
                and call.reasoning_effort == "high"
                for call in deepseek
            )
            and all(
                call.gateway == "alibaba_model_studio"
                and call.requested_model == "qwen3.7-text-embedding"
                and call.returned_model == "qwen3.7-text-embedding"
                and call.role == "experience_candidate_retrieval"
                and (
                    (call.execution_kind == "network" and bool(call.provider_request_id))
                    or (
                        call.execution_kind == "cache"
                        and bool(call.cache_origin_call_ids)
                        and bool(call.cache_origin_provider_request_ids)
                    )
                )
                for call in qwen
            ),
            "required_roles": sorted(required_roles),
            "actual_roles": sorted(actual_roles),
            "deepseek_call_count": len(deepseek),
            "qwen_call_count": len(qwen),
        },
        "formal_metadata_chain": {
            "passed": not missing
            and all(
                item.run_mode == "formal" and item.run_id == run_id
                for item in representations.values()
            )
            and len(representation_config_hashes) == 1
            and None not in representation_config_hashes
            and len(representation_data_hashes) == 1
            and None not in representation_data_hashes
            and len(representation_profile_hashes) == 1
            and None not in representation_profile_hashes
            and bool(successful)
            and all(
                call.run_mode == "formal"
                and call.run_id == run_id
                and call.config_hash in representation_config_hashes
                and call.data_manifest_hash in representation_data_hashes
                for call in successful
            ),
            "config_hashes": sorted(
                value for value in representation_config_hashes if value
            ),
            "data_manifest_hashes": sorted(
                value for value in representation_data_hashes if value
            ),
            "representation_profile_hashes": sorted(
                value for value in representation_profile_hashes if value
            ),
        },
    }
    passed = all(check["passed"] for check in checks.values())
    return {
        "audit_version": "gate3r-v1",
        "gate": "3R",
        "run_mode": "formal",
        "run_id": run_id,
        "config_hash": (
            next(iter(representation_config_hashes))
            if len(representation_config_hashes) == 1
            else None
        ),
        "data_manifest_hash": (
            next(iter(representation_data_hashes))
            if len(representation_data_hashes) == 1
            else None
        ),
        "provider_profile_hashes": sorted(
            {call.provider_profile_hash for call in successful}
        ),
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
        "checks": checks,
    }


def blocked_gate3_formal_report(reason: str, detail: Any) -> dict[str, Any]:
    return {
        "audit_version": "gate3r-v1",
        "gate": "3R",
        "run_mode": "formal",
        "status": "BLOCKED",
        "passed": False,
        "blockers": [{"reason": reason, "detail": detail}],
    }
