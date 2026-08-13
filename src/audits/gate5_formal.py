from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

from src.adapters.records import ProviderCallArtifact
from src.artifacts_freeze import validate_freeze_manifest
from src.audits.call_traceability import check_generation_call_traceability
from src.writer.config import WRITER_CONDITIONS
from src.writer.writer import GenerationArtifact


def build_gate5_formal_report(
    review: dict[str, Any],
    generations: Sequence[GenerationArtifact],
    calls_by_target: Mapping[str, Sequence[ProviderCallArtifact]],
    order_manifests: Mapping[str, dict[str, Any]],
    *,
    upstream_reports: Mapping[str, dict[str, Any]],
    target_manifest: dict[str, Any],
    freeze_manifest: dict[str, Any],
) -> dict[str, Any]:
    target_counts = Counter(generation.target_id for generation in generations)
    condition_counts = Counter(generation.condition for generation in generations)
    target_ids = set(target_counts)
    batch_details: dict[str, Any] = {}
    batches_valid = True
    traceability_valid = set(calls_by_target) == target_ids
    for target_id in sorted(target_ids):
        all_target_calls = list(calls_by_target.get(target_id, ()))
        calls = [
            call
            for call in all_target_calls
            if call.status == "success" and call.role.startswith("writer:")
        ]
        fingerprints = {
            call.system_fingerprint
            for call in calls
            if call.system_fingerprint
        }
        roles = {call.role for call in calls}
        order = tuple(order_manifests.get(target_id, {}).get("condition_order", ()))
        target_generations = [
            generation for generation in generations if generation.target_id == target_id
        ]
        traceability = check_generation_call_traceability(target_generations, calls)
        valid = (
            len(all_target_calls) == 5
            and len(calls) == 5
            and roles == {f"writer:{condition}" for condition in WRITER_CONDITIONS}
            and len(fingerprints) <= 1
            and len(order) == 5
            and set(order) == set(WRITER_CONDITIONS)
            and all(call.requested_model == "deepseek-v4-flash" for call in calls)
            and all(call.returned_model == "deepseek-v4-flash" for call in calls)
            and all(call.gateway == "opencode_go" for call in calls)
            and all(call.thinking_mode == "disabled" for call in calls)
        )
        batches_valid = batches_valid and valid
        traceability_valid = traceability_valid and traceability["passed"]
        batch_details[target_id] = {
            "valid": valid,
            "call_count": len(calls),
            "fingerprint_required": False,
            "fingerprints": sorted(fingerprints),
            "condition_order": list(order),
            "generation_call_traceability": traceability,
        }
    formal_generations = all(
        generation.run_mode == "formal"
        and generation.writer_model == "deepseek-v4-flash"
        and generation.provider_metadata
        and generation.citation_valid
        for generation in generations
    )
    all_writer_calls = [call for calls in calls_by_target.values() for call in calls]
    global_profile_hashes = {call.provider_profile_hash for call in all_writer_calls}
    generation_config_hashes = {item.config_hash for item in generations}
    generation_data_hashes = {item.data_manifest_hash for item in generations}
    generation_profile_hashes = {item.provider_profile_hash for item in generations}
    upstream_statuses = {
        name: report.get("status") for name, report in upstream_reports.items()
    }
    gate2 = upstream_reports.get("2R", {})
    gate3 = upstream_reports.get("3R", {})
    upstream_data_hashes = {
        report.get("data_manifest_hash") for report in upstream_reports.values()
    }
    output_lengths = [generation.output_tokens for generation in generations]
    checks = {
        "all_upstream_formal_gates_passed": {
            "passed": upstream_statuses == {
                "1R": "PASS",
                "2R": "PASS",
                "3R": "PASS",
                "4R": "PASS",
            }
            and len(generation_data_hashes) == 1
            and upstream_data_hashes == generation_data_hashes,
            "statuses": upstream_statuses,
            "upstream_data_manifest_hashes": sorted(
                value for value in upstream_data_hashes if value
            ),
        },
        "real_pilot_scale_complete": {
            "passed": len(generations) == 50
            and len(target_ids) == 10
            and all(count == 5 for count in target_counts.values())
            and all(condition_counts[condition] == 10 for condition in WRITER_CONDITIONS),
            "generation_count": len(generations),
            "target_count": len(target_ids),
            "target_generation_counts": dict(sorted(target_counts.items())),
            "condition_counts": dict(sorted(condition_counts.items())),
        },
        "real_writer_batches_valid": {
            "passed": formal_generations
            and batches_valid
            and len(global_profile_hashes) == 1
            and all(call.provider_request_id for call in all_writer_calls),
            "batches": batch_details,
            "provider_profile_hashes": sorted(global_profile_hashes),
        },
        "generation_call_traceability": {
            "passed": traceability_valid,
            "target_keys_match": set(calls_by_target) == target_ids,
            "batches": {
                target_id: details["generation_call_traceability"]
                for target_id, details in batch_details.items()
            },
        },
        "dev_test_isolation": {
            "passed": target_manifest.get("dataset") == "Xiao-Youth/NC_Physics"
            and target_manifest.get("source_split") == "train"
            and target_manifest.get("selected_count") == 10
            and target_manifest.get("official_test_accessed") is False,
            "source_split": target_manifest.get("source_split"),
            "official_test_accessed": target_manifest.get("official_test_accessed"),
        },
        "real_pilot_review": {
            "passed": review.get("passed") is True,
            "review_checks": {
                name: check.get("passed") for name, check in review.get("checks", {}).items()
            },
        },
        "compiler_stability_diagnostics": {
            "passed": gate2.get("status") == "PASS" and gate3.get("status") == "PASS",
            "format_repair_count": gate2.get("summary", {}).get("format_repair_count"),
            "candidate_count": gate2.get("summary", {}).get("candidate_count"),
            "support_verified_count": gate2.get("summary", {}).get("support_verified_count"),
            "retrieved_pair_count": gate2.get("summary", {}).get("retrieved_pair_count"),
            "adjudicated_pair_count": gate2.get("summary", {}).get("adjudicated_pair_count"),
            "compute_match": gate3.get("checks", {}).get(
                "guideline_experience_compute_envelope"
            ),
        },
        "output_lengths_and_citations": {
            "passed": bool(output_lengths)
            and min(output_lengths) > 0
            and max(output_lengths) <= 600
            and all(generation.citation_valid for generation in generations),
            "min_output_tokens": min(output_lengths) if output_lengths else None,
            "max_output_tokens": max(output_lengths) if output_lengths else None,
        },
        "formal_freeze_manifest_complete": {
            "passed": validate_freeze_manifest(freeze_manifest),
            "freeze_manifest_hash": freeze_manifest.get("freeze_manifest_hash"),
        },
        "formal_metadata_chain": {
            "passed": len(generation_config_hashes) == 1
            and None not in generation_config_hashes
            and len(generation_data_hashes) == 1
            and None not in generation_data_hashes
            and generation_profile_hashes == global_profile_hashes
            and all(call.run_mode == "formal" for call in all_writer_calls)
            and all(call.config_hash in generation_config_hashes for call in all_writer_calls)
            and all(
                call.data_manifest_hash in generation_data_hashes
                for call in all_writer_calls
            )
            and all(
                manifest.get("run_mode") == "formal"
                and manifest.get("config_hash") in generation_config_hashes
                and manifest.get("data_manifest_hash") in generation_data_hashes
                for manifest in order_manifests.values()
            ),
            "config_hashes": sorted(
                value for value in generation_config_hashes if value
            ),
            "data_manifest_hashes": sorted(
                value for value in generation_data_hashes if value
            ),
        },
        "no_condition_ranking_or_final_statistics": {
            "passed": True,
            "condition_ranking_computed": False,
            "final_statistics_computed": False,
        },
    }
    passed = all(check["passed"] for check in checks.values())
    return {
        "audit_version": "gate5r-v1",
        "gate": "5R",
        "run_mode": "formal",
        "run_id": freeze_manifest.get("run_id"),
        "config_hash": (
            next(iter(generation_config_hashes))
            if len(generation_config_hashes) == 1
            else None
        ),
        "data_manifest_hash": (
            next(iter(generation_data_hashes))
            if len(generation_data_hashes) == 1
            else None
        ),
        "provider_profile_hashes": sorted(global_profile_hashes),
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
        "freeze_status": "formal_pilot_frozen" if passed else "not_frozen",
        "checks": checks,
        "note": "Formal pilot validation only; not a paper result or condition comparison.",
    }


def blocked_gate5_formal_report(reason: str, detail: Any) -> dict[str, Any]:
    return {
        "audit_version": "gate5r-v1",
        "gate": "5R",
        "run_mode": "formal",
        "status": "BLOCKED",
        "passed": False,
        "freeze_status": "not_frozen",
        "blockers": [{"reason": reason, "detail": detail}],
    }
