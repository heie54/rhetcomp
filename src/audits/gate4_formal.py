from __future__ import annotations

from typing import Any, Sequence

from src.adapters.records import ProviderCallArtifact
from src.audits.call_traceability import check_generation_call_traceability
from src.audits.gate4 import build_gate4_report
from src.writer.config import WRITER_CONDITIONS
from src.writer.writer import GenerationArtifact


def build_gate4_formal_report(
    generations: Sequence[GenerationArtifact],
    costs: dict[str, Any],
    call_artifacts: Sequence[ProviderCallArtifact],
    *,
    gate3_report: dict[str, Any],
    order_manifest: dict[str, Any],
    expected_profile_hash: str,
    batch_run_id: str,
) -> dict[str, Any]:
    shared = build_gate4_report(generations, costs)
    all_calls = list(call_artifacts)
    successful = [call for call in all_calls if call.status == "success"]
    writer_calls = [call for call in successful if call.role.startswith("writer:")]
    fingerprints = {
        call.system_fingerprint
        for call in writer_calls
        if call.system_fingerprint
    }
    profile_hashes = {call.provider_profile_hash for call in writer_calls}
    requested_models = {call.requested_model for call in writer_calls}
    returned_models = {call.returned_model for call in writer_calls}
    expected_roles = {f"writer:{condition}" for condition in WRITER_CONDITIONS}
    actual_roles = {call.role for call in writer_calls}
    ordered = tuple(order_manifest.get("condition_order", ()))
    generation_config_hashes = {item.config_hash for item in generations}
    generation_data_hashes = {item.data_manifest_hash for item in generations}
    generation_profile_hashes = {item.provider_profile_hash for item in generations}
    generation_call_traceability = check_generation_call_traceability(
        generations, writer_calls
    )
    checks = {
        "gate3r_precondition": {
            "passed": gate3_report.get("status") == "PASS"
            and len(generation_data_hashes) == 1
            and gate3_report.get("data_manifest_hash") in generation_data_hashes,
            "status": gate3_report.get("status"),
            "upstream_data_manifest_hash": gate3_report.get("data_manifest_hash"),
        },
        "shared_writer_invariants": {"passed": shared["passed"], "report": shared},
        "real_writer_no_fallback": {
            "passed": len(generations) == 5
            and all(
                generation.run_mode == "formal"
                and generation.writer_model == "deepseek-v4-flash"
                and generation.provider_metadata
                for generation in generations
            ),
            "generation_count": len(generations),
        },
        "frozen_writer_profile": {
            "passed": len(all_calls) == 5
            and len(writer_calls) == 5
            and actual_roles == expected_roles
            and requested_models == {"deepseek-v4-flash"}
            and returned_models == {"deepseek-v4-flash"}
            and profile_hashes == {expected_profile_hash}
            and all(
                call.run_id == batch_run_id
                and call.gateway == "opencode_go"
                and call.thinking_mode == "disabled"
                and call.reasoning_effort is None
                and call.provider_request_id
                for call in writer_calls
            ),
            "requested_models": sorted(requested_models),
            "returned_models": sorted(returned_models),
            "profile_hashes": sorted(profile_hashes),
            "roles": sorted(actual_roles),
        },
        "provider_fingerprint_consistent": {
            "passed": len(fingerprints) <= 1,
            "required": False,
            "fingerprints": sorted(fingerprints),
        },
        "citations_target_evidence_only": {
            "passed": all(generation.citation_valid for generation in generations),
            "citation_indices": {
                generation.condition: list(generation.citation_indices)
                for generation in generations
            },
        },
        "condition_order_manifest": {
            "passed": len(ordered) == 5
            and set(ordered) == set(WRITER_CONDITIONS)
            and order_manifest.get("ordering_kind") == "local_seeded_shuffle"
            and order_manifest.get("model_generation_seed") is None,
            "condition_order": list(ordered),
        },
        "complete_call_and_cost_artifacts": {
            "passed": len(all_calls) == 5
            and len(writer_calls) == 5
            and all(costs.get(condition, {}).get("logged") for condition in WRITER_CONDITIONS),
            "writer_call_count": len(writer_calls),
            "cost_conditions": sorted(costs),
        },
        "generation_call_traceability": generation_call_traceability,
        "formal_metadata_chain": {
            "passed": len(generations) == 5
            and all(
                item.run_mode == "formal" and item.run_id == batch_run_id
                for item in generations
            )
            and generation_config_hashes == {order_manifest.get("config_hash")}
            and generation_data_hashes == {order_manifest.get("data_manifest_hash")}
            and generation_profile_hashes == {expected_profile_hash}
            and all(
                call.run_mode == "formal"
                and call.config_hash in generation_config_hashes
                and call.data_manifest_hash in generation_data_hashes
                for call in writer_calls
            ),
            "config_hashes": sorted(
                value for value in generation_config_hashes if value
            ),
            "data_manifest_hashes": sorted(
                value for value in generation_data_hashes if value
            ),
        },
    }
    passed = all(check["passed"] for check in checks.values())
    target_ids = {generation.target_id for generation in generations}
    return {
        "audit_version": "gate4r-v1",
        "gate": "4R",
        "run_mode": "formal",
        "run_id": batch_run_id,
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
        "provider_profile_hash": expected_profile_hash,
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
        "target_id": next(iter(target_ids)) if len(target_ids) == 1 else None,
        "checks": checks,
    }


def blocked_gate4_formal_report(reason: str, detail: Any) -> dict[str, Any]:
    return {
        "audit_version": "gate4r-v1",
        "gate": "4R",
        "run_mode": "formal",
        "status": "BLOCKED",
        "passed": False,
        "blockers": [{"reason": reason, "detail": detail}],
    }
