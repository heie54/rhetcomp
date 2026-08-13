from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from src.adapters.records import ProviderCallArtifact
from src.common.jsonio import read_json
from src.compilers.config import CompilerSettings
from src.compilers.experience.pipeline import ExperienceLibraryResult
from src.domain.models import SourcePaper


VERIFIER_KEYS = {"observation_support", "strategy_generalization", "notes"}
OBSERVATION_VALUES = {"supported", "partial", "unsupported"}
GENERALIZATION_VALUES = {"reasonable", "overgeneralized", "unsupported"}


def load_call_artifacts(root: Path) -> list[ProviderCallArtifact]:
    calls_root = root / "calls"
    if not calls_root.exists():
        return []
    return [
        ProviderCallArtifact.from_dict(read_json(path))
        for path in sorted(calls_root.glob("*.json"))
    ]


def _evidence_exact(result: ExperienceLibraryResult, sources: Sequence[SourcePaper]) -> bool:
    source_by_id = {source.source_id: source for source in sources}
    for experience in result.experiences:
        for evidence in experience.evidence:
            source = source_by_id.get(evidence.source_id)
            if source is None:
                return False
            location = evidence.location
            try:
                paragraph = source.introduction.paragraphs[location.paragraph - 1]
                first = paragraph.sentences[location.sentence_start - 1]
                last = paragraph.sentences[location.sentence_end - 1]
            except IndexError:
                return False
            window = source.introduction.normalized_text[first.char_start:last.char_end]
            if evidence.span not in window:
                return False
    return True


def build_gate2_formal_report(
    result: ExperienceLibraryResult,
    settings: CompilerSettings,
    writing_condition_tokens: int,
    *,
    sources: Sequence[SourcePaper],
    acl_manifest: dict[str, Any],
    gate1_report: dict[str, Any],
    call_artifacts: Sequence[ProviderCallArtifact],
) -> dict[str, Any]:
    trace = list(result.trace)
    verifier_entries = [item for item in trace if item.get("stage") == "verify"]
    span_rejects = [
        item
        for item in trace
        if item.get("stage") == "span_validate" and item.get("level") == "error"
    ]
    verifier_rejects = [
        item
        for item in verifier_entries
        if item.get("grounding_status") == "rejected"
    ]
    verifier_structured = bool(verifier_entries) and all(
        isinstance(item.get("verifier_result"), dict)
        and VERIFIER_KEYS.issubset(item["verifier_result"])
        and item["verifier_result"]["observation_support"] in OBSERVATION_VALUES
        and item["verifier_result"]["strategy_generalization"] in GENERALIZATION_VALUES
        for item in verifier_entries
    )
    calls = list(call_artifacts)
    call_by_id = {call.call_id: call for call in calls}
    successful = [call for call in calls if call.status == "success"]
    deepseek = [call for call in successful if call.provider == "deepseek"]
    qwen = [call for call in successful if call.provider == "qwen"]
    config_hashes = {call.config_hash for call in successful}
    data_manifest_hashes = {call.data_manifest_hash for call in successful}
    required_deepseek_roles = {"experience_extractor", "experience_verifier"}
    if result.adjudicated_pair_count:
        required_deepseek_roles.add("experience_adjudicator")
    actual_deepseek_roles = {call.role for call in deepseek}
    merge_entries = [
        item for item in trace if item.get("stage") == "adjudicate" and item.get("merges")
    ]
    merges_have_calls = all(
        isinstance(item.get("provider_call"), dict)
        and item["provider_call"].get("call_id") in call_by_id
        and call_by_id[item["provider_call"]["call_id"]].role == "experience_adjudicator"
        for item in merge_entries
    )
    tiers_reproducible = all(
        meta.tier == ("stable_core" if meta.distinct_source_count >= 2 else "supported_rare")
        for meta in result.derived_meta
    )
    evidence_count = sum(len(item.evidence) for item in result.experiences)
    checks = {
        "gate1r_precondition": {
            "passed": gate1_report.get("status") == "PASS"
            and len(data_manifest_hashes) == 1
            and gate1_report.get("data_manifest_hash") in data_manifest_hashes,
            "status": gate1_report.get("status"),
            "upstream_data_manifest_hash": gate1_report.get("data_manifest_hash"),
        },
        "formal_real_backends": {
            "passed": result.run_mode == "formal"
            and result.adapter_mode == "model:deepseek-v4-flash"
            and result.embedding_backend == "qwen3_7_text_embedding"
            and result.embedding_model == "qwen3.7-text-embedding"
            and result.embedding_dimensions == 1024
            and settings.retrieval_top_k == 20
            and result.deterministic_fallback_count == 0,
            "adapter_mode": result.adapter_mode,
            "embedding_backend": result.embedding_backend,
            "embedding_model": result.embedding_model,
            "embedding_dimensions": result.embedding_dimensions,
            "deterministic_fallback_count": result.deterministic_fallback_count,
        },
        "real_acl_corpus": {
            "passed": acl_manifest.get("provider") == "ACL Anthology"
            and acl_manifest.get("ready_count") == 20
            and acl_manifest.get("source_corpus_hash") == result.source_corpus_hash
            and len(sources) == 20,
            "source_count": len(sources),
            "source_corpus_hash": result.source_corpus_hash,
        },
        "retained_spans_exact": {
            "passed": evidence_count > 0 and _evidence_exact(result, sources),
            "retained_evidence_count": evidence_count,
            "retained_exactness_ratio": 1.0 if evidence_count and _evidence_exact(result, sources) else 0.0,
        },
        "structured_blind_verifier": {
            "passed": verifier_structured,
            "verifier_entries": len(verifier_entries),
        },
        "all_rejections_traceable": {
            "passed": len(span_rejects) == result.span_rejected_count
            and len(verifier_rejects) == result.verifier_rejected_count,
            "span_rejections": result.span_rejected_count,
            "span_rejections_traced": len(span_rejects),
            "verifier_rejections": result.verifier_rejected_count,
            "verifier_rejections_traced": len(verifier_rejects),
        },
        "embedding_call_recorded": {
            "passed": bool(qwen)
            and all(call.gateway == "alibaba_model_studio" for call in qwen)
            and all(call.requested_model == "qwen3.7-text-embedding" for call in qwen)
            and any(call.role == "experience_candidate_retrieval" for call in qwen)
            and all(
                call.execution_kind == "network"
                or (
                    call.cache_origin_call_ids
                    and call.cache_origin_provider_request_ids
                )
                for call in qwen
            ),
            "qwen_call_count": len(qwen),
            "embedding_call_id": result.embedding_call_id,
            "cache_hits": result.embedding_cache_hits,
            "cache_misses": result.embedding_cache_misses,
        },
        "merge_only_via_adjudication": {
            "passed": not result.merged_without_adjudication and merges_have_calls,
            "merge_count": len(merge_entries),
            "adjudicated_pair_count": result.adjudicated_pair_count,
        },
        "tier_reproducibility": {
            "passed": tiers_reproducible,
            "stable_core": result.stable_core_count,
            "supported_rare": result.supported_rare_count,
        },
        "library_within_formal_budget": {
            "passed": writing_condition_tokens == 4000
            and result.library.content_tokens <= writing_condition_tokens,
            "content_tokens": result.library.content_tokens,
            "budget_tokens": writing_condition_tokens,
        },
        "provider_metadata_complete": {
            "passed": required_deepseek_roles.issubset(actual_deepseek_roles)
            and all(
                call.requested_model == "deepseek-v4-flash"
                and call.gateway == "opencode_go"
                and call.returned_model == "deepseek-v4-flash"
                and call.provider_request_id
                and call.thinking_mode == "enabled"
                and call.reasoning_effort == "high"
                for call in deepseek
            ),
            "required_roles": sorted(required_deepseek_roles),
            "actual_roles": sorted(actual_deepseek_roles),
            "deepseek_call_count": len(deepseek),
        },
        "formal_metadata_chain": {
            "passed": bool(successful)
            and all(
                call.run_mode == "formal" and call.run_id == result.run_id
                for call in successful
            )
            and len(config_hashes) == 1
            and None not in config_hashes
            and len(data_manifest_hashes) == 1
            and None not in data_manifest_hashes,
            "config_hashes": sorted(value for value in config_hashes if value),
            "data_manifest_hashes": sorted(
                value for value in data_manifest_hashes if value
            ),
        },
        "three_run_consensus_not_implemented": {"passed": True, "implemented": False},
    }
    passed = all(check["passed"] for check in checks.values())
    return {
        "audit_version": "gate2r-v1",
        "gate": "2R",
        "run_mode": "formal",
        "run_id": result.run_id,
        "config_hash": next(iter(config_hashes)) if len(config_hashes) == 1 else None,
        "data_manifest_hash": (
            next(iter(data_manifest_hashes)) if len(data_manifest_hashes) == 1 else None
        ),
        "provider_profile_hashes": sorted(
            {call.provider_profile_hash for call in successful}
        ),
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
        "checks": checks,
        "summary": {
            "candidate_count": result.candidate_count,
            "support_verified_count": result.support_verified_count,
            "canonical_count": result.canonical_count,
            "retrieved_pair_count": result.retrieved_pair_count,
            "adjudicated_pair_count": result.adjudicated_pair_count,
            "format_repair_count": result.format_repair_count,
            "compiler_calls": result.compiler_calls,
            "compiler_input_tokens": result.compiler_input_tokens,
            "compiler_output_tokens": result.compiler_output_tokens,
        },
    }


def blocked_gate2_formal_report(reason: str, detail: Any) -> dict[str, Any]:
    return {
        "audit_version": "gate2r-v1",
        "gate": "2R",
        "run_mode": "formal",
        "status": "BLOCKED",
        "passed": False,
        "blockers": [{"reason": reason, "detail": detail}],
    }
