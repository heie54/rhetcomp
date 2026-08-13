from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

from src.adapters.records import ProviderCallArtifact
from src.common.jsonio import sha256_text
from src.writer.writer import GenerationArtifact


def check_generation_call_traceability(
    generations: Sequence[GenerationArtifact],
    calls: Sequence[ProviderCallArtifact],
) -> dict[str, Any]:
    """Verify a one-to-one, content-bound link between generations and calls."""

    call_id_counts = Counter(call.call_id for call in calls)
    call_by_id = {call.call_id: call for call in calls}
    generation_call_ids = [
        str((generation.provider_metadata or {}).get("call_id") or "")
        for generation in generations
    ]
    generation_call_id_counts = Counter(generation_call_ids)
    links: list[dict[str, Any]] = []

    for generation, call_id in zip(generations, generation_call_ids, strict=True):
        metadata = generation.provider_metadata or {}
        call = call_by_id.get(call_id)
        link_checks = {
            "call_exists": call is not None,
            "call_id_unique": bool(call_id)
            and call_id_counts.get(call_id) == 1
            and generation_call_id_counts.get(call_id) == 1,
            "role_matches_condition": call is not None
            and call.role == f"writer:{generation.condition}",
            "response_hash_matches_text": call is not None
            and call.response_hash == sha256_text(generation.text),
            "usage_matches": call is not None
            and call.input_tokens == generation.input_tokens
            and call.output_tokens == generation.output_tokens
            and call.latency_ms == generation.latency_ms,
            "run_and_profile_match": call is not None
            and call.run_id == generation.run_id
            and call.provider_profile_hash == generation.provider_profile_hash,
            "provider_metadata_matches": call is not None
            and metadata.get("provider") == call.provider
            and metadata.get("gateway") == call.gateway
            and metadata.get("requested_model") == call.requested_model
            and metadata.get("returned_model") == call.returned_model
            and metadata.get("provider_profile_hash") == call.provider_profile_hash
            and metadata.get("provider_request_id") == call.provider_request_id,
        }
        links.append(
            {
                "generation_id": generation.generation_id,
                "call_id": call_id or None,
                "passed": all(link_checks.values()),
                "checks": link_checks,
            }
        )

    call_ids = set(call_id_counts)
    linked_call_ids = {value for value in generation_call_ids if value}
    passed = (
        len(generations) == len(calls)
        and len(call_ids) == len(calls)
        and len(linked_call_ids) == len(generations)
        and linked_call_ids == call_ids
        and all(link["passed"] for link in links)
    )
    return {
        "passed": passed,
        "generation_count": len(generations),
        "call_count": len(calls),
        "duplicate_call_ids": sorted(
            call_id for call_id, count in call_id_counts.items() if count > 1
        ),
        "duplicate_generation_call_ids": sorted(
            call_id
            for call_id, count in generation_call_id_counts.items()
            if call_id and count > 1
        ),
        "orphan_call_ids": sorted(call_ids - linked_call_ids),
        "missing_call_ids": sum(not value for value in generation_call_ids),
        "links": links,
    }
