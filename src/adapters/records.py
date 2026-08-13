from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from src.common.jsonio import write_json


CALL_STATUSES = frozenset({"success", "failed"})


@dataclass(frozen=True)
class ProviderCallArtifact:
    call_id: str
    run_id: str
    role: str
    provider: str
    requested_model: str
    returned_model: str
    provider_profile_hash: str
    thinking_mode: str
    reasoning_effort: str | None
    prompt_hash: str
    input_hash: str
    response_hash: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    system_fingerprint: str | None
    provider_request_id: str | None
    retry_count: int
    status: str
    execution_kind: str = "network"
    run_mode: str = "mechanics"
    config_hash: str | None = None
    data_manifest_hash: str | None = None
    gateway: str | None = None
    cache_origin_call_ids: tuple[str, ...] = ()
    cache_origin_provider_request_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in CALL_STATUSES:
            raise ValueError(f"Invalid provider call status: {self.status}")
        if self.execution_kind not in {"network", "cache"}:
            raise ValueError(f"Invalid provider execution kind: {self.execution_kind}")
        if self.run_mode not in {"mechanics", "formal"}:
            raise ValueError(f"Invalid provider run mode: {self.run_mode}")
        if self.run_mode == "formal" and (
            not self.config_hash or not self.data_manifest_hash
        ):
            raise ValueError("Formal provider artifacts require config and data manifest hashes")
        if self.run_mode == "formal" and self.execution_kind == "cache" and (
            not self.cache_origin_call_ids
            or not self.cache_origin_provider_request_ids
        ):
            raise ValueError("Formal cache artifacts require real provider-call provenance")
        if min(self.input_tokens, self.output_tokens, self.latency_ms, self.retry_count) < 0:
            raise ValueError("Provider call counters cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProviderCallArtifact":
        value = dict(payload)
        # Backward-compatible mechanics artifacts created before formal metadata existed.
        if "run_mode" not in value:
            value["run_mode"] = "mechanics"
        for field_name in (
            "cache_origin_call_ids",
            "cache_origin_provider_request_ids",
        ):
            if field_name in value:
                value[field_name] = tuple(value[field_name])
        return cls(**value)


class ProviderCallRecorder:
    """Writes metadata-only call artifacts; prompts, responses, and reasoning are not stored."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def record(self, artifact: ProviderCallArtifact) -> Path:
        destination = self.root / "calls" / f"{artifact.call_id}.json"
        write_json(destination, artifact.to_dict())
        return destination


def new_call_id() -> str:
    return f"call_{uuid4().hex}"
