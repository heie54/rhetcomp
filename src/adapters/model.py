from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol


@dataclass(frozen=True)
class ModelRequest:
    system_prompt: str
    user_prompt: str
    max_output_tokens: int
    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None
    thinking_enabled: bool | None = None
    reasoning_effort: str | None = None
    response_format: str = "text"
    stream: bool = False
    run_id: str = "unscoped"
    role: str = "unspecified"
    run_mode: str = "mechanics"
    config_hash: str | None = None
    data_manifest_hash: str | None = None


@dataclass(frozen=True)
class ModelResponse:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    metadata: Mapping[str, Any]


class ModelAdapter(Protocol):
    @property
    def model_name(self) -> str: ...

    def generate(self, request: ModelRequest) -> ModelResponse: ...


ChatModelAdapter = ModelAdapter
