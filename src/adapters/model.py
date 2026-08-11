from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol


@dataclass(frozen=True)
class ModelRequest:
    system_prompt: str
    user_prompt: str
    max_output_tokens: int
    temperature: float
    top_p: float
    seed: int | None = None


@dataclass(frozen=True)
class ModelResponse:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    metadata: Mapping[str, str]


class ModelAdapter(Protocol):
    @property
    def model_name(self) -> str: ...

    def generate(self, request: ModelRequest) -> ModelResponse: ...
