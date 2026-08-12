from __future__ import annotations

from dataclasses import dataclass, field

from src.adapters.model import ModelAdapter, ModelRequest, ModelResponse


@dataclass
class CountingAdapter:
    """Wraps a ModelAdapter and accumulates call and token accounting (spec §2.10, §9)."""

    _inner: ModelAdapter
    calls: int = field(default=0)
    input_tokens: int = field(default=0)
    output_tokens: int = field(default=0)
    latency_ms: int = field(default=0)

    @property
    def model_name(self) -> str:
        return self._inner.model_name

    def generate(self, request: ModelRequest) -> ModelResponse:
        self.calls += 1
        response = self._inner.generate(request)
        self.input_tokens += response.input_tokens
        self.output_tokens += response.output_tokens
        self.latency_ms += response.latency_ms
        return response
