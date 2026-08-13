from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol


@dataclass(frozen=True)
class EmbeddingRequest:
    inputs: tuple[str, ...]
    dimensions: int
    output_type: str = "dense"
    run_id: str = "unscoped"
    role: str = "embedding"
    run_mode: str = "mechanics"
    config_hash: str | None = None
    data_manifest_hash: str | None = None

    def __post_init__(self) -> None:
        if not self.inputs or any(not isinstance(value, str) or not value for value in self.inputs):
            raise ValueError("EmbeddingRequest requires non-empty text inputs")
        if self.dimensions <= 0:
            raise ValueError("Embedding dimensions must be positive")
        if self.output_type != "dense":
            raise ValueError("Only dense embeddings are supported")


@dataclass(frozen=True)
class EmbeddingResponse:
    vectors: tuple[tuple[float, ...], ...]
    model: str
    dimensions: int
    input_tokens: int
    request_id: str | None
    latency_ms: int
    metadata: Mapping[str, Any]


class EmbeddingAdapter(Protocol):
    @property
    def model_name(self) -> str: ...

    def embed(self, request: EmbeddingRequest) -> EmbeddingResponse: ...
