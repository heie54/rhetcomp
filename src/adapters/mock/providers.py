from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256

from src.adapters.embedding.base import EmbeddingRequest, EmbeddingResponse
from src.adapters.model import ModelRequest, ModelResponse


@dataclass
class MockChatAdapter:
    response_text: str = "{}"
    model: str = "mechanics-mock-chat"
    requests: list[ModelRequest] = field(default_factory=list)

    @property
    def model_name(self) -> str:
        return self.model

    def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return ModelResponse(
            text=self.response_text,
            input_tokens=max(1, len(request.system_prompt.split()) + len(request.user_prompt.split())),
            output_tokens=max(1, len(self.response_text.split())),
            latency_ms=0,
            metadata={"provider": "mock", "formal": False},
        )


@dataclass
class MockEmbeddingAdapter:
    model: str = "mechanics-mock-embedding"
    requests: list[EmbeddingRequest] = field(default_factory=list)

    @property
    def model_name(self) -> str:
        return self.model

    def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        self.requests.append(request)
        vectors: list[tuple[float, ...]] = []
        for text in request.inputs:
            digest = sha256(text.encode("utf-8")).digest()
            vector = tuple(
                (digest[index % len(digest)] / 127.5) - 1.0
                for index in range(request.dimensions)
            )
            vectors.append(vector)
        return EmbeddingResponse(
            vectors=tuple(vectors),
            model=self.model,
            dimensions=request.dimensions,
            input_tokens=sum(max(1, len(text.split())) for text in request.inputs),
            request_id=None,
            latency_ms=0,
            metadata={"provider": "mock", "formal": False},
        )
