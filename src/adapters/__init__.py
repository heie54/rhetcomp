"""Thin chat and embedding provider contracts for mechanics and formal execution."""

from src.adapters.embedding.base import EmbeddingAdapter, EmbeddingRequest, EmbeddingResponse
from src.adapters.model import ChatModelAdapter, ModelAdapter, ModelRequest, ModelResponse
from src.adapters.records import ProviderCallArtifact, ProviderCallRecorder

__all__ = [
    "ChatModelAdapter",
    "ModelAdapter",
    "ModelRequest",
    "ModelResponse",
    "EmbeddingAdapter",
    "EmbeddingRequest",
    "EmbeddingResponse",
    "ProviderCallArtifact",
    "ProviderCallRecorder",
]
