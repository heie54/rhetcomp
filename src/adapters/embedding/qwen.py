from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from src.adapters.config import ProviderProfile
from src.adapters.embedding.base import EmbeddingRequest, EmbeddingResponse
from src.adapters.http import HttpTransport, ProviderTransportError, post_json
from src.adapters.records import ProviderCallArtifact, ProviderCallRecorder, new_call_id
from src.common.jsonio import read_json, sha256_json, sha256_text, write_json


@dataclass(frozen=True)
class CachedEmbedding:
    vector: tuple[float, ...]
    origin_call_id: str | None
    origin_provider_request_id: str | None
    provider_profile_hash: str | None


class EmbeddingCache:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    @staticmethod
    def key(model: str, dimensions: int, text: str) -> str:
        return sha256_json(
            {
                "embedding_model": model,
                "dimensions": dimensions,
                "embedding_text_hash": sha256_text(text),
            }
        )

    def get(self, model: str, dimensions: int, text: str) -> tuple[float, ...] | None:
        entry = self.get_entry(model, dimensions, text)
        return None if entry is None else entry.vector

    def get_entry(
        self, model: str, dimensions: int, text: str
    ) -> CachedEmbedding | None:
        path = self.root / f"{self.key(model, dimensions, text)}.json"
        if not path.exists():
            return None
        payload = read_json(path)
        if (
            payload.get("model") != model
            or payload.get("dimensions") != dimensions
            or payload.get("embedding_text_hash") != sha256_text(text)
        ):
            raise ValueError(f"Embedding cache metadata mismatch: {path}")
        vector = payload.get("vector")
        if not isinstance(vector, list) or len(vector) != dimensions:
            raise ValueError(f"Embedding cache vector dimension mismatch: {path}")
        return CachedEmbedding(
            vector=tuple(float(value) for value in vector),
            origin_call_id=payload.get("origin_call_id"),
            origin_provider_request_id=payload.get("origin_provider_request_id"),
            provider_profile_hash=payload.get("provider_profile_hash"),
        )

    def put(
        self,
        model: str,
        dimensions: int,
        text: str,
        vector: tuple[float, ...],
        *,
        origin_call_id: str | None = None,
        origin_provider_request_id: str | None = None,
        provider_profile_hash: str | None = None,
    ) -> Path:
        if len(vector) != dimensions:
            raise ValueError("Cannot cache vector with wrong dimensions")
        path = self.root / f"{self.key(model, dimensions, text)}.json"
        write_json(
            path,
            {
                "model": model,
                "dimensions": dimensions,
                "embedding_text_hash": sha256_text(text),
                "vector": list(vector),
                "origin_call_id": origin_call_id,
                "origin_provider_request_id": origin_provider_request_id,
                "provider_profile_hash": provider_profile_hash,
            },
        )
        return path


class QwenEmbeddingAdapter:
    def __init__(
        self,
        profile: ProviderProfile,
        api_key: str,
        base_url: str,
        *,
        cache: EmbeddingCache,
        recorder: ProviderCallRecorder | None = None,
        transport: HttpTransport = post_json,
        timeout_seconds: float = 120.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if profile.provider != "qwen" or profile.protocol != "openai_embeddings":
            raise ValueError("QwenEmbeddingAdapter requires a Qwen embeddings profile")
        if not api_key.strip() or not base_url.strip():
            raise ValueError("Qwen API key and base URL are required")
        self.profile = profile
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._cache = cache
        self._recorder = recorder
        self._transport = transport
        self._timeout_seconds = timeout_seconds
        self._sleep = sleep

    @classmethod
    def from_env(
        cls,
        profile: ProviderProfile,
        *,
        cache: EmbeddingCache,
        recorder: ProviderCallRecorder | None = None,
        environ: Mapping[str, str] | None = None,
        **kwargs: Any,
    ) -> "QwenEmbeddingAdapter":
        env = os.environ if environ is None else environ
        missing = [
            name
            for name in ("RHETCOMP_QWEN_API_KEY", "RHETCOMP_QWEN_BASE_URL")
            if not env.get(name)
        ]
        if missing:
            raise RuntimeError(f"Missing Qwen environment variables: {', '.join(missing)}")
        return cls(
            profile,
            env["RHETCOMP_QWEN_API_KEY"],
            env["RHETCOMP_QWEN_BASE_URL"],
            cache=cache,
            recorder=recorder,
            **kwargs,
        )

    @property
    def model_name(self) -> str:
        return self.profile.model

    def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        if request.dimensions != self.profile.dimensions:
            raise ValueError(
                f"Embedding dimensions must match profile: {request.dimensions} != {self.profile.dimensions}"
            )
        if request.output_type != self.profile.output_type:
            raise ValueError("Embedding output type must match the provider profile")

        vectors_by_text: dict[str, tuple[float, ...]] = {}
        cache_entries_by_text: dict[str, CachedEmbedding] = {}
        cache_hits = 0
        missing_texts: list[str] = []
        for text in request.inputs:
            if text in vectors_by_text or text in missing_texts:
                cache_hits += 1
                continue
            cached = self._cache.get_entry(self.profile.model, request.dimensions, text)
            provenance_complete = bool(
                cached
                and cached.origin_call_id
                and cached.origin_provider_request_id
                and cached.provider_profile_hash == self.profile.profile_hash
            )
            if cached is None or (request.run_mode == "formal" and not provenance_complete):
                missing_texts.append(text)
            else:
                vectors_by_text[text] = cached.vector
                cache_entries_by_text[text] = cached
                cache_hits += 1

        max_batch_size = int(self.profile.max_batch_size or 0)
        if max_batch_size <= 0:
            raise ValueError("Qwen embedding profile requires a positive max batch size")
        if len(missing_texts) > max_batch_size:
            batch_responses: list[EmbeddingResponse] = []
            for start in range(0, len(missing_texts), max_batch_size):
                chunk = tuple(missing_texts[start : start + max_batch_size])
                batch_responses.append(
                    self.embed(
                        EmbeddingRequest(
                            inputs=chunk,
                            dimensions=request.dimensions,
                            output_type=request.output_type,
                            run_id=request.run_id,
                            role=request.role,
                            run_mode=request.run_mode,
                            config_hash=request.config_hash,
                            data_manifest_hash=request.data_manifest_hash,
                        )
                    )
                )
            for text in missing_texts:
                entry = self._cache.get_entry(self.profile.model, request.dimensions, text)
                if entry is None:
                    raise RuntimeError("Embedding batch completed without populating its cache")
                vectors_by_text[text] = entry.vector
            returned_models = {response.model for response in batch_responses}
            if len(returned_models) != 1:
                raise ValueError("Qwen embedding batches returned inconsistent model names")
            call_ids = tuple(
                call_id
                for response in batch_responses
                for call_id in response.metadata.get(
                    "call_ids", (response.metadata.get("call_id"),)
                )
                if call_id
            )
            provider_request_ids = tuple(
                request_id
                for response in batch_responses
                for request_id in response.metadata.get(
                    "provider_request_ids", (response.request_id,)
                )
                if request_id
            )
            return EmbeddingResponse(
                vectors=tuple(vectors_by_text[text] for text in request.inputs),
                model=next(iter(returned_models)),
                dimensions=request.dimensions,
                input_tokens=sum(response.input_tokens for response in batch_responses),
                request_id=provider_request_ids[0] if provider_request_ids else None,
                latency_ms=sum(response.latency_ms for response in batch_responses),
                metadata={
                    "provider": "qwen",
                    "gateway": self.profile.gateway,
                    "requested_model": self.profile.model,
                    "returned_model": next(iter(returned_models)),
                    "provider_profile_hash": self.profile.profile_hash,
                    "cache_hits": cache_hits,
                    "cache_misses": len(missing_texts),
                    "retry_count": sum(
                        int(response.metadata.get("retry_count", 0))
                        for response in batch_responses
                    ),
                    "call_id": call_ids[0] if call_ids else None,
                    "call_ids": call_ids,
                    "provider_request_ids": provider_request_ids,
                    "network_batch_count": len(batch_responses),
                    "execution_kind": "network_batched",
                },
            )

        if not missing_texts:
            call_id = new_call_id()
            vectors = tuple(vectors_by_text[text] for text in request.inputs)
            origin_call_ids = tuple(
                sorted({entry.origin_call_id for entry in cache_entries_by_text.values() if entry.origin_call_id})
            )
            origin_request_ids = tuple(
                sorted(
                    {
                        entry.origin_provider_request_id
                        for entry in cache_entries_by_text.values()
                        if entry.origin_provider_request_id
                    }
                )
            )
            if self._recorder:
                artifact = self._artifact(
                    call_id=call_id,
                    request=request,
                    payload={"input": list(request.inputs)},
                    returned_model=self.profile.model,
                    response_hash=sha256_json(vectors),
                    input_tokens=0,
                    latency_ms=0,
                    provider_request_id=None,
                    retry_count=0,
                    status="success",
                    execution_kind="cache",
                    cache_origin_call_ids=origin_call_ids,
                    cache_origin_provider_request_ids=origin_request_ids,
                )
                self._recorder.record(artifact)
            return EmbeddingResponse(
                vectors=vectors,
                model=self.profile.model,
                dimensions=request.dimensions,
                input_tokens=0,
                request_id=None,
                latency_ms=0,
                metadata={
                    "provider": "qwen",
                    "gateway": self.profile.gateway,
                    "provider_profile_hash": self.profile.profile_hash,
                    "cache_hits": cache_hits,
                    "cache_misses": 0,
                    "retry_count": 0,
                    "call_id": call_id,
                    "execution_kind": "cache",
                    "cache_origin_call_ids": origin_call_ids,
                    "cache_origin_provider_request_ids": origin_request_ids,
                    "cache_provenance_complete": bool(origin_call_ids and origin_request_ids),
                },
            )

        payload: dict[str, Any] = {
            "model": self.profile.model,
            "input": missing_texts,
            "dimensions": request.dimensions,
        }
        call_id = new_call_id()
        started = time.perf_counter()
        retry_count = 0
        try:
            while True:
                try:
                    http_response = self._transport(
                        f"{self._base_url}/embeddings",
                        {
                            "Authorization": f"Bearer {self._api_key}",
                            "Content-Type": "application/json",
                            "Accept": "application/json",
                        },
                        payload,
                        self._timeout_seconds,
                    )
                    break
                except ProviderTransportError as exc:
                    if not exc.retryable or retry_count >= self.profile.max_retries:
                        raise
                    retry_count += 1
                    self._sleep(0.25 * (2 ** (retry_count - 1)))
            response_payload = http_response.payload
            data = response_payload.get("data")
            if not isinstance(data, list) or len(data) != len(missing_texts):
                raise ValueError("Qwen response data count does not match input count")
            ordered: list[tuple[float, ...] | None] = [None] * len(missing_texts)
            for position, item in enumerate(data):
                if not isinstance(item, dict) or not isinstance(item.get("embedding"), list):
                    raise ValueError("Qwen response contains an invalid embedding item")
                index = int(item.get("index", position))
                if index < 0 or index >= len(ordered) or ordered[index] is not None:
                    raise ValueError("Qwen response contains an invalid embedding index")
                vector = tuple(float(value) for value in item["embedding"])
                if len(vector) != request.dimensions:
                    raise ValueError(
                        f"Qwen returned {len(vector)} dimensions; expected {request.dimensions}"
                    )
                ordered[index] = vector
            if any(vector is None for vector in ordered):
                raise ValueError("Qwen response omitted an embedding index")
            usage = response_payload.get("usage")
            usage = usage if isinstance(usage, dict) else {}
            input_tokens = int(usage.get("prompt_tokens", usage.get("total_tokens", 0)))
            raw_returned_model = response_payload.get("model")
            if request.run_mode == "formal" and not (
                isinstance(raw_returned_model, str) and raw_returned_model.strip()
            ):
                raise ValueError("Formal Qwen response requires an explicit returned model")
            returned_model = str(raw_returned_model or self.profile.model)
            provider_request_id = str(
                http_response.headers.get("x-request-id") or response_payload.get("id") or ""
            ) or None
            if request.run_mode == "formal" and returned_model != self.profile.model:
                raise ValueError(
                    f"Formal Qwen returned model mismatch: {returned_model} != {self.profile.model}"
                )
            if request.run_mode == "formal" and not provider_request_id:
                raise ValueError("Formal Qwen response requires a provider request ID")
            for text, vector in zip(missing_texts, ordered, strict=True):
                assert vector is not None
                vectors_by_text[text] = vector
                self._cache.put(
                    self.profile.model,
                    request.dimensions,
                    text,
                    vector,
                    origin_call_id=call_id,
                    origin_provider_request_id=provider_request_id,
                    provider_profile_hash=self.profile.profile_hash,
                )
            latency_ms = round((time.perf_counter() - started) * 1000)
            network_vectors = tuple(vector for vector in ordered if vector is not None)
            artifact = self._artifact(
                call_id=call_id,
                request=request,
                payload=payload,
                returned_model=returned_model,
                response_hash=sha256_json(network_vectors),
                input_tokens=input_tokens,
                latency_ms=latency_ms,
                provider_request_id=provider_request_id,
                retry_count=retry_count,
                status="success",
            )
            if self._recorder:
                self._recorder.record(artifact)
            return EmbeddingResponse(
                vectors=tuple(vectors_by_text[text] for text in request.inputs),
                model=returned_model,
                dimensions=request.dimensions,
                input_tokens=input_tokens,
                request_id=provider_request_id,
                latency_ms=latency_ms,
                metadata={
                    "provider": "qwen",
                    "gateway": self.profile.gateway,
                    "requested_model": self.profile.model,
                    "returned_model": returned_model,
                    "provider_profile_hash": self.profile.profile_hash,
                    "cache_hits": cache_hits,
                    "cache_misses": len(missing_texts),
                    "retry_count": retry_count,
                    "call_id": call_id,
                    "execution_kind": "network",
                },
            )
        except Exception:
            latency_ms = round((time.perf_counter() - started) * 1000)
            if self._recorder:
                self._recorder.record(
                    self._artifact(
                        call_id=call_id,
                        request=request,
                        payload=payload,
                        returned_model=self.profile.model,
                        response_hash=sha256_text(""),
                        input_tokens=0,
                        latency_ms=latency_ms,
                        provider_request_id=None,
                        retry_count=retry_count,
                        status="failed",
                    )
                )
            raise

    def _artifact(
        self,
        *,
        call_id: str,
        request: EmbeddingRequest,
        payload: Mapping[str, Any],
        returned_model: str,
        response_hash: str,
        input_tokens: int,
        latency_ms: int,
        provider_request_id: str | None,
        retry_count: int,
        status: str,
        execution_kind: str = "network",
        cache_origin_call_ids: tuple[str, ...] = (),
        cache_origin_provider_request_ids: tuple[str, ...] = (),
    ) -> ProviderCallArtifact:
        hashed_inputs = [sha256_text(text) for text in payload["input"]]
        return ProviderCallArtifact(
            call_id=call_id,
            run_id=request.run_id,
            role=request.role,
            provider="qwen",
            gateway=self.profile.gateway,
            requested_model=self.profile.model,
            returned_model=returned_model,
            provider_profile_hash=self.profile.profile_hash,
            thinking_mode="not_applicable",
            reasoning_effort=None,
            prompt_hash=sha256_text(""),
            input_hash=sha256_json(
                {
                    "model": self.profile.model,
                    "dimensions": request.dimensions,
                    "embedding_text_hashes": hashed_inputs,
                }
            ),
            response_hash=response_hash,
            input_tokens=input_tokens,
            output_tokens=0,
            latency_ms=latency_ms,
            system_fingerprint=None,
            provider_request_id=provider_request_id,
            retry_count=retry_count,
            status=status,
            run_mode=request.run_mode,
            config_hash=request.config_hash,
            data_manifest_hash=request.data_manifest_hash,
            execution_kind=execution_kind,
            cache_origin_call_ids=cache_origin_call_ids,
            cache_origin_provider_request_ids=cache_origin_provider_request_ids,
        )
