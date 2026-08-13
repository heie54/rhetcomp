from __future__ import annotations

import os
import time
from typing import Any, Callable, Mapping

from src.adapters.config import ProviderProfile
from src.adapters.http import HttpTransport, ProviderTransportError, post_json
from src.adapters.model import ModelRequest, ModelResponse
from src.adapters.records import ProviderCallArtifact, ProviderCallRecorder, new_call_id
from src.common.jsonio import sha256_json, sha256_text


class DeepSeekChatAdapter:
    def __init__(
        self,
        profile: ProviderProfile,
        api_key: str,
        base_url: str,
        *,
        recorder: ProviderCallRecorder | None = None,
        transport: HttpTransport = post_json,
        timeout_seconds: float = 120.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if profile.provider != "deepseek" or profile.protocol != "openai_chat_completions":
            raise ValueError("DeepSeekChatAdapter requires a DeepSeek chat-completions profile")
        if not api_key.strip() or not base_url.strip():
            raise ValueError("DeepSeek API key and base URL are required")
        self.profile = profile
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._recorder = recorder
        self._transport = transport
        self._timeout_seconds = timeout_seconds
        self._sleep = sleep

    @classmethod
    def from_env(
        cls,
        profile: ProviderProfile,
        *,
        recorder: ProviderCallRecorder | None = None,
        environ: Mapping[str, str] | None = None,
        **kwargs: Any,
    ) -> "DeepSeekChatAdapter":
        env = os.environ if environ is None else environ
        missing = [
            name
            for name in ("RHETCOMP_DEEPSEEK_API_KEY", "RHETCOMP_DEEPSEEK_BASE_URL")
            if not env.get(name)
        ]
        if missing:
            raise RuntimeError(f"Missing DeepSeek environment variables: {', '.join(missing)}")
        return cls(
            profile,
            env["RHETCOMP_DEEPSEEK_API_KEY"],
            env["RHETCOMP_DEEPSEEK_BASE_URL"],
            recorder=recorder,
            **kwargs,
        )

    @property
    def model_name(self) -> str:
        return self.profile.model

    def _request_parameters(self, request: ModelRequest) -> tuple[bool, str | None, Any, Any, Any]:
        thinking = (
            self.profile.thinking_enabled
            if request.thinking_enabled is None
            else request.thinking_enabled
        )
        thinking = bool(thinking)
        reasoning_effort = request.reasoning_effort or self.profile.reasoning_effort
        temperature = (
            self.profile.temperature if request.temperature is None else request.temperature
        )
        top_p = self.profile.top_p if request.top_p is None else request.top_p
        seed = self.profile.seed if request.seed is None else request.seed
        if thinking:
            if reasoning_effort not in {"high", "max"}:
                raise ValueError("Thinking requests require reasoning_effort high or max")
            if any(value is not None for value in (temperature, top_p, seed)):
                raise ValueError("Thinking requests must omit temperature, top_p, and seed")
        elif reasoning_effort is not None:
            raise ValueError("Non-thinking requests must omit reasoning_effort")
        return thinking, reasoning_effort, temperature, top_p, seed

    def _payload(self, request: ModelRequest) -> dict[str, Any]:
        if request.stream or self.profile.stream:
            raise ValueError("Formal DeepSeek requests must be non-streaming")
        if request.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        if request.response_format not in {"text", "json_object"}:
            raise ValueError(f"Unsupported response_format: {request.response_format}")
        thinking, effort, temperature, top_p, seed = self._request_parameters(request)
        payload: dict[str, Any] = {
            "model": self.profile.model,
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_prompt},
            ],
            "max_tokens": request.max_output_tokens,
            "stream": False,
            "thinking": {"type": "enabled" if thinking else "disabled"},
        }
        if effort is not None:
            payload["reasoning_effort"] = effort
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        if seed is not None:
            payload["seed"] = seed
        if request.response_format == "json_object":
            payload["response_format"] = {"type": "json_object"}
        return payload

    def generate(self, request: ModelRequest) -> ModelResponse:
        payload = self._payload(request)
        call_id = new_call_id()
        started = time.perf_counter()
        retry_count = 0
        try:
            while True:
                try:
                    headers = {
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                    }
                    if self.profile.user_agent:
                        headers["User-Agent"] = self.profile.user_agent
                    http_response = self._transport(
                        f"{self._base_url}/chat/completions",
                        headers,
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
            choices = response_payload.get("choices")
            if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                raise ValueError("DeepSeek response is missing choices[0]")
            message = choices[0].get("message")
            if not isinstance(message, dict) or not isinstance(message.get("content"), str):
                raise ValueError("DeepSeek response is missing message.content")
            usage = response_payload.get("usage")
            if not isinstance(usage, dict):
                raise ValueError("DeepSeek response is missing usage")
            raw_returned_model = response_payload.get("model")
            if request.run_mode == "formal" and not (
                isinstance(raw_returned_model, str) and raw_returned_model.strip()
            ):
                raise ValueError("Formal DeepSeek response requires an explicit returned model")
            returned_model = str(raw_returned_model or self.profile.model)
            provider_request_id = str(
                http_response.headers.get("x-request-id") or response_payload.get("id") or ""
            ) or None
            if request.run_mode == "formal" and returned_model != self.profile.model:
                raise ValueError(
                    f"Formal DeepSeek returned model mismatch: {returned_model} != {self.profile.model}"
                )
            if request.run_mode == "formal" and not provider_request_id:
                raise ValueError("Formal DeepSeek response requires a provider request ID")
            fingerprint = (
                str(response_payload["system_fingerprint"])
                if response_payload.get("system_fingerprint") is not None
                else None
            )
            finish_reason = choices[0].get("finish_reason")
            text = str(message["content"])
            latency_ms = round((time.perf_counter() - started) * 1000)
            input_tokens = int(usage.get("prompt_tokens", 0))
            output_tokens = int(usage.get("completion_tokens", 0))
            artifact = self._artifact(
                call_id=call_id,
                request=request,
                payload=payload,
                returned_model=returned_model,
                response_hash=sha256_text(text),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                fingerprint=fingerprint,
                provider_request_id=provider_request_id,
                retry_count=retry_count,
                status="success",
            )
            if self._recorder:
                self._recorder.record(artifact)
            return ModelResponse(
                text=text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                metadata={
                    "provider": self.profile.provider,
                    "gateway": self.profile.gateway,
                    "requested_model": self.profile.model,
                    "returned_model": returned_model,
                    "provider_profile_hash": self.profile.profile_hash,
                    "provider_request_id": provider_request_id,
                    "system_fingerprint": fingerprint,
                    "finish_reason": finish_reason,
                    "retry_count": retry_count,
                    "call_id": call_id,
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
                        output_tokens=0,
                        latency_ms=latency_ms,
                        fingerprint=None,
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
        request: ModelRequest,
        payload: Mapping[str, Any],
        returned_model: str,
        response_hash: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: int,
        fingerprint: str | None,
        provider_request_id: str | None,
        retry_count: int,
        status: str,
    ) -> ProviderCallArtifact:
        thinking = payload["thinking"]["type"]
        return ProviderCallArtifact(
            call_id=call_id,
            run_id=request.run_id,
            role=request.role,
            provider=self.profile.provider,
            gateway=self.profile.gateway,
            requested_model=self.profile.model,
            returned_model=returned_model,
            provider_profile_hash=self.profile.profile_hash,
            thinking_mode=str(thinking),
            reasoning_effort=payload.get("reasoning_effort"),
            prompt_hash=sha256_text(f"{request.system_prompt}\n\n{request.user_prompt}"),
            input_hash=sha256_json(payload),
            response_hash=response_hash,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            system_fingerprint=fingerprint,
            provider_request_id=provider_request_id,
            retry_count=retry_count,
            status=status,
            run_mode=request.run_mode,
            config_hash=request.config_hash,
            data_manifest_hash=request.data_manifest_hash,
        )
