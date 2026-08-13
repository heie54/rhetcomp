from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from src.common.jsonio import sha256_json
from src.config import load_config


@dataclass(frozen=True)
class ProviderProfile:
    profile_id: str
    provider: str
    model: str
    protocol: str
    gateway: str | None = None
    user_agent: str | None = None
    thinking_enabled: bool | None = None
    reasoning_effort: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None
    stream: bool = False
    dimensions: int | None = None
    output_type: str | None = None
    max_retries: int = 3
    max_batch_size: int | None = None

    def __post_init__(self) -> None:
        for name in ("profile_id", "provider", "model", "protocol"):
            if not getattr(self, name):
                raise ValueError(f"Provider profile requires {name}")
        if self.user_agent is not None and not self.user_agent.strip():
            raise ValueError("Provider profile user_agent must be non-empty when set")
        if self.max_retries < 0 or self.max_retries > 3:
            raise ValueError("max_retries must be between 0 and 3")
        if self.stream:
            raise ValueError("Formal provider profiles must be non-streaming")
        if self.thinking_enabled:
            if self.reasoning_effort not in {"high", "max"}:
                raise ValueError("Thinking profiles require reasoning_effort high or max")
            if any(value is not None for value in (self.temperature, self.top_p, self.seed)):
                raise ValueError("Thinking profiles must omit temperature, top_p, and seed")
        if self.protocol == "openai_embeddings":
            if not self.dimensions or self.dimensions <= 0:
                raise ValueError("Embedding profiles require positive dimensions")
            if self.output_type != "dense":
                raise ValueError("Formal embedding profile must use dense output")
            if not self.max_batch_size or self.max_batch_size <= 0:
                raise ValueError("Embedding profiles require a positive max batch size")

    @property
    def profile_hash(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "provider": self.provider,
            "model": self.model,
            "protocol": self.protocol,
            "gateway": self.gateway,
            "user_agent": self.user_agent,
            "thinking_enabled": self.thinking_enabled,
            "reasoning_effort": self.reasoning_effort,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": self.seed,
            "stream": self.stream,
            "dimensions": self.dimensions,
            "output_type": self.output_type,
            "max_retries": self.max_retries,
            "max_batch_size": self.max_batch_size,
        }


@dataclass(frozen=True)
class ProviderProfiles:
    config_version: str
    profiles: Mapping[str, ProviderProfile]

    def require(self, name: str) -> ProviderProfile:
        try:
            return self.profiles[name]
        except KeyError as exc:
            raise KeyError(f"Unknown provider profile: {name}") from exc


def load_provider_profiles(path: str | Path) -> ProviderProfiles:
    raw = load_config(path)
    profiles_raw = raw.get("profiles")
    if not isinstance(profiles_raw, dict) or not profiles_raw:
        raise ValueError("Provider config requires a non-empty profiles object")
    profiles: dict[str, ProviderProfile] = {}
    for name, value in profiles_raw.items():
        if not isinstance(value, dict):
            raise ValueError(f"Provider profile {name} must be an object")
        payload = dict(value)
        payload.setdefault("profile_id", name)
        profile = ProviderProfile(**payload)
        if profile.profile_id != name:
            raise ValueError(f"Provider profile key/id mismatch: {name}")
        profiles[name] = profile
    return ProviderProfiles(config_version=str(raw["config_version"]), profiles=profiles)
