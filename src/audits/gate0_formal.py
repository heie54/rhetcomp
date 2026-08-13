from __future__ import annotations

from pathlib import Path
from typing import Any

from src.adapters.config import load_provider_profiles
from src.adapters.records import ProviderCallArtifact
from src.common.jsonio import read_json, write_json


REQUIRED_ENV_NAMES = {
    "RHETCOMP_DEEPSEEK_API_KEY",
    "RHETCOMP_DEEPSEEK_BASE_URL",
    "RHETCOMP_QWEN_API_KEY",
    "RHETCOMP_QWEN_BASE_URL",
}


def _check(name: str, passed: bool, detail: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def build_gate0_formal_report(root: Path) -> dict[str, Any]:
    providers = load_provider_profiles(root / "configs" / "providers.yaml")
    compiler = providers.require("deepseek_compiler_v1")
    writer = providers.require("deepseek_writer_v1")
    embedding = providers.require("qwen_embedding_v1")
    env_lines = [
        line.strip()
        for line in (root / ".env.example").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    env_names = {line.split("=", 1)[0] for line in env_lines}
    env_values_empty = all(line.endswith("=") for line in env_lines)
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8").lower()
    smoke_source = (root / "src" / "cli" / "check_providers.py").read_text(encoding="utf-8")

    sample = ProviderCallArtifact(
        call_id="call_gate0r_roundtrip",
        run_id="gate0r",
        role="audit",
        provider="deepseek",
        gateway="opencode_go",
        requested_model="deepseek-v4-flash",
        returned_model="deepseek-v4-flash",
        provider_profile_hash=compiler.profile_hash,
        thinking_mode="enabled",
        reasoning_effort="high",
        prompt_hash="a" * 64,
        input_hash="b" * 64,
        response_hash="c" * 64,
        input_tokens=1,
        output_tokens=1,
        latency_ms=1,
        system_fingerprint="fp_gate0r",
        provider_request_id="request_gate0r",
        retry_count=0,
        status="success",
    )
    round_trip = ProviderCallArtifact.from_dict(sample.to_dict()) == sample
    checks = [
        _check("provider_config_versioned", bool(providers.config_version), providers.config_version),
        _check(
            "compiler_profile_frozen",
            compiler.provider == "deepseek"
            and compiler.gateway == "opencode_go"
            and compiler.user_agent == "rhetcomp/0.1.0"
            and compiler.model == "deepseek-v4-flash"
            and compiler.thinking_enabled is True
            and compiler.reasoning_effort == "high",
            compiler.to_dict(),
        ),
        _check(
            "thinking_parameters_omitted",
            compiler.temperature is None and compiler.top_p is None and compiler.seed is None,
            {"temperature": compiler.temperature, "top_p": compiler.top_p, "seed": compiler.seed},
        ),
        _check(
            "writer_profile_frozen",
            writer.provider == "deepseek"
            and writer.gateway == "opencode_go"
            and writer.user_agent == "rhetcomp/0.1.0"
            and writer.model == "deepseek-v4-flash"
            and writer.thinking_enabled is False
            and writer.temperature == 0.0
            and writer.top_p is None
            and writer.seed is None,
            writer.to_dict(),
        ),
        _check(
            "embedding_profile_frozen",
            embedding.provider == "qwen"
            and embedding.gateway == "alibaba_model_studio"
            and embedding.model == "qwen3.7-text-embedding"
            and embedding.dimensions == 1024
            and embedding.max_batch_size == 20
            and embedding.output_type == "dense",
            embedding.to_dict(),
        ),
        _check("non_stream_profiles", not compiler.stream and not writer.stream, None),
        _check("call_artifact_round_trip", round_trip, sample.to_dict()),
        _check(
            "environment_only_secrets",
            env_names == REQUIRED_ENV_NAMES and env_values_empty,
            {"names": sorted(env_names), "values_empty": env_values_empty},
        ),
        _check("live_calls_explicit", '"--live"' in smoke_source, "check_providers --live"),
        _check(
            "no_orchestration_frameworks",
            all(name not in pyproject for name in ("langchain", "langgraph", "litellm")),
            "pyproject.toml",
        ),
    ]
    return {
        "gate": "0R",
        "run_mode": "formal",
        "config_version": providers.config_version,
        "status": "PASS" if all(check["passed"] for check in checks) else "FAIL",
        "checks": checks,
        "live_provider_smoke": "NOT_RUN_BY_AUDIT",
        "mechanics_regression": "REQUIRES_TEST_COMMAND_EVIDENCE",
    }


def write_gate0_formal_report(root: Path, destination: Path) -> dict[str, Any]:
    report = build_gate0_formal_report(root)
    write_json(destination, report)
    return read_json(destination)
