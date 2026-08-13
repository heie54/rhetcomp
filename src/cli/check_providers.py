from __future__ import annotations

import argparse
import json
from pathlib import Path
from uuid import uuid4

from src.adapters.chat.deepseek import DeepSeekChatAdapter
from src.adapters.config import load_provider_profiles
from src.adapters.embedding.base import EmbeddingRequest
from src.adapters.embedding.qwen import EmbeddingCache, QwenEmbeddingAdapter
from src.adapters.environment import load_provider_environment
from src.adapters.model import ModelRequest
from src.adapters.records import ProviderCallRecorder
from src.common.jsonio import sha256_json


ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate provider config or run paid live smokes")
    parser.add_argument("--live", action="store_true", help="explicitly enable paid provider calls")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "providers.yaml")
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=ROOT / "artifacts" / "formal_pilot" / "provider_smoke",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=ROOT / "artifacts" / "formal_pilot" / "provider_smoke" / "embedding_cache",
    )
    parser.add_argument("--run-id", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    profiles = load_provider_profiles(args.config)
    profiles.require("deepseek_compiler_v1")
    profiles.require("deepseek_writer_v1")
    profiles.require("qwen_embedding_v1")
    print("PROVIDER_CONFIG=PASS")
    if not args.live:
        print("LIVE_REQUESTS=DISABLED")
        return 0

    run_id = args.run_id or f"provider-smoke-{uuid4().hex[:12]}"
    run_root = args.artifact_root / run_id
    recorder = ProviderCallRecorder(run_root)
    provider_env = load_provider_environment(ROOT)
    config_hash = sha256_json(
        {
            "config_version": profiles.config_version,
            "profiles": {
                name: profile.to_dict() for name, profile in profiles.profiles.items()
            },
        }
    )
    smoke_data_hash = sha256_json({"scope": "provider_smoke", "dataset": None})
    failures: list[str] = []

    deepseek_passed = True
    for profile_name, thinking_enabled, reasoning_effort, role in (
        ("deepseek_compiler_v1", True, "high", "provider_smoke_compiler"),
        ("deepseek_writer_v1", False, None, "provider_smoke_writer"),
    ):
        try:
            chat = DeepSeekChatAdapter.from_env(
                profiles.require(profile_name),
                recorder=recorder,
                environ=provider_env,
            )
            response = chat.generate(
                ModelRequest(
                    system_prompt="Return only a valid JSON object.",
                    user_prompt='Return exactly {"provider_smoke":"ok"}.',
                    max_output_tokens=64,
                    thinking_enabled=thinking_enabled,
                    reasoning_effort=reasoning_effort,
                    response_format="json_object",
                    run_id=run_id,
                    role=role,
                    run_mode="formal",
                    config_hash=config_hash,
                    data_manifest_hash=smoke_data_hash,
                )
            )
            payload = json.loads(response.text)
            if not isinstance(payload, dict) or payload.get("provider_smoke") != "ok":
                raise ValueError(
                    f"{profile_name} smoke response did not match the JSON contract"
                )
            print(f"{role.upper()}=PASS")
        except Exception as exc:
            deepseek_passed = False
            failures.append(f"{profile_name}: {exc}")
            print(f"{role.upper()}=FAIL")
    print(f"DEEPSEEK_PROVIDER={'PASS' if deepseek_passed else 'FAIL'}")

    try:
        embedding = QwenEmbeddingAdapter.from_env(
            profiles.require("qwen_embedding_v1"),
            cache=EmbeddingCache(args.cache_root),
            recorder=recorder,
            environ=provider_env,
        )
        response = embedding.embed(
            EmbeddingRequest(
                inputs=("Use evidence to motivate the research gap.",),
                dimensions=1024,
                output_type="dense",
                run_id=run_id,
                role="provider_smoke",
                run_mode="formal",
                config_hash=config_hash,
                data_manifest_hash=smoke_data_hash,
            )
        )
        if len(response.vectors) != 1 or len(response.vectors[0]) != 1024:
            raise ValueError("Qwen smoke response did not return one 1024-dimensional vector")
        print("QWEN_EMBEDDING=PASS")
    except Exception as exc:
        failures.append(f"Qwen: {exc}")
        print("QWEN_EMBEDDING=FAIL")

    print(f"RUN_ID={run_id}")
    print(f"CALL_ARTIFACT_ROOT={run_root}")
    for failure in failures:
        print(f"ERROR={failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
