from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.adapters.config import ProviderProfiles
from src.common.jsonio import sha256_json


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_formal_freeze_manifest(
    *,
    root: Path,
    run_id: str,
    attempt_id: str,
    providers: ProviderProfiles,
    acl_manifest: Mapping[str, Any],
    target_manifest: Mapping[str, Any],
    tokenizer_manifest: Mapping[str, Any],
    frozen_files: Sequence[Path],
    artifact_files: Mapping[str, Sequence[Path]],
    generation_ids: Sequence[str],
    evaluation_id: str,
) -> dict[str, Any]:
    file_hashes = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(frozen_files, key=lambda item: item.as_posix())
    }
    artifact_file_hashes = {
        category: {
            path.relative_to(root).as_posix(): sha256_file(path)
            for path in sorted(paths, key=lambda item: item.as_posix())
        }
        for category, paths in sorted(artifact_files.items())
    }
    compiler = providers.require("deepseek_compiler_v1")
    writer = providers.require("deepseek_writer_v1")
    embedding = providers.require("qwen_embedding_v1")
    manifest = {
        "freeze_version": "formal-pilot-freeze-v1",
        "freeze_kind": "formal_pilot",
        "run_id": run_id,
        "attempt_id": attempt_id,
        "provider_config_version": providers.config_version,
        "provider_profiles": {
            name: {
                "profile_hash": profile.profile_hash,
                "provider": profile.provider,
                "model": profile.model,
                "thinking_enabled": profile.thinking_enabled,
                "reasoning_effort": profile.reasoning_effort,
                "temperature": profile.temperature,
                "dimensions": profile.dimensions,
            }
            for name, profile in sorted(providers.profiles.items())
        },
        "model_ids": {
            "compiler": compiler.model,
            "writer": writer.model,
            "embedding": embedding.model,
        },
        "thinking_modes": {
            "compiler": compiler.thinking_enabled,
            "writer": writer.thinking_enabled,
        },
        "embedding": {"model": embedding.model, "dimensions": embedding.dimensions},
        "tokenizer": {
            "model_repo": tokenizer_manifest.get("model_repo"),
            "revision": tokenizer_manifest.get("resolved_revision")
            or tokenizer_manifest.get("configured_revision"),
            "asset_hash": tokenizer_manifest.get("tokenizer_hash"),
            "tokenizer_version": tokenizer_manifest.get("tokenizer_version"),
        },
        "source_corpus_manifest_hash": acl_manifest.get("manifest_hash"),
        "source_corpus_hash": acl_manifest.get("source_corpus_hash"),
        "target_manifest_hash": target_manifest.get("manifest_hash"),
        "target_split": target_manifest.get("source_split"),
        "official_test_accessed": target_manifest.get("official_test_accessed"),
        "compute_match_rules": {
            "call_count_difference": "<= max(1, ceil(experience_calls * 0.10))",
            "input_token_difference_ratio": "<= 0.15",
        },
        "budget_tokens": {"target_evidence": 8000, "writing_condition": 4000},
        "frozen_file_hashes": file_hashes,
        "artifact_file_hashes": artifact_file_hashes,
        "schema_authority": "src/domain/schemas.py",
        "generation_ids": sorted(generation_ids),
        "evaluation_ids": [evaluation_id],
        "post_freeze_policy": "no changes before full experiment",
    }
    manifest["freeze_manifest_hash"] = sha256_json(manifest)
    return manifest


def validate_freeze_manifest(manifest: Mapping[str, Any]) -> bool:
    expected = manifest.get("freeze_manifest_hash")
    payload = {key: value for key, value in manifest.items() if key != "freeze_manifest_hash"}
    artifact_hashes = manifest.get("artifact_file_hashes")
    required_counts = {
        "data_manifests": 3,
        "upstream_audits": 4,
        "evidence_packs": 10,
        "generation_artifacts": 50,
        "order_manifests": 10,
        "writer_call_artifacts": 50,
        "writer_cost_artifacts": 1,
        "run_manifests": 1,
        "evaluation_artifacts": 1,
    }
    artifact_inventory_valid = (
        isinstance(artifact_hashes, dict)
        and all(
            isinstance(artifact_hashes.get(category), dict)
            and len(artifact_hashes[category]) == count
            for category, count in required_counts.items()
        )
        and isinstance(artifact_hashes.get("compiled_artifacts"), dict)
        and bool(artifact_hashes["compiled_artifacts"])
        and all(
            isinstance(digest, str) and len(digest) == 64
            for files in artifact_hashes.values()
            if isinstance(files, dict)
            for digest in files.values()
        )
    )
    generation_files = (
        artifact_hashes.get("generation_artifacts", {})
        if isinstance(artifact_hashes, dict)
        else {}
    )
    generation_names = {Path(path).name for path in generation_files}
    expected_generation_names = {
        f"{generation_id}.json" for generation_id in manifest.get("generation_ids", ())
    }
    return (
        isinstance(expected, str)
        and expected == sha256_json(payload)
        and manifest.get("freeze_kind") == "formal_pilot"
        and manifest.get("official_test_accessed") is False
        and len(manifest.get("generation_ids", ())) == 50
        and bool(manifest.get("evaluation_ids"))
        and bool(manifest.get("frozen_file_hashes"))
        and artifact_inventory_valid
        and generation_names == expected_generation_names
    )
