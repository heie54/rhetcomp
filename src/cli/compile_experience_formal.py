from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.adapters.chat.deepseek import DeepSeekChatAdapter
from src.adapters.config import load_provider_profiles
from src.adapters.embedding.qwen import EmbeddingCache, QwenEmbeddingAdapter
from src.adapters.environment import load_provider_environment
from src.adapters.records import ProviderCallRecorder
from src.audits.gate1_formal import build_gate1_formal_report
from src.audits.gate2_formal import (
    blocked_gate2_formal_report,
    build_gate2_formal_report,
    load_call_artifacts,
)
from src.budget.formal_tokenizer import load_formal_tokenizer
from src.common.jsonio import read_json, write_json
from src.compilers.config import load_compiler_settings
from src.compilers.experience.pipeline import compile_experience_library
from src.config import load_config
from src.domain.models import SourcePaper
from src.formal_metadata import build_formal_metadata


ROOT = Path(__file__).resolve().parents[2]


def _resolve(root: Path, configured: str) -> Path:
    path = Path(configured)
    return path if path.is_absolute() else root / path


def _write_blocked(path: Path, reason: str, detail: Any) -> tuple[dict[str, Any], Path]:
    report = blocked_gate2_formal_report(reason, detail)
    write_json(path, report)
    return report, path


def compile_experience_formal(
    dataset_config_path: str | Path,
    compiler_config_path: str | Path,
    budget_config_path: str | Path,
    provider_config_path: str | Path,
    *,
    project_root: str | Path | None = None,
    execute_live: bool,
) -> tuple[dict[str, Any], Path]:
    dataset_path = Path(dataset_config_path).resolve()
    compiler_path = Path(compiler_config_path).resolve()
    budget_path = Path(budget_config_path).resolve()
    provider_path = Path(provider_config_path).resolve()
    root = Path(project_root).resolve() if project_root else ROOT
    dataset = load_config(dataset_path)
    budget = load_config(budget_path)
    if dataset.get("run_mode") != "formal" or budget.get("run_mode") != "formal":
        raise ValueError("Formal compiler requires formal dataset and budget configs")
    base_run_id = str(dataset["run_id"])
    paths = dataset["paths"]
    audits_root = _resolve(root, paths["audits"])
    audit_path = audits_root / "gate2_formal.json"
    gate1 = build_gate1_formal_report(root, dataset_path, budget_path)
    if gate1.get("status") != "PASS":
        return _write_blocked(audit_path, "gate1r_not_passed", gate1)
    if not execute_live:
        return _write_blocked(
            audit_path,
            "live_execution_not_authorized",
            "rerun with --execute-live after reviewing paid provider use",
        )
    attempt_id = f"attempt_{uuid4().hex[:12]}"
    run_id = f"{base_run_id}:stage2r:{attempt_id}"

    manifests_root = _resolve(root, paths["manifests"])
    manifest_path = manifests_root / "acl_pilot.json"
    acl_manifest = read_json(manifest_path)
    target_manifest = read_json(manifests_root / "nc_physics_pilot.json")
    source_root = _resolve(root, paths["source_normalized"])
    sources = [
        SourcePaper.from_dict(read_json(source_root / f"{entry['source_id']}.json"))
        for entry in acl_manifest["entries"]
    ]
    settings = load_compiler_settings(compiler_path)
    tokenizer = load_formal_tokenizer(budget_path, root)
    profiles = load_provider_profiles(provider_path)
    metadata = build_formal_metadata(
        run_id,
        configs={
            "dataset": dataset,
            "compiler": load_config(compiler_path),
            "budget": budget,
            "providers": load_config(provider_path),
        },
        manifests={
            "acl": acl_manifest["manifest_hash"],
            "targets": target_manifest["manifest_hash"],
        },
    )
    if gate1.get("data_manifest_hash") != metadata.data_manifest_hash:
        return _write_blocked(
            audit_path,
            "gate1r_data_manifest_mismatch",
            {
                "gate1r": gate1.get("data_manifest_hash"),
                "current": metadata.data_manifest_hash,
            },
        )
    costs_root = _resolve(root, paths["costs"])
    stage_calls_root = costs_root / "stage2r" / attempt_id
    recorder = ProviderCallRecorder(stage_calls_root)
    provider_env = load_provider_environment(root)
    try:
        chat = DeepSeekChatAdapter.from_env(
            profiles.require("deepseek_compiler_v1"), recorder=recorder, environ=provider_env
        )
        embedding = QwenEmbeddingAdapter.from_env(
            profiles.require("qwen_embedding_v1"),
            cache=EmbeddingCache(root / "artifacts" / "formal_pilot" / "embedding_cache"),
            recorder=recorder,
            environ=provider_env,
        )
    except RuntimeError as exc:
        return _write_blocked(audit_path, "provider_environment_missing", str(exc))

    result = compile_experience_library(
        sources,
        settings,
        tokenizer,
        int(budget["writing_condition_tokens"]),
        adapter=chat,
        run_mode="formal",
        run_id=run_id,
        embedding_adapter=embedding,
        config_hash=metadata.config_hash,
        data_manifest_hash=metadata.data_manifest_hash,
    )
    experiences_root = _resolve(root, paths["experiences"]) / "stage2r" / attempt_id
    write_json(
        experiences_root / "experience_library.json",
        {
            "run_mode": "formal",
            "run_id": run_id,
            "attempt_id": attempt_id,
            "config_hash": metadata.config_hash,
            "data_manifest_hash": metadata.data_manifest_hash,
            "provider_profile_hash": profiles.require("deepseek_compiler_v1").profile_hash,
            "source_corpus_hash": result.source_corpus_hash,
            "adapter_mode": result.adapter_mode,
            "embedding_backend": result.embedding_backend,
            "embedding_model": result.embedding_model,
            "experiences": [item.to_dict() for item in result.experiences],
            "derived_meta": [item.to_dict() for item in result.derived_meta],
            "library_content": result.library.content,
            "library_content_hash": result.library.content_hash,
            "library_content_tokens": result.library.content_tokens,
        },
    )
    write_json(experiences_root / "trace.json", list(result.trace))
    report = build_gate2_formal_report(
        result,
        settings,
        int(budget["writing_condition_tokens"]),
        sources=sources,
        acl_manifest=acl_manifest,
        gate1_report=gate1,
        call_artifacts=load_call_artifacts(stage_calls_root),
    )
    report.update(
        {
            "attempt_id": attempt_id,
            "call_artifact_root": stage_calls_root.relative_to(root).as_posix(),
            "experience_artifact_root": experiences_root.relative_to(root).as_posix(),
        }
    )
    write_json(audit_path, report)
    return report, audit_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compile the formal real Experience library")
    parser.add_argument("--dataset-config", default="configs/dataset_formal_pilot.yaml")
    parser.add_argument("--compiler-config", default="configs/compiler_formal.yaml")
    parser.add_argument("--budget-config", default="configs/budget_formal.yaml")
    parser.add_argument("--provider-config", default="configs/providers.yaml")
    parser.add_argument("--execute-live", action="store_true")
    args = parser.parse_args(argv)
    report, path = compile_experience_formal(
        args.dataset_config,
        args.compiler_config,
        args.budget_config,
        args.provider_config,
        execute_live=args.execute_live,
    )
    print(f"GATE_2R={report['status']}")
    print(f"AUDIT={path}")
    return 0 if report["status"] == "PASS" else (2 if report["status"] == "BLOCKED" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
