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
from src.audits.gate2_formal import load_call_artifacts
from src.audits.gate3 import build_cost_record
from src.audits.gate3_formal import blocked_gate3_formal_report, build_gate3_formal_report
from src.budget.formal_tokenizer import load_formal_tokenizer
from src.common.jsonio import read_json, write_json
from src.compilers.config import load_compiler_settings
from src.compilers.experience.pipeline import compile_experience_library
from src.compilers.formal_representations import (
    ComputeEnvelope,
    compile_guideline_formal,
    compile_raw_formal,
    compile_summary_formal,
    representation_from_experience_result,
)
from src.config import load_config
from src.domain.models import SourcePaper
from src.formal_metadata import build_formal_metadata


ROOT = Path(__file__).resolve().parents[2]


def _resolve(root: Path, configured: str) -> Path:
    path = Path(configured)
    return path if path.is_absolute() else root / path


def _blocked(path: Path, reason: str, detail: Any) -> tuple[dict[str, Any], Path]:
    report = blocked_gate3_formal_report(reason, detail)
    write_json(path, report)
    return report, path


def compile_representations_formal(
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
        raise ValueError("Stage 3R requires formal dataset and budget configs")
    base_run_id = str(dataset["run_id"])
    paths = dataset["paths"]
    audit_path = _resolve(root, paths["audits"]) / "gate3_formal.json"
    gate2_path = _resolve(root, paths["audits"]) / "gate2_formal.json"
    if not gate2_path.exists():
        return _blocked(audit_path, "gate2r_audit_missing", str(gate2_path))
    gate2 = read_json(gate2_path)
    if gate2.get("status") != "PASS":
        return _blocked(audit_path, "gate2r_not_passed", gate2)
    if not execute_live:
        return _blocked(
            audit_path,
            "live_execution_not_authorized",
            "rerun with --execute-live after reviewing paid provider use",
        )
    attempt_id = f"attempt_{uuid4().hex[:12]}"
    run_id = f"{base_run_id}:stage3r:{attempt_id}"

    acl_manifest = read_json(_resolve(root, paths["manifests"]) / "acl_pilot.json")
    target_manifest = read_json(
        _resolve(root, paths["manifests"]) / "nc_physics_pilot.json"
    )
    source_root = _resolve(root, paths["source_normalized"])
    sources = [
        SourcePaper.from_dict(read_json(source_root / f"{entry['source_id']}.json"))
        for entry in acl_manifest["entries"]
    ]
    settings = load_compiler_settings(compiler_path)
    tokenizer = load_formal_tokenizer(budget_path, root)
    limit = int(budget["writing_condition_tokens"])
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
    if (
        gate2.get("config_hash") != metadata.config_hash
        or gate2.get("data_manifest_hash") != metadata.data_manifest_hash
    ):
        return _blocked(
            audit_path,
            "gate2r_metadata_mismatch",
            {
                "gate2r_config_hash": gate2.get("config_hash"),
                "current_config_hash": metadata.config_hash,
                "gate2r_data_manifest_hash": gate2.get("data_manifest_hash"),
                "current_data_manifest_hash": metadata.data_manifest_hash,
            },
        )
    compiler_profile_hash = profiles.require("deepseek_compiler_v1").profile_hash
    costs_root = _resolve(root, paths["costs"])
    stage_calls_root = costs_root / "stage3r" / attempt_id
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
        return _blocked(audit_path, "provider_environment_missing", str(exc))

    experience_result = compile_experience_library(
        sources,
        settings,
        tokenizer,
        limit,
        adapter=chat,
        run_mode="formal",
        run_id=run_id,
        embedding_adapter=embedding,
        config_hash=metadata.config_hash,
        data_manifest_hash=metadata.data_manifest_hash,
    )
    experience, experience_metrics = representation_from_experience_result(
        experience_result,
        settings,
        tokenizer,
        metadata,
        compiler_profile_hash,
    )
    raw, raw_metrics = compile_raw_formal(
        sources, settings, tokenizer, limit, metadata, compiler_profile_hash
    )
    summary, summary_metrics = compile_summary_formal(
        sources,
        settings,
        tokenizer,
        limit,
        chat,
        run_id=run_id,
        metadata=metadata,
        provider_profile_hash=compiler_profile_hash,
    )
    guideline, guideline_metrics = compile_guideline_formal(
        sources,
        settings,
        tokenizer,
        limit,
        chat,
        ComputeEnvelope(
            calls=experience.compiler_calls,
            input_tokens=experience.compiler_input_tokens,
            output_tokens=experience.compiler_output_tokens,
        ),
        run_id=run_id,
        metadata=metadata,
        provider_profile_hash=compiler_profile_hash,
    )
    representations = {
        "raw": raw,
        "summary": summary,
        "guideline": guideline,
        "experience": experience,
    }
    metrics = {
        "raw": raw_metrics,
        "summary": summary_metrics,
        "guideline": guideline_metrics,
        "experience": experience_metrics,
    }
    representations_root = _resolve(root, paths["representations"]) / "stage3r" / attempt_id
    for name, artifact in representations.items():
        write_json(representations_root / f"{name}.json", artifact.to_dict())
    write_json(
        representations_root / "representation_budgets_formal.json",
        {
            "run_mode": "formal",
            "run_id": run_id,
            "attempt_id": attempt_id,
            "config_hash": metadata.config_hash,
            "data_manifest_hash": metadata.data_manifest_hash,
            "provider_profile_hash": compiler_profile_hash,
            "representations": {name: item.to_dict() for name, item in metrics.items()},
        },
    )
    compiler_costs_path = stage_calls_root / "compiler_costs.json"
    write_json(
        compiler_costs_path,
        {
            "run_id": run_id,
            "run_mode": "formal",
            "attempt_id": attempt_id,
            "config_hash": metadata.config_hash,
            "data_manifest_hash": metadata.data_manifest_hash,
            "provider_profile_hash": compiler_profile_hash,
            "source_corpus_hash": raw.source_corpus_hash,
            "representations": {
                name: build_cost_record(artifact, mode="formal_real_provider")
                for name, artifact in representations.items()
            },
            "compute_envelope": {
                "experience": {
                    "calls": experience.compiler_calls,
                    "input_tokens": experience.compiler_input_tokens,
                    "output_tokens": experience.compiler_output_tokens,
                },
                "guideline": {
                    "calls": guideline.compiler_calls,
                    "input_tokens": guideline.compiler_input_tokens,
                    "output_tokens": guideline.compiler_output_tokens,
                },
            },
        },
    )
    report = build_gate3_formal_report(
        representations,
        metrics,
        load_call_artifacts(stage_calls_root),
        gate2_report=gate2,
        writing_condition_tokens=limit,
        run_id=run_id,
    )
    report.update(
        {
            "attempt_id": attempt_id,
            "call_artifact_root": stage_calls_root.relative_to(root).as_posix(),
            "representation_artifact_root": representations_root.relative_to(root).as_posix(),
            "cost_artifact_path": compiler_costs_path.relative_to(root).as_posix(),
        }
    )
    write_json(audit_path, report)
    return report, audit_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compile formal Stage 3R representations")
    parser.add_argument("--dataset-config", default="configs/dataset_formal_pilot.yaml")
    parser.add_argument("--compiler-config", default="configs/compiler_formal.yaml")
    parser.add_argument("--budget-config", default="configs/budget_formal.yaml")
    parser.add_argument("--provider-config", default="configs/providers.yaml")
    parser.add_argument("--execute-live", action="store_true")
    args = parser.parse_args(argv)
    report, path = compile_representations_formal(
        args.dataset_config,
        args.compiler_config,
        args.budget_config,
        args.provider_config,
        execute_live=args.execute_live,
    )
    print(f"GATE_3R={report['status']}")
    print(f"AUDIT={path}")
    return 0 if report["status"] == "PASS" else (2 if report["status"] == "BLOCKED" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
