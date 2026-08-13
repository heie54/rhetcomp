from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.adapters.chat.deepseek import DeepSeekChatAdapter
from src.adapters.config import load_provider_profiles
from src.adapters.environment import load_provider_environment
from src.adapters.records import ProviderCallRecorder
from src.audits.gate2_formal import load_call_artifacts
from src.audits.gate4_formal import blocked_gate4_formal_report, build_gate4_formal_report
from src.budget.formal_tokenizer import load_formal_tokenizer
from src.common.jsonio import read_json, write_json
from src.config import load_config
from src.domain.models import RepresentationArtifact, TargetEvidencePack
from src.writer.config import WRITER_CONDITIONS, load_writer_settings
from src.writer.order import condition_order
from src.writer.writer import Writer
from src.formal_metadata import build_formal_metadata


ROOT = Path(__file__).resolve().parents[2]
CONDITION_TO_TYPE = {
    "evidence_only": None,
    "raw": "raw",
    "summary": "summary",
    "guideline": "guideline",
    "experience": "experience",
}


def _resolve(root: Path, configured: str) -> Path:
    path = Path(configured)
    return path if path.is_absolute() else root / path


def _upstream_artifact_root(
    root: Path, report: dict[str, Any], key: str, stage: str
) -> Path:
    configured = report.get(key)
    if not configured:
        raise ValueError(f"{stage} report is missing {key}")
    resolved_root = root.resolve()
    candidate = (resolved_root / str(configured)).resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"{stage} {key} escapes project root") from exc
    return candidate


def _blocked(path: Path, reason: str, detail: Any) -> tuple[dict[str, Any], Path]:
    report = blocked_gate4_formal_report(reason, detail)
    write_json(path, report)
    return report, path


def generate_writer_formal(
    dataset_config_path: str | Path,
    writer_config_path: str | Path,
    budget_config_path: str | Path,
    provider_config_path: str | Path,
    *,
    project_root: str | Path | None = None,
    execute_live: bool,
) -> tuple[dict[str, Any], Path]:
    dataset_path = Path(dataset_config_path).resolve()
    writer_path = Path(writer_config_path).resolve()
    budget_path = Path(budget_config_path).resolve()
    provider_path = Path(provider_config_path).resolve()
    root = Path(project_root).resolve() if project_root else ROOT
    dataset = load_config(dataset_path)
    paths = dataset["paths"]
    audit_path = _resolve(root, paths["audits"]) / "gate4_formal.json"
    gate3_path = _resolve(root, paths["audits"]) / "gate3_formal.json"
    if not gate3_path.exists():
        return _blocked(audit_path, "gate3r_audit_missing", str(gate3_path))
    gate3 = read_json(gate3_path)
    if gate3.get("status") != "PASS":
        return _blocked(audit_path, "gate3r_not_passed", gate3)
    if not execute_live:
        return _blocked(
            audit_path,
            "live_execution_not_authorized",
            "rerun with --execute-live after reviewing paid provider use",
        )
    attempt_id = f"attempt_{uuid4().hex[:12]}"

    manifests_root = _resolve(root, paths["manifests"])
    target_manifest = read_json(manifests_root / "nc_physics_pilot.json")
    acl_manifest = read_json(manifests_root / "acl_pilot.json")
    target_ids = sorted(entry["target_id"] for entry in target_manifest["entries"])
    if len(target_ids) != 10:
        return _blocked(audit_path, "formal_target_count_invalid", len(target_ids))
    target_id = target_ids[0]
    pack = TargetEvidencePack.from_dict(
        read_json(_resolve(root, paths["evidence_packs"]) / f"{target_id}.json")
    )
    try:
        representations_root = _upstream_artifact_root(
            root, gate3, "representation_artifact_root", "Gate 3R"
        )
    except ValueError as exc:
        return _blocked(audit_path, "gate3r_artifact_root_invalid", str(exc))
    representations = {
        condition: (
            None
            if representation_type is None
            else RepresentationArtifact.from_dict(
                read_json(representations_root / f"{representation_type}.json")
            )
        )
        for condition, representation_type in CONDITION_TO_TYPE.items()
    }
    settings = load_writer_settings(writer_path)
    tokenizer = load_formal_tokenizer(budget_path, root)
    profiles = load_provider_profiles(provider_path)
    writer_profile = profiles.require("deepseek_writer_v1")
    metadata = build_formal_metadata(
        str(dataset["run_id"]),
        configs={
            "dataset": dataset,
            "writer": load_config(writer_path),
            "budget": load_config(budget_path),
            "providers": load_config(provider_path),
        },
        manifests={
            "acl": acl_manifest["manifest_hash"],
            "targets": target_manifest["manifest_hash"],
        },
    )
    representation_config_hashes = {
        representation.config_hash
        for representation in representations.values()
        if representation is not None
    }
    representation_data_hashes = {
        representation.data_manifest_hash
        for representation in representations.values()
        if representation is not None
    }
    if (
        gate3.get("data_manifest_hash") != metadata.data_manifest_hash
        or representation_config_hashes != {gate3.get("config_hash")}
        or representation_data_hashes != {gate3.get("data_manifest_hash")}
    ):
        return _blocked(
            audit_path,
            "gate3r_metadata_mismatch",
            {
                "gate3r_config_hash": gate3.get("config_hash"),
                "representation_config_hashes": sorted(
                    value for value in representation_config_hashes if value
                ),
                "gate3r_data_manifest_hash": gate3.get("data_manifest_hash"),
                "current_data_manifest_hash": metadata.data_manifest_hash,
                "representation_data_manifest_hashes": sorted(
                    value for value in representation_data_hashes if value
                ),
            },
        )
    batch_run_id = f"{dataset['run_id']}:stage4r:{attempt_id}:{target_id}"
    costs_root = _resolve(root, paths["costs"])
    attempt_costs_root = costs_root / "stage4r" / attempt_id
    calls_root = attempt_costs_root / target_id
    recorder = ProviderCallRecorder(calls_root)
    provider_env = load_provider_environment(root)
    try:
        adapter = DeepSeekChatAdapter.from_env(
            writer_profile, recorder=recorder, environ=provider_env
        )
    except RuntimeError as exc:
        return _blocked(audit_path, "provider_environment_missing", str(exc))
    writer = Writer(settings, tokenizer, adapter)
    order = condition_order(target_id, str(dataset["writer_order_seed"]))
    order_manifest = {
        "target_id": target_id,
        "condition_order": list(order),
        "ordering_kind": "local_seeded_shuffle",
        "ordering_seed_id": dataset["writer_order_seed"],
        "model_generation_seed": None,
        "batch_run_id": batch_run_id,
        "run_mode": "formal",
        "config_hash": metadata.config_hash,
        "data_manifest_hash": metadata.data_manifest_hash,
        "provider_profile_hash": writer_profile.profile_hash,
    }
    generations = []
    costs: dict[str, Any] = {}
    generations_root = _resolve(root, paths["generations"]) / "stage4r" / attempt_id
    for condition in order:
        generation = writer.generate(
            pack,
            condition,
            representations[condition],
            formal_mode=True,
            run_id=batch_run_id,
            config_hash=metadata.config_hash,
            data_manifest_hash=metadata.data_manifest_hash,
            provider_profile_hash=writer_profile.profile_hash,
        )
        generations.append(generation)
        write_json(generations_root / f"{generation.generation_id}.json", generation.to_dict())
        costs[condition] = {
            "generation_id": generation.generation_id,
            "input_tokens": generation.input_tokens,
            "output_tokens": generation.output_tokens,
            "latency_ms": generation.latency_ms,
            "logged": True,
        }
    write_json(generations_root / f"order_{target_id}.json", order_manifest)
    writer_costs_path = attempt_costs_root / f"writer_costs_{target_id}.json"
    write_json(
        writer_costs_path,
        {
            "run_id": batch_run_id,
            "run_mode": "formal",
            "attempt_id": attempt_id,
            "config_hash": metadata.config_hash,
            "data_manifest_hash": metadata.data_manifest_hash,
            "provider_profile_hash": writer_profile.profile_hash,
            "generations": costs,
        },
    )
    report = build_gate4_formal_report(
        generations,
        costs,
        load_call_artifacts(calls_root),
        gate3_report=gate3,
        order_manifest=order_manifest,
        expected_profile_hash=writer_profile.profile_hash,
        batch_run_id=batch_run_id,
    )
    report.update(
        {
            "attempt_id": attempt_id,
            "call_artifact_root": calls_root.relative_to(root).as_posix(),
            "generation_artifact_root": generations_root.relative_to(root).as_posix(),
            "cost_artifact_path": writer_costs_path.relative_to(root).as_posix(),
        }
    )
    write_json(audit_path, report)
    return report, audit_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the one-target formal Stage 4R Writer")
    parser.add_argument("--dataset-config", default="configs/dataset_formal_pilot.yaml")
    parser.add_argument("--writer-config", default="configs/writer_formal.yaml")
    parser.add_argument("--budget-config", default="configs/budget_formal.yaml")
    parser.add_argument("--provider-config", default="configs/providers.yaml")
    parser.add_argument("--execute-live", action="store_true")
    args = parser.parse_args(argv)
    report, path = generate_writer_formal(
        args.dataset_config,
        args.writer_config,
        args.budget_config,
        args.provider_config,
        execute_live=args.execute_live,
    )
    print(f"GATE_4R={report['status']}")
    print(f"AUDIT={path}")
    return 0 if report["status"] == "PASS" else (2 if report["status"] == "BLOCKED" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
