from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.adapters.chat.deepseek import DeepSeekChatAdapter
from src.adapters.config import load_provider_profiles
from src.adapters.environment import load_provider_environment
from src.adapters.records import ProviderCallRecorder
from src.artifacts import artifact_id
from src.artifacts_freeze import build_formal_freeze_manifest
from src.audits.gate2_formal import load_call_artifacts
from src.audits.gate5_formal import blocked_gate5_formal_report, build_gate5_formal_report
from src.budget.formal_tokenizer import load_formal_tokenizer
from src.common.jsonio import read_json, write_json
from src.compilers.config import load_compiler_settings
from src.config import load_config
from src.domain.models import RepresentationArtifact, SourcePaper, TargetEvidencePack
from src.evaluation.review import run_pilot_review
from src.runtime import EvaluatorDataAccess, load_evaluator_runtime_config
from src.writer.config import WRITER_CONDITIONS, load_writer_settings
from src.writer.order import condition_order
from src.writer.writer import GenerationArtifact, Writer
from src.formal_metadata import build_formal_metadata


ROOT = Path(__file__).resolve().parents[2]


def _json_files(path: Path) -> list[Path]:
    return sorted(item for item in path.rglob("*.json") if item.is_file())


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


def _accepted_upstream_artifacts(
    root: Path, upstream: dict[str, dict[str, Any]]
) -> list[Path]:
    resolved_root = root.resolve()
    artifacts: list[Path] = []
    for stage in ("2R", "3R", "4R"):
        report = upstream[stage]
        for key in (
            "call_artifact_root",
            "experience_artifact_root",
            "representation_artifact_root",
            "generation_artifact_root",
        ):
            configured = report.get(key)
            if not configured:
                continue
            candidate = _upstream_artifact_root(root, report, key, f"Upstream {stage}")
            artifacts.extend(_json_files(candidate))
        configured_cost = report.get("cost_artifact_path")
        if configured_cost:
            cost_path = (resolved_root / str(configured_cost)).resolve()
            try:
                cost_path.relative_to(resolved_root)
            except ValueError as exc:
                raise ValueError(
                    f"Upstream {stage} cost path escapes project root: {configured_cost}"
                ) from exc
            artifacts.append(cost_path)
    return sorted(set(artifacts), key=lambda path: path.as_posix())
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


def _blocked(path: Path, reason: str, detail: Any) -> tuple[dict[str, Any], Path, Path | None]:
    report = blocked_gate5_formal_report(reason, detail)
    write_json(path, report)
    return report, path, None


def run_formal_pilot(
    dataset_config_path: str | Path,
    compiler_config_path: str | Path,
    budget_config_path: str | Path,
    writer_config_path: str | Path,
    provider_config_path: str | Path,
    *,
    project_root: str | Path | None = None,
    execute_live: bool,
) -> tuple[dict[str, Any], Path, Path | None]:
    dataset_path = Path(dataset_config_path).resolve()
    compiler_path = Path(compiler_config_path).resolve()
    budget_path = Path(budget_config_path).resolve()
    writer_path = Path(writer_config_path).resolve()
    provider_path = Path(provider_config_path).resolve()
    root = Path(project_root).resolve() if project_root else ROOT
    dataset = load_config(dataset_path)
    budget = load_config(budget_path)
    paths = dataset["paths"]
    audits_root = _resolve(root, paths["audits"])
    audit_path = audits_root / "gate5_formal.json"
    upstream_paths = {
        "1R": audits_root / "gate1_formal.json",
        "2R": audits_root / "gate2_formal.json",
        "3R": audits_root / "gate3_formal.json",
        "4R": audits_root / "gate4_formal.json",
    }
    missing_upstream = {name: str(path) for name, path in upstream_paths.items() if not path.exists()}
    if missing_upstream:
        return _blocked(audit_path, "upstream_formal_audits_missing", missing_upstream)
    upstream = {name: read_json(path) for name, path in upstream_paths.items()}
    if any(report.get("status") != "PASS" for report in upstream.values()):
        return _blocked(
            audit_path,
            "upstream_formal_gate_not_passed",
            {name: report.get("status") for name, report in upstream.items()},
        )
    if not execute_live:
        return _blocked(
            audit_path,
            "live_execution_not_authorized",
            "rerun with --execute-live after reviewing the 50 paid Writer calls",
        )

    acl_manifest = read_json(_resolve(root, paths["manifests"]) / "acl_pilot.json")
    target_manifest = read_json(_resolve(root, paths["manifests"]) / "nc_physics_pilot.json")
    if target_manifest.get("official_test_accessed") is not False:
        return _blocked(audit_path, "official_test_isolation_not_proven", target_manifest)
    target_ids = sorted(entry["target_id"] for entry in target_manifest["entries"])
    if len(target_ids) != 10:
        return _blocked(audit_path, "formal_target_count_invalid", len(target_ids))
    settings = load_writer_settings(writer_path)
    compiler_settings = load_compiler_settings(compiler_path)
    tokenizer = load_formal_tokenizer(budget_path, root)
    providers = load_provider_profiles(provider_path)
    writer_profile = providers.require("deepseek_writer_v1")
    metadata = build_formal_metadata(
        str(dataset["run_id"]),
        configs={
            "dataset": dataset,
            "compiler": load_config(compiler_path),
            "budget": budget,
            "writer": load_config(writer_path),
            "providers": load_config(provider_path),
        },
        manifests={
            "acl": acl_manifest["manifest_hash"],
            "targets": target_manifest["manifest_hash"],
        },
    )
    try:
        representations_root = _upstream_artifact_root(
            root, upstream["3R"], "representation_artifact_root", "Gate 3R"
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
    review_representations = {
        name: RepresentationArtifact.from_dict(read_json(representations_root / f"{name}.json"))
        for name in ("raw", "summary", "guideline", "experience")
    }
    upstream_data_hashes = {
        report.get("data_manifest_hash") for report in upstream.values()
    }
    representation_config_hashes = {
        representation.config_hash for representation in review_representations.values()
    }
    representation_data_hashes = {
        representation.data_manifest_hash for representation in review_representations.values()
    }
    if (
        upstream_data_hashes != {metadata.data_manifest_hash}
        or representation_config_hashes != {upstream["3R"].get("config_hash")}
        or representation_data_hashes != {metadata.data_manifest_hash}
    ):
        return _blocked(
            audit_path,
            "upstream_formal_metadata_mismatch",
            {
                "upstream_data_manifest_hashes": sorted(
                    value for value in upstream_data_hashes if value
                ),
                "current_data_manifest_hash": metadata.data_manifest_hash,
                "gate3r_config_hash": upstream["3R"].get("config_hash"),
                "representation_config_hashes": sorted(
                    value for value in representation_config_hashes if value
                ),
                "representation_data_manifest_hashes": sorted(
                    value for value in representation_data_hashes if value
                ),
            },
        )
    attempt_id = f"attempt_{uuid4().hex[:12]}"
    generation_attempt_root = _resolve(root, paths["generations"]) / "stage5r" / attempt_id
    costs_root = _resolve(root, paths["costs"])
    attempt_costs_root = costs_root / "stage5r" / attempt_id
    packs: dict[str, TargetEvidencePack] = {}
    generations: list[GenerationArtifact] = []
    costs: dict[str, Any] = {}
    order_manifests: dict[str, dict[str, Any]] = {}
    calls_by_target: dict[str, list] = {}
    provider_env = load_provider_environment(root)
    for target_id in target_ids:
        pack = TargetEvidencePack.from_dict(
            read_json(_resolve(root, paths["evidence_packs"]) / f"{target_id}.json")
        )
        packs[target_id] = pack
        order = condition_order(target_id, str(dataset["writer_order_seed"]))
        batch_run_id = f"{dataset['run_id']}:stage5r:{attempt_id}:{target_id}"
        calls_root = attempt_costs_root / target_id
        recorder = ProviderCallRecorder(calls_root)
        try:
            adapter = DeepSeekChatAdapter.from_env(
                writer_profile, recorder=recorder, environ=provider_env
            )
        except RuntimeError as exc:
            return _blocked(audit_path, "provider_environment_missing", str(exc))
        writer = Writer(settings, tokenizer, adapter)
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
        order_manifests[target_id] = order_manifest
        write_json(generation_attempt_root / f"order_{target_id}.json", order_manifest)
        costs[target_id] = {}
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
            write_json(
                generation_attempt_root / f"{generation.generation_id}.json",
                generation.to_dict(),
            )
            costs[target_id][condition] = {
                "generation_id": generation.generation_id,
                "input_tokens": generation.input_tokens,
                "output_tokens": generation.output_tokens,
                "latency_ms": generation.latency_ms,
                "logged": True,
            }
        calls_by_target[target_id] = load_call_artifacts(calls_root)
    generation_manifest = {
        "run_mode": "formal",
        "run_id": dataset["run_id"],
        "attempt_id": attempt_id,
        "target_ids": target_ids,
        "generation_ids": [generation.generation_id for generation in generations],
        "generation_count": len(generations),
        "official_test_accessed": False,
        "config_hash": metadata.config_hash,
        "data_manifest_hash": metadata.data_manifest_hash,
        "provider_profile_hash": writer_profile.profile_hash,
    }
    write_json(generation_attempt_root / "manifest.json", generation_manifest)
    write_json(
        attempt_costs_root / "writer_costs.json",
        {
            "run_id": dataset["run_id"],
            "run_mode": "formal",
            "attempt_id": attempt_id,
            "config_hash": metadata.config_hash,
            "data_manifest_hash": metadata.data_manifest_hash,
            "provider_profile_hash": writer_profile.profile_hash,
            "targets": costs,
        },
    )

    # Evaluator-only phase begins after every Writer artifact is materialized.
    evaluator = EvaluatorDataAccess(load_evaluator_runtime_config(dataset_path, root))
    gold_by_target = {
        target_id: read_json(evaluator.target_gold_path(target_id))["introduction"]
        for target_id in target_ids
    }
    sources_root = _resolve(root, paths["source_normalized"])
    sources = [
        SourcePaper.from_dict(read_json(sources_root / f"{entry['source_id']}.json"))
        for entry in acl_manifest["entries"]
    ]
    generation_payloads = [generation.to_dict() for generation in generations]
    review = run_pilot_review(
        generation_payloads,
        gold_by_target,
        packs,
        review_representations,
        sources,
        int(budget["writing_condition_tokens"]),
        settings.max_output_tokens,
    )
    review.update(
        {
            "run_mode": "formal",
            "attempt_id": attempt_id,
            "official_test_accessed": False,
            "condition_ranking_computed": False,
            "final_statistics_computed": False,
            "config_hash": metadata.config_hash,
            "data_manifest_hash": metadata.data_manifest_hash,
            "provider_profile_hash": writer_profile.profile_hash,
        }
    )
    evaluation_id = artifact_id("formal_pilot_review", review)
    review["evaluation_id"] = evaluation_id
    review_path = _resolve(root, paths["evaluations"]) / f"pilot_review_formal_{attempt_id}.json"
    write_json(review_path, review)

    tokenizer_manifest = read_json(
        _resolve(root, paths["manifests"]) / "deepseek_formal_tokenizer.json"
    )
    frozen_files = [
        dataset_path,
        compiler_path,
        budget_path,
        writer_path,
        provider_path,
        *sorted((root / "src").rglob("*.py"), key=lambda path: path.as_posix()),
    ]
    manifests_root = _resolve(root, paths["manifests"])
    audits_root = _resolve(root, paths["audits"])
    evidence_packs_root = _resolve(root, paths["evidence_packs"])
    compiled_artifacts = _accepted_upstream_artifacts(root, upstream)
    artifact_files = {
        "data_manifests": [
            manifests_root / "acl_pilot.json",
            manifests_root / "nc_physics_pilot.json",
            manifests_root / "deepseek_formal_tokenizer.json",
        ],
        "upstream_audits": [audits_root / f"gate{stage}_formal.json" for stage in range(1, 5)],
        "compiled_artifacts": compiled_artifacts,
        "evidence_packs": [evidence_packs_root / f"{target_id}.json" for target_id in target_ids],
        "generation_artifacts": [
            generation_attempt_root / f"{generation.generation_id}.json"
            for generation in generations
        ],
        "order_manifests": [
            generation_attempt_root / f"order_{target_id}.json" for target_id in target_ids
        ],
        "writer_call_artifacts": [],
        "writer_cost_artifacts": [attempt_costs_root / "writer_costs.json"],
        "run_manifests": [generation_attempt_root / "manifest.json"],
        "evaluation_artifacts": [review_path],
    }
    artifact_files["writer_call_artifacts"] = [
        path
        for target_id in target_ids
        for path in _json_files(attempt_costs_root / target_id / "calls")
    ]
    freeze_manifest = build_formal_freeze_manifest(
        root=root,
        run_id=str(dataset["run_id"]),
        attempt_id=attempt_id,
        providers=providers,
        acl_manifest=acl_manifest,
        target_manifest=target_manifest,
        tokenizer_manifest=tokenizer_manifest,
        frozen_files=frozen_files,
        artifact_files=artifact_files,
        generation_ids=[generation.generation_id for generation in generations],
        evaluation_id=evaluation_id,
    )
    report = build_gate5_formal_report(
        review,
        generations,
        calls_by_target,
        order_manifests,
        upstream_reports=upstream,
        target_manifest=target_manifest,
        freeze_manifest=freeze_manifest,
    )
    report.update(
        {
            "attempt_id": attempt_id,
            "config_versions": {
                "dataset": dataset["config_version"],
                "compiler": compiler_settings.config_version,
                "budget": budget["config_version"],
                "writer": settings.config_version,
                "providers": providers.config_version,
            },
            "evaluation_id": evaluation_id,
        }
    )
    write_json(audit_path, report)
    freeze_path: Path | None = None
    if report["status"] == "PASS":
        freeze_path = audits_root / "formal_pilot_freeze.json"
        write_json(freeze_path, freeze_manifest)
    return report, audit_path, freeze_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the 10-target/50-generation formal pilot")
    parser.add_argument("--dataset-config", default="configs/dataset_formal_pilot.yaml")
    parser.add_argument("--compiler-config", default="configs/compiler_formal.yaml")
    parser.add_argument("--budget-config", default="configs/budget_formal.yaml")
    parser.add_argument("--writer-config", default="configs/writer_formal.yaml")
    parser.add_argument("--provider-config", default="configs/providers.yaml")
    parser.add_argument("--execute-live", action="store_true")
    args = parser.parse_args(argv)
    report, audit_path, freeze_path = run_formal_pilot(
        args.dataset_config,
        args.compiler_config,
        args.budget_config,
        args.writer_config,
        args.provider_config,
        execute_live=args.execute_live,
    )
    print(f"GATE_5R={report['status']}")
    print(f"AUDIT={audit_path}")
    print(f"FORMAL_FREEZE={freeze_path or 'NOT_WRITTEN'}")
    return 0 if report["status"] == "PASS" else (2 if report["status"] == "BLOCKED" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
