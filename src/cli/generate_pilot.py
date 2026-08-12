from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.audits.gate4 import build_gate4_report
from src.artifacts import artifact_id
from src.budget.tokenizer import DeterministicRegexTokenizer
from src.common.jsonio import read_json, write_json
from src.config import load_config
from src.domain.models import RepresentationArtifact, TargetEvidencePack
from src.logging_config import configure_logging
from src.runtime import WriterDataAccess, load_writer_runtime_config
from src.writer.config import WRITER_CONDITIONS, load_writer_settings
from src.writer.writer import Writer

LOGGER = logging.getLogger(__name__)

CONDITION_TO_TYPE = {
    "evidence_only": None,
    "raw": "raw",
    "summary": "summary",
    "guideline": "guideline",
    "experience": "experience",
}


def _resolve(root: Path, configured_path: str) -> Path:
    path = Path(configured_path)
    return path if path.is_absolute() else root / path


def generate_pilot(
    dataset_config_path: str | Path,
    writer_config_path: str | Path,
    budget_config_path: str | Path,
    project_root: str | Path | None = None,
    target_ids: list[str] | None = None,
) -> tuple[dict, Path, list[dict]]:
    dataset_path = Path(dataset_config_path).resolve()
    writer_path = Path(writer_config_path).resolve()
    budget_path = Path(budget_config_path).resolve()
    root = Path(project_root).resolve() if project_root else dataset_path.parent.parent
    dataset = load_config(dataset_path)
    budget = load_config(budget_path)
    settings = load_writer_settings(writer_path)
    access = WriterDataAccess(load_writer_runtime_config(dataset_path, root))
    tokenizer_config = budget["tokenizer"]
    tokenizer = DeterministicRegexTokenizer(version=str(tokenizer_config["version"]))
    writer = Writer(settings, tokenizer, adapter=None)  # deterministic mechanics mode

    representations = {
        condition: (
            RepresentationArtifact.from_dict(
                read_json(access.representation_path(CONDITION_TO_TYPE[condition]))
            )
            if CONDITION_TO_TYPE[condition]
            else None
        )
        for condition in WRITER_CONDITIONS
    }

    available_targets = sorted(
        path.stem
        for path in access.configured_roots()["evidence_packs"].glob("*.json")
    )
    selected = sorted(target_ids) if target_ids else available_targets
    missing = [target_id for target_id in selected if target_id not in available_targets]
    if missing:
        raise ValueError(f"Unknown target ids: {', '.join(missing)}")

    generations: list[dict] = []
    generation_objects: list = []
    costs: dict[str, dict] = {}
    manifest: dict[str, dict] = {}
    for target_id in selected:
        pack = TargetEvidencePack.from_dict(read_json(access.evidence_pack_path(target_id)))
        manifest[target_id] = {}
        for condition in WRITER_CONDITIONS:
            generation = writer.generate(
                pack, condition, representations[condition]
            )
            artifact = generation.to_dict()
            write_json(access.generation_path(generation.generation_id), artifact)
            generations.append(artifact)
            generation_objects.append(generation)
            manifest[target_id][condition] = generation.generation_id
            costs[generation.generation_id] = {
                "target_id": target_id,
                "condition": condition,
                "writer_model": generation.writer_model,
                "writer_prompt_hash": generation.writer_prompt_hash,
                "input_tokens": generation.input_tokens,
                "output_tokens": generation.output_tokens,
                "latency_ms": generation.latency_ms,
                "logged": True,
            }

    costs_path = access.costs_dir() / "writer_costs.json"
    write_json(
        costs_path,
        {
            "writer_config_version": settings.config_version,
            "budget_config_version": budget["config_version"],
            "adapter_mode": "deterministic",
            "generations": costs,
        },
    )
    manifest_path = access.configured_roots()["generations"] / "manifest.json"
    write_json(manifest_path, manifest)

    report: dict | None = None
    audit_path: Path | None = None
    if len(selected) == 1:
        target_generation_objects = [
            generation for generation in generation_objects if generation.target_id == selected[0]
        ]
        condition_costs = {
            generation.condition: costs[generation.generation_id]
            for generation in target_generation_objects
        }
        report = build_gate4_report(target_generation_objects, condition_costs)
        report["audit_id"] = artifact_id("audit", report)
        report["writer_config_version"] = settings.config_version
        report["budget_config_version"] = budget["config_version"]
        report["fixture_kind"] = dataset["pilot"].get("fixture_kind")
        audit_path = _resolve(root, dataset["paths"]["audits"]) / "gate4.json"
        write_json(audit_path, report)

    LOGGER.info(
        "Generated %s generations for %s targets (gate4=%s)",
        len(generations),
        len(selected),
        report["passed"] if report else "not-single-target",
    )
    return report, audit_path or costs_path, generations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate pilot Introductions (Stage 4/5)")
    parser.add_argument("--dataset-config", default="configs/dataset.yaml")
    parser.add_argument("--writer-config", default="configs/writer.yaml")
    parser.add_argument("--budget-config", default="configs/budget.yaml")
    parser.add_argument("--target", action="append", default=None)
    args = parser.parse_args(argv)
    configure_logging()
    report, audit_path, _ = generate_pilot(
        args.dataset_config,
        args.writer_config,
        args.budget_config,
        target_ids=args.target,
    )
    if report is not None:
        print(f"GATE4={'PASS' if report['passed'] else 'FAIL'}")
        print(f"AUDIT={audit_path}")
        return 0 if report["passed"] else 1
    print(f"GENERATED={audit_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
