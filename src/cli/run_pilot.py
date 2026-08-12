from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.artifacts import artifact_id
from src.audits.gate5 import build_gate5_report
from src.budget.tokenizer import DeterministicRegexTokenizer
from src.common.jsonio import read_json, write_json
from src.compilers.config import load_compiler_settings
from src.config import load_config
from src.domain.models import RepresentationArtifact, SourcePaper, TargetEvidencePack
from src.evaluation.derive import derive_desired_introduction_length
from src.evaluation.review import run_pilot_review
from src.logging_config import configure_logging
from src.runtime import (
    CompilerDataAccess,
    EvaluatorDataAccess,
    load_compiler_runtime_config,
    load_evaluator_runtime_config,
)
from src.cli.compile_representations import compile_representations
from src.cli.generate_pilot import generate_pilot
from src.writer.config import load_writer_settings

LOGGER = logging.getLogger(__name__)
REPRESENTATION_NAMES = ("raw", "summary", "guideline", "experience")


def _resolve(root: Path, configured_path: str) -> Path:
    path = Path(configured_path)
    return path if path.is_absolute() else root / path


def _reset_derived_artifacts(compiler_access: CompilerDataAccess, generations_root: Path) -> None:
    """Clear deterministically-regenerable content-addressed artifacts before a fresh run."""
    for path in generations_root.glob("gen_*.json"):
        if path.is_file():
            path.unlink()
    for root_name in ("representations", "experiences"):
        root = compiler_access.configured_roots()[root_name]
        for path in sorted(root.glob("*.json")):
            if path.is_file():
                path.unlink()


def run_pilot(
    dataset_config_path: str | Path,
    compiler_config_path: str | Path,
    budget_config_path: str | Path,
    writer_config_path: str | Path,
    project_root: str | Path | None = None,
) -> tuple[dict, Path, Path]:
    dataset_path = Path(dataset_config_path).resolve()
    compiler_path = Path(compiler_config_path).resolve()
    budget_path = Path(budget_config_path).resolve()
    writer_path = Path(writer_config_path).resolve()
    root = Path(project_root).resolve() if project_root else dataset_path.parent.parent
    dataset = load_config(dataset_path)
    budget = load_config(budget_path)
    compiler_settings = load_compiler_settings(compiler_path)
    writer_settings = load_writer_settings(writer_path)

    compiler_access = CompilerDataAccess(load_compiler_runtime_config(dataset_path, root))
    evaluator_access = EvaluatorDataAccess(load_evaluator_runtime_config(dataset_path, root))
    generations_root = _resolve(root, dataset["paths"]["generations"])
    _reset_derived_artifacts(compiler_access, generations_root)
    _, _ = compile_representations(dataset_config_path, compiler_config_path, budget_config_path, root)
    _, _, generations = generate_pilot(
        dataset_config_path, writer_config_path, budget_config_path, root, target_ids=None
    )
    sources = [
        SourcePaper.from_dict(read_json(compiler_access.source_paper_path(record.stem)))
        for record in sorted(compiler_access.configured_roots()["source_normalized"].glob("*.json"))
    ]
    target_ids = sorted(
        path.stem for path in compiler_access.configured_roots()["evidence_packs"].glob("*.json")
    )
    packs = {
        target_id: TargetEvidencePack.from_dict(
            read_json(compiler_access.evidence_pack_path(target_id))
        )
        for target_id in target_ids
    }
    gold_by_target = {
        target_id: read_json(evaluator_access.target_gold_path(target_id))["introduction"]
        for target_id in target_ids
    }
    representations = {
        name: RepresentationArtifact.from_dict(
            read_json(compiler_access.representation_path(name))
        )
        for name in REPRESENTATION_NAMES
    }
    writing_condition_tokens = int(budget["writing_condition_tokens"])
    review = run_pilot_review(
        generations,
        gold_by_target,
        packs,
        representations,
        sources,
        writing_condition_tokens,
        writer_settings.max_output_tokens,
    )
    review_path = evaluator_access.evaluations_dir() / "pilot_review.json"
    write_json(review_path, review)

    config_versions = {
        "dataset": dataset["config_version"],
        "compiler": compiler_settings.config_version,
        "budget": budget["config_version"],
        "writer": writer_settings.config_version,
    }
    prompt_versions = {
        "writer_system": writer_settings.system_prompt_version,
        "writer_task": writer_settings.task_prompt_version,
        "experience_extract": compiler_settings.extraction_prompt_version,
        "experience_verify": compiler_settings.verifier_prompt_version,
        "experience_adjudicate": compiler_settings.adjudication_prompt_version,
        "summary": compiler_settings.summary_prompt_version,
        "guideline": compiler_settings.guideline_prompt_version,
    }
    report = build_gate5_report(
        review,
        config_versions,
        prompt_versions,
        expected_generations=len(target_ids) * 5,
        desired_length_derivation=derive_desired_introduction_length(
            list(gold_by_target.values()),
            DeterministicRegexTokenizer(
                version=str(budget["tokenizer"]["version"])
            ),
        ),
    )
    report["audit_id"] = artifact_id("audit", report)
    audit_path = _resolve(root, dataset["paths"]["audits"]) / "gate5.json"
    write_json(audit_path, report)
    LOGGER.info(
        "Pilot review passed=%s generations=%s freeze=%s",
        review["passed"],
        review["diagnostics"]["generation_count"],
        report["freeze_status"],
    )
    return report, audit_path, review_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the 10-target pilot and Gate 5 freeze (Stage 5)")
    parser.add_argument("--dataset-config", default="configs/dataset.yaml")
    parser.add_argument("--compiler-config", default="configs/compiler.yaml")
    parser.add_argument("--budget-config", default="configs/budget.yaml")
    parser.add_argument("--writer-config", default="configs/writer.yaml")
    args = parser.parse_args(argv)
    configure_logging()
    report, audit_path, review_path = run_pilot(
        args.dataset_config,
        args.compiler_config,
        args.budget_config,
        args.writer_config,
    )
    print(f"GATE5={'PASS' if report['passed'] else 'FAIL'}")
    print(f"AUDIT={audit_path}")
    print(f"REVIEW={review_path}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
