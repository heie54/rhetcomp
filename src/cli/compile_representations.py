from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.artifacts import artifact_id
from src.audits.gate3 import build_cost_record, build_gate3_report
from src.budget.tokenizer import DeterministicRegexTokenizer
from src.common.jsonio import read_json, write_json
from src.compilers.config import load_compiler_settings
from src.compilers.experience.representation import compile_experience_representation
from src.compilers.guideline import compile_guideline
from src.compilers.raw import compile_raw
from src.compilers.summary import compile_summary
from src.config import load_config
from src.domain.models import SourcePaper
from src.logging_config import configure_logging
from src.runtime import CompilerDataAccess, load_compiler_runtime_config


LOGGER = logging.getLogger(__name__)


def _resolve(root: Path, configured_path: str) -> Path:
    path = Path(configured_path)
    return path if path.is_absolute() else root / path


def compile_representations(
    dataset_config_path: str | Path,
    compiler_config_path: str | Path,
    budget_config_path: str | Path,
    project_root: str | Path | None = None,
) -> tuple[dict, Path]:
    dataset_path = Path(dataset_config_path).resolve()
    compiler_path = Path(compiler_config_path).resolve()
    budget_path = Path(budget_config_path).resolve()
    root = Path(project_root).resolve() if project_root else dataset_path.parent.parent
    dataset = load_config(dataset_path)
    budget = load_config(budget_path)
    settings = load_compiler_settings(compiler_path)
    access = CompilerDataAccess(load_compiler_runtime_config(dataset_path, root))

    sources = [
        SourcePaper.from_dict(read_json(access.source_paper_path(record.stem)))
        for record in sorted(access.configured_roots()["source_normalized"].glob("*.json"))
    ]
    tokenizer_config = budget["tokenizer"]
    tokenizer = DeterministicRegexTokenizer(version=str(tokenizer_config["version"]))
    writing_condition_tokens = int(budget["writing_condition_tokens"])
    adapter = None  # deterministic mechanics mode (no model configured/authorized)

    raw = compile_raw(sources, settings, tokenizer, writing_condition_tokens)
    summary = compile_summary(sources, settings, tokenizer, writing_condition_tokens, adapter)
    guideline = compile_guideline(sources, settings, tokenizer, writing_condition_tokens, adapter)
    experience, experience_result = compile_experience_representation(
        sources, settings, tokenizer, writing_condition_tokens, adapter
    )
    representations = {
        "raw": raw,
        "summary": summary,
        "guideline": guideline,
        "experience": experience,
    }
    for name, artifact in representations.items():
        write_json(
            access.configured_roots()["representations"] / f"{name}.json",
            artifact.to_dict(),
        )

    experiences_payload = {
        "source_corpus_hash": experience_result.source_corpus_hash,
        "adapter_mode": experience_result.adapter_mode,
        "experiences": [item.to_dict() for item in experience_result.experiences],
        "derived_meta": [item.to_dict() for item in experience_result.derived_meta],
    }
    write_json(access.experiences_dir() / "experience_library.json", experiences_payload)
    write_json(access.experiences_dir() / "trace.json", list(experience_result.trace))

    costs = {
        name: build_cost_record(artifact, mode="deterministic")
        for name, artifact in representations.items()
    }
    costs_path = access.costs_dir() / "compiler_costs.json"
    write_json(
        costs_path,
        {
            "source_corpus_hash": raw.source_corpus_hash,
            "writing_condition_tokens": writing_condition_tokens,
            "compiler_config_version": settings.config_version,
            "representations": costs,
        },
    )

    report = build_gate3_report(representations, costs, writing_condition_tokens)
    report["audit_id"] = artifact_id("audit", report)
    report["compiler_config_version"] = settings.config_version
    report["budget_config_version"] = budget["config_version"]
    report["fixture_kind"] = dataset["pilot"].get("fixture_kind")
    audit_path = _resolve(root, dataset["paths"]["audits"]) / "gate3.json"
    write_json(audit_path, report)
    LOGGER.info(
        "Compiled representations raw=%s summary=%s guideline=%s experience=%s",
        raw.content_tokens,
        summary.content_tokens,
        guideline.content_tokens,
        experience.content_tokens,
    )
    return report, audit_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compile all baseline representations (Stage 3)")
    parser.add_argument("--dataset-config", default="configs/dataset.yaml")
    parser.add_argument("--compiler-config", default="configs/compiler.yaml")
    parser.add_argument("--budget-config", default="configs/budget.yaml")
    args = parser.parse_args(argv)
    configure_logging()
    report, audit_path = compile_representations(
        args.dataset_config, args.compiler_config, args.budget_config
    )
    print(f"GATE3={'PASS' if report['passed'] else 'FAIL'}")
    print(f"AUDIT={audit_path}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
