from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.audits.gate2 import build_gate2_report
from src.artifacts import artifact_id
from src.budget.tokenizer import DeterministicRegexTokenizer
from src.common.jsonio import read_json, write_json
from src.compilers.config import load_compiler_settings
from src.compilers.experience.pipeline import compile_experience_library
from src.config import load_config
from src.domain.models import SourcePaper
from src.logging_config import configure_logging
from src.runtime import CompilerDataAccess, load_compiler_runtime_config


LOGGER = logging.getLogger(__name__)


def _resolve(root: Path, configured_path: str) -> Path:
    path = Path(configured_path)
    return path if path.is_absolute() else root / path


def compile_experience(
    dataset_config_path: str | Path,
    compiler_config_path: str | Path,
    budget_config_path: str | Path,
    project_root: str | Path | None = None,
) -> tuple[dict, Path, Path]:
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
    result = compile_experience_library(
        sources,
        settings,
        tokenizer,
        writing_condition_tokens,
        adapter=None,  # deterministic mechanics mode (no model configured/authorized)
    )

    experiences_payload = {
        "source_corpus_hash": result.source_corpus_hash,
        "adapter_mode": result.adapter_mode,
        "experiences": [item.to_dict() for item in result.experiences],
        "derived_meta": [item.to_dict() for item in result.derived_meta],
    }
    experiences_path = access.experiences_dir() / "experience_library.json"
    trace_path = access.experiences_dir() / "trace.json"
    write_json(experiences_path, experiences_payload)
    write_json(trace_path, list(result.trace))

    report = build_gate2_report(result, settings, writing_condition_tokens)
    report["audit_id"] = artifact_id("audit", report)
    report["compiler_config_version"] = settings.config_version
    report["budget_config_version"] = budget["config_version"]
    report["fixture_kind"] = dataset["pilot"].get("fixture_kind")
    audit_path = _resolve(root, dataset["paths"]["audits"]) / "gate2.json"
    write_json(audit_path, report)
    LOGGER.info(
        "Compiled Experience library: candidates=%s canonical=%s library_tokens=%s",
        result.candidate_count,
        result.canonical_count,
        result.library.content_tokens,
    )
    return report, audit_path, experiences_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compile the Experience library (Stage 2)")
    parser.add_argument("--dataset-config", default="configs/dataset.yaml")
    parser.add_argument("--compiler-config", default="configs/compiler.yaml")
    parser.add_argument("--budget-config", default="configs/budget.yaml")
    args = parser.parse_args(argv)
    configure_logging()
    report, audit_path, _ = compile_experience(
        args.dataset_config, args.compiler_config, args.budget_config
    )
    print(f"GATE2={'PASS' if report['passed'] else 'FAIL'}")
    print(f"AUDIT={audit_path}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
