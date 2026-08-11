from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from src.artifacts import artifact_id
from src.audits.gate1 import build_gate1_report
from src.budget.tokenizer import BudgetController, DeterministicRegexTokenizer
from src.common.jsonio import read_jsonl, write_json
from src.config import load_config
from src.evidence_pack.builder import build_target_evidence_pack
from src.ingest.source import normalize_source_record
from src.ingest.target import normalize_target_record
from src.logging_config import configure_logging


LOGGER = logging.getLogger(__name__)


def _resolve(root: Path, configured_path: str) -> Path:
    path = Path(configured_path)
    return path if path.is_absolute() else root / path


def _unique_ids(records: list[dict[str, Any]], field: str, label: str) -> None:
    values = [record.get(field) for record in records]
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"Every {label} record must have a non-empty {field}")
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate {field} in {label} input")


def prepare_pilot(
    dataset_config_path: str | Path,
    budget_config_path: str | Path,
    project_root: str | Path | None = None,
) -> tuple[dict[str, Any], Path]:
    dataset_path = Path(dataset_config_path).resolve()
    budget_path = Path(budget_config_path).resolve()
    root = Path(project_root).resolve() if project_root else dataset_path.parent.parent
    dataset = load_config(dataset_path)
    budget = load_config(budget_path)
    paths = dataset["paths"]

    source_input = _resolve(root, paths["source_input"])
    target_input = _resolve(root, paths["target_input"])
    source_output = _resolve(root, paths["source_normalized"])
    visible_output = _resolve(root, paths["target_visible"])
    evidence_output = _resolve(root, paths["target_evidence"])
    gold_output = _resolve(root, paths["target_gold"])
    packs_output = _resolve(root, paths["evidence_packs"])
    audits_output = _resolve(root, paths["audits"])

    source_records = read_jsonl(source_input)
    target_records = read_jsonl(target_input)
    _unique_ids(source_records, "source_id", "source")
    _unique_ids(target_records, "target_id", "target")

    sources = [normalize_source_record(record) for record in source_records]
    for source in sources:
        write_json(source_output / f"{source.source_id}.json", source.to_dict())

    tokenizer_config = budget["tokenizer"]
    tokenizer = DeterministicRegexTokenizer(version=str(tokenizer_config["version"]))
    controller = BudgetController(tokenizer)
    visible_payloads: list[dict[str, Any]] = []
    evidence_payloads: list[dict[str, Any]] = []
    gold_payloads: list[dict[str, Any]] = []
    packs = []
    for record in target_records:
        visible, evidence, gold = normalize_target_record(record)
        visible_payload = visible.to_dict()
        evidence_payload = evidence.to_dict()
        gold_payload = gold.to_dict()
        write_json(visible_output / f"{visible.target_id}.json", visible_payload)
        write_json(evidence_output / f"{evidence.target_id}.json", evidence_payload)
        write_json(gold_output / f"{gold.target_id}.json", gold_payload)
        pack = build_target_evidence_pack(
            visible,
            evidence,
            budget_tokens=int(budget["target_evidence_tokens"]),
            budget_controller=controller,
        )
        write_json(packs_output / f"{pack.target_id}.json", pack.to_dict())
        visible_payloads.append(visible_payload)
        evidence_payloads.append(evidence_payload)
        gold_payloads.append(gold_payload)
        packs.append(pack)

    pilot = dataset["pilot"]
    report = build_gate1_report(
        project_root=root,
        sources=sources,
        visible_payloads=visible_payloads,
        evidence_payloads=evidence_payloads,
        gold_payloads=gold_payloads,
        packs=packs,
        expected_source_count=int(pilot["expected_source_count"]),
        expected_target_count=int(pilot["expected_target_count"]),
        roots=(visible_output, evidence_output, gold_output),
    )
    report["fixture_kind"] = pilot.get("fixture_kind")
    report["dataset_config_version"] = dataset["config_version"]
    report["budget_config_version"] = budget["config_version"]
    report["audit_id"] = artifact_id("audit", report)
    audit_path = audits_output / "gate1.json"
    write_json(audit_path, report)
    LOGGER.info("Prepared pilot artifacts; Gate 1 passed=%s", report["passed"])
    return report, audit_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare Baseline v0.1 Stage 1 pilot artifacts")
    parser.add_argument("--dataset-config", default="configs/dataset.yaml")
    parser.add_argument("--budget-config", default="configs/budget.yaml")
    args = parser.parse_args(argv)
    configure_logging()
    report, audit_path = prepare_pilot(args.dataset_config, args.budget_config)
    print(f"GATE1={'PASS' if report['passed'] else 'FAIL'}")
    print(f"AUDIT={audit_path}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
