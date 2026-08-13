from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.audits.gate1 import build_gate1_report
from src.common.jsonio import (
    read_json,
    read_jsonl,
    sha256_file,
    sha256_json,
    sha256_text,
    write_json,
)
from src.compilers.corpus import source_corpus_hash
from src.config import load_config
from src.domain.models import SourcePaper, TargetEvidencePack
from src.runtime import (
    CompilerDataAccess,
    EvaluatorDataAccess,
    WriterDataAccess,
    load_compiler_runtime_config,
    load_evaluator_runtime_config,
    load_writer_runtime_config,
)


def _resolve(root: Path, configured_path: str) -> Path:
    path = Path(configured_path)
    return path if path.is_absolute() else root / path


def _manifest_hash_valid(manifest: dict[str, Any]) -> bool:
    expected = manifest.get("manifest_hash")
    payload = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    return isinstance(expected, str) and expected == sha256_json(payload)


def _pinned_source_slice_matches(
    source_path: Path,
    snapshot_path: Path,
    *,
    start: int,
    count: int,
    expected_total: int,
) -> bool:
    try:
        selected: list[dict[str, Any]] = []
        total = 0
        stop = start + count
        with source_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    return False
                if start <= total < stop:
                    selected.append(value)
                total += 1
        return total == expected_total and selected == read_jsonl(snapshot_path)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        return False


def build_gate1_formal_report(
    root: Path,
    dataset_config_path: Path,
    budget_config_path: Path,
) -> dict[str, Any]:
    dataset = load_config(dataset_config_path)
    budget = load_config(budget_config_path)
    paths = dataset["paths"]
    manifests_root = _resolve(root, paths["manifests"])
    acl_manifest_path = manifests_root / "acl_pilot.json"
    target_manifest_path = manifests_root / "nc_physics_pilot.json"
    missing = [str(path) for path in (acl_manifest_path, target_manifest_path) if not path.exists()]
    if missing:
        return {
            "gate": "1R",
            "run_mode": "formal",
            "status": "BLOCKED",
            "blockers": [{"reason": "missing_manifest", "path": path} for path in missing],
        }
    acl_manifest = read_json(acl_manifest_path)
    target_manifest = read_json(target_manifest_path)
    source_root = _resolve(root, paths["source_normalized"])
    visible_root = _resolve(root, paths["target_visible"])
    evidence_root = _resolve(root, paths["target_evidence"])
    gold_root = _resolve(root, paths["target_gold"])
    packs_root = _resolve(root, paths["evidence_packs"])
    source_ids = [entry["source_id"] for entry in acl_manifest["entries"]]
    target_ids = [entry["target_id"] for entry in target_manifest["entries"]]
    length_stats_path = _resolve(root, paths["audits"]) / "length_stats.json"
    acl_metadata_path = _resolve(root, paths["acl_metadata_raw"])
    source_trainval_path = _resolve(root, paths["nc_trainval_raw"])
    snapshot_path = _resolve(root, paths["nc_validation_snapshot"])
    snapshot_metadata_path = _resolve(root, paths["nc_validation_snapshot_metadata"])
    required_artifacts = [
        acl_metadata_path,
        source_trainval_path,
        snapshot_path,
        snapshot_metadata_path,
        length_stats_path,
        *(source_root / f"{source_id}.json" for source_id in source_ids),
        *(visible_root / f"{target_id}.json" for target_id in target_ids),
        *(evidence_root / f"{target_id}.json" for target_id in target_ids),
        *(gold_root / f"{target_id}.json" for target_id in target_ids),
        *(packs_root / f"{target_id}.json" for target_id in target_ids),
    ]
    missing_artifacts = [str(path) for path in required_artifacts if not path.exists()]
    if missing_artifacts:
        return {
            "gate": "1R",
            "run_mode": "formal",
            "status": "FAIL",
            "checks": {
                "derived_artifacts_present": {
                    "passed": False,
                    "missing": missing_artifacts,
                }
            },
        }
    sources = [SourcePaper.from_dict(read_json(source_root / f"{source_id}.json")) for source_id in source_ids]
    visible = [read_json(visible_root / f"{target_id}.json") for target_id in target_ids]
    evidence = [read_json(evidence_root / f"{target_id}.json") for target_id in target_ids]
    gold = [read_json(gold_root / f"{target_id}.json") for target_id in target_ids]
    packs = [TargetEvidencePack.from_dict(read_json(packs_root / f"{target_id}.json")) for target_id in target_ids]
    runtime_accesses = (
        CompilerDataAccess(load_compiler_runtime_config(dataset_config_path, root)),
        WriterDataAccess(load_writer_runtime_config(dataset_config_path, root)),
        EvaluatorDataAccess(load_evaluator_runtime_config(dataset_config_path, root)),
    )
    shared = build_gate1_report(
        project_root=root,
        sources=sources,
        visible_payloads=visible,
        evidence_payloads=evidence,
        gold_payloads=gold,
        packs=packs,
        expected_source_count=int(dataset["source"]["expected_count"]),
        expected_target_count=int(dataset["target"]["expected_count"]),
        roots=(visible_root, evidence_root, gold_root),
        runtime_accesses=runtime_accesses,
    )
    length_stats = read_json(length_stats_path)
    snapshot_metadata = read_json(snapshot_metadata_path)
    snapshot_metadata_payload = {
        key: value for key, value in snapshot_metadata.items() if key != "metadata_hash"
    }
    acl_entries = {entry["source_id"]: entry for entry in acl_manifest["entries"]}
    target_entries = {entry["target_id"]: entry for entry in target_manifest["entries"]}
    sources_by_id = {source.source_id: source for source in sources}
    visible_by_id = {item["target_id"]: item for item in visible}
    evidence_by_id = {item["target_id"]: item for item in evidence}
    gold_by_id = {item["target_id"]: item for item in gold}
    packs_by_id = {pack.target_id: pack.to_dict() for pack in packs}
    expected_revision = str(dataset["target"]["dataset_revision"])
    source_trainval_sha256 = sha256_file(source_trainval_path)
    snapshot_sha256 = sha256_file(snapshot_path)
    pinned_source_slice_matches = _pinned_source_slice_matches(
        source_trainval_path,
        snapshot_path,
        start=int(dataset["target"]["validation_start"]),
        count=int(dataset["target"]["validation_count"]),
        expected_total=int(dataset["target"]["trainval_expected_count"]),
    )
    checks = {
        "formal_namespaces": {
            "passed": dataset.get("run_mode") == "formal"
            and budget.get("run_mode") == "formal"
            and all("formal" in str(path).lower() for path in (source_root, visible_root, packs_root)),
        },
        "real_acl_corpus": {
            "passed": acl_manifest.get("dataset_config_version") == dataset["config_version"]
            and acl_manifest.get("provider") == "ACL Anthology"
            and acl_manifest.get("license_identifier")
            == dataset["source"]["license_identifier"]
            and acl_manifest.get("license_url") == dataset["source"]["license_url"]
            and acl_manifest.get("source_license_confirmation") is True
            and acl_manifest.get("expected_count") == int(dataset["source"]["expected_count"])
            and acl_manifest.get("ready_count") == int(dataset["source"]["expected_count"])
            and all(entry.get("status") == "ready" for entry in acl_manifest["entries"])
            and all(str(source_id).startswith("2024.acl-") for source_id in source_ids)
            and acl_manifest.get("source_corpus_hash") == source_corpus_hash(sources),
            "source_count": len(sources),
            "source_corpus_hash": source_corpus_hash(sources),
        },
        "real_nc_development_targets": {
            "passed": target_manifest.get("dataset_config_version") == dataset["config_version"]
            and target_manifest.get("dataset") == "Xiao-Youth/NC_Physics"
            and target_manifest.get("dataset_license_identifier")
            == dataset["target"]["dataset_license_identifier"]
            and target_manifest.get("license_review_url")
            == dataset["target"]["license_review_url"]
            and target_manifest.get("source_article_license_confirmation") is True
            and target_manifest.get("dataset_revision") == expected_revision
            and target_manifest.get("resolved_dataset_revision") == expected_revision
            and target_manifest.get("source_split") == "train"
            and target_manifest.get("expected_count") == int(dataset["target"]["expected_count"])
            and target_manifest.get("selected_count") == int(dataset["target"]["expected_count"])
            and target_manifest.get("official_test_accessed") is False
            and length_stats.get("official_test_accessed") is False
            and length_stats.get("count") == int(dataset["target"]["validation_count"]),
            "target_count": len(target_ids),
        },
        "manifest_integrity": {
            "passed": _manifest_hash_valid(acl_manifest) and _manifest_hash_valid(target_manifest),
            "acl_manifest_hash": acl_manifest.get("manifest_hash"),
            "target_manifest_hash": target_manifest.get("manifest_hash"),
        },
        "acquisition_and_derived_hashes": {
            "passed": acl_manifest.get("metadata_sha256")
            == sha256_file(acl_metadata_path)
            and snapshot_metadata.get("metadata_hash") == sha256_json(snapshot_metadata_payload)
            and snapshot_metadata.get("acquisition_method")
            == "pinned_huggingface_source_file"
            and snapshot_metadata.get("dataset_revision") == expected_revision
            and snapshot_metadata.get("source_url")
            == dataset["target"]["trainval_source_url"]
            and snapshot_metadata.get("source_file_sha256") == source_trainval_sha256
            and snapshot_metadata.get("source_record_count")
            == int(dataset["target"]["trainval_expected_count"])
            and snapshot_metadata.get("validation_start")
            == int(dataset["target"]["validation_start"])
            and snapshot_metadata.get("validation_count")
            == int(dataset["target"]["validation_count"])
            and snapshot_metadata.get("snapshot_sha256") == snapshot_sha256
            and target_manifest.get("source_url")
            == dataset["target"]["trainval_source_url"]
            and target_manifest.get("source_file_sha256") == source_trainval_sha256
            and target_manifest.get("source_record_count")
            == int(dataset["target"]["trainval_expected_count"])
            and pinned_source_slice_matches
            and target_manifest.get("snapshot_sha256") == snapshot_sha256
            and target_manifest.get("snapshot_metadata_hash")
            == snapshot_metadata.get("metadata_hash")
            and target_manifest.get("length_stats_hash") == sha256_json(length_stats)
            and all(
                acl_entries[source_id].get("normalized_payload_hash")
                == sha256_json(sources_by_id[source_id].to_dict())
                and acl_entries[source_id].get("document_hash")
                == sources_by_id[source_id].document_hash
                and acl_entries[source_id].get("introduction_sha256")
                == sha256_text(sources_by_id[source_id].introduction.normalized_text)
                for source_id in source_ids
            )
            and all(
                target_entries[target_id].get("visible_hash")
                == sha256_json(visible_by_id[target_id])
                and target_entries[target_id].get("evidence_hash")
                == sha256_json(evidence_by_id[target_id])
                and target_entries[target_id].get("gold_hash")
                == sha256_json(gold_by_id[target_id])
                and target_entries[target_id].get("evidence_pack_hash")
                == sha256_json(packs_by_id[target_id])
                for target_id in target_ids
            ),
            "dataset_revision": target_manifest.get("resolved_dataset_revision"),
            "source_file_sha256": target_manifest.get("source_file_sha256"),
            "snapshot_sha256": target_manifest.get("snapshot_sha256"),
        },
        "formal_tokenizer_and_budget": {
            "passed": all(pack.tokenizer_version.startswith("deepseek_formal:") for pack in packs)
            and len({pack.tokenizer_version for pack in packs}) == 1
            and all(pack.budget_tokens == 8000 and pack.post_truncation_tokens <= 8000 for pack in packs),
            "tokenizer_versions": sorted({pack.tokenizer_version for pack in packs}),
        },
        "shared_gate1_contracts": {"passed": shared["passed"], "report": shared},
    }
    return {
        "audit_version": "gate1r-v1",
        "gate": "1R",
        "run_mode": "formal",
        "data_manifest_hash": sha256_json(
            {
                "acl": acl_manifest["manifest_hash"],
                "targets": target_manifest["manifest_hash"],
            }
        ),
        "status": "PASS" if all(check["passed"] for check in checks.values()) else "FAIL",
        "checks": checks,
        "mechanics_gate1_regression": "REQUIRES_TEST_COMMAND_EVIDENCE",
    }


def write_gate1_formal_report(
    root: Path,
    dataset_config_path: Path,
    budget_config_path: Path,
    destination: Path,
) -> dict[str, Any]:
    report = build_gate1_formal_report(root, dataset_config_path, budget_config_path)
    write_json(destination, report)
    return report
