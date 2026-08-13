from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

from src.budget.formal_tokenizer import load_formal_tokenizer
from src.budget.tokenizer import BudgetController, Tokenizer
from src.common.jsonio import (
    read_json,
    read_jsonl,
    sha256_file,
    sha256_json,
    write_json,
    write_jsonl,
)
from src.config import load_config
from src.evidence_pack.builder import build_target_evidence_pack
from src.ingest.formal_data import (
    adapt_nc_physics_record,
    derive_development_length_statistics,
    deterministic_select,
)
from src.ingest.network import fetch_huggingface_dataset_to_path
from src.ingest.target import normalize_target_record


ROOT = Path(__file__).resolve().parents[2]


def _resolve(root: Path, configured_path: str) -> Path:
    path = Path(configured_path)
    return path if path.is_absolute() else root / path


def _load_validation_pool(
    target_config: dict[str, Any], source_path: Path
) -> tuple[list[dict[str, Any]], int]:
    if target_config.get("split") != "train":
        raise ValueError("Formal pilot target acquisition refuses any split except train")
    revision = str(target_config["dataset_revision"])
    source_url = str(target_config["trainval_source_url"])
    if f"/resolve/{revision}/" not in source_url:
        raise ValueError("NC_Physics trainval source URL is not pinned to dataset_revision")
    start = int(target_config["validation_start"])
    stop = start + int(target_config["validation_count"])
    records: list[dict[str, Any]] = []
    record_count = 0
    with source_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{source_path}:{line_number}: expected a JSON object")
            if start <= record_count < stop:
                records.append(value)
            record_count += 1
    expected_source_count = int(target_config["trainval_expected_count"])
    if record_count != expected_source_count:
        raise ValueError(
            f"Expected {expected_source_count} NC_Physics trainval rows, got {record_count}"
        )
    expected_validation_count = int(target_config["validation_count"])
    if len(records) != expected_validation_count:
        raise ValueError(
            f"Expected {expected_validation_count} NC_Physics validation rows, got {len(records)}"
        )
    return records, record_count


def prepare_nc_physics_targets(
    config_path: str | Path,
    budget_config_path: str | Path,
    *,
    project_root: str | Path | None = None,
    accept_source_licenses: bool,
    tokenizer: Tokenizer | None = None,
    file_fetcher: Callable[[str, Path], None] = fetch_huggingface_dataset_to_path,
) -> tuple[dict[str, Any], Path]:
    if not accept_source_licenses:
        raise PermissionError(
            "NC_Physics processing requires explicit confirmation of source-article licenses"
        )
    config_file = Path(config_path).resolve()
    budget_file = Path(budget_config_path).resolve()
    root = Path(project_root).resolve() if project_root else config_file.parent.parent
    config = load_config(config_file)
    budget = load_config(budget_file)
    if config.get("run_mode") != "formal" or budget.get("run_mode") != "formal":
        raise ValueError("NC_Physics formal preparation requires formal dataset and budget configs")
    target_config = config["target"]
    if target_config.get("official_test_access") != "forbidden":
        raise ValueError("Formal pilot config must explicitly forbid official test access")
    revision = str(target_config["dataset_revision"])
    source_url = str(target_config["trainval_source_url"])
    if f"/resolve/{revision}/" not in source_url:
        raise ValueError("NC_Physics trainval source URL is not pinned to dataset_revision")
    paths = config["paths"]
    source_path = _resolve(root, paths["nc_trainval_raw"])
    snapshot_path = _resolve(root, paths["nc_validation_snapshot"])
    snapshot_metadata_path = _resolve(root, paths["nc_validation_snapshot_metadata"])
    if snapshot_path.exists():
        if not snapshot_metadata_path.exists() or not source_path.exists():
            raise ValueError("Existing NC_Physics snapshot lacks its pinned source or acquisition metadata")
        validation_pool = read_jsonl(snapshot_path)
        snapshot_metadata = read_json(snapshot_metadata_path)
        metadata_hash = snapshot_metadata.get("metadata_hash")
        metadata_payload = {
            key: value for key, value in snapshot_metadata.items() if key != "metadata_hash"
        }
        if metadata_hash != sha256_json(metadata_payload):
            raise ValueError("NC_Physics snapshot acquisition metadata hash is invalid")
        if snapshot_metadata.get("snapshot_sha256") != sha256_file(snapshot_path):
            raise ValueError("NC_Physics snapshot content hash does not match acquisition metadata")
        if snapshot_metadata.get("acquisition_method") != "pinned_huggingface_source_file":
            raise ValueError("NC_Physics snapshot acquisition method is invalid")
        if snapshot_metadata.get("dataset_revision") != target_config["dataset_revision"]:
            raise ValueError("NC_Physics snapshot revision does not match frozen dataset config")
        if snapshot_metadata.get("source_url") != target_config["trainval_source_url"]:
            raise ValueError("NC_Physics snapshot source URL does not match frozen dataset config")
        if snapshot_metadata.get("source_file_sha256") != sha256_file(source_path):
            raise ValueError("NC_Physics pinned source hash does not match acquisition metadata")
        derived_pool, source_record_count = _load_validation_pool(target_config, source_path)
        if derived_pool != validation_pool:
            raise ValueError("NC_Physics validation snapshot does not match its pinned source slice")
        if snapshot_metadata.get("source_record_count") != source_record_count:
            raise ValueError("NC_Physics pinned source record count does not match acquisition metadata")
    else:
        if source_path.exists() or snapshot_metadata_path.exists():
            raise ValueError("Partial NC_Physics acquisition artifacts already exist")
        file_fetcher(source_url, source_path)
        validation_pool, source_record_count = _load_validation_pool(target_config, source_path)
        write_jsonl(snapshot_path, validation_pool)
        snapshot_metadata = {
            "metadata_version": "nc-physics-snapshot-2",
            "acquisition_method": "pinned_huggingface_source_file",
            "dataset": target_config["provider"],
            "dataset_revision": target_config["dataset_revision"],
            "source_url": target_config["trainval_source_url"],
            "source_file_sha256": sha256_file(source_path),
            "source_record_count": source_record_count,
            "validation_start": int(target_config["validation_start"]),
            "validation_count": int(target_config["validation_count"]),
            "row_count": len(validation_pool),
            "snapshot_sha256": sha256_file(snapshot_path),
        }
        snapshot_metadata["metadata_hash"] = sha256_json(snapshot_metadata)
        write_json(snapshot_metadata_path, snapshot_metadata)
    if len(validation_pool) != int(target_config["validation_count"]):
        raise ValueError("NC_Physics validation snapshot count does not match config")
    selected = deterministic_select(
        validation_pool,
        id_field="unique_id",
        count=int(target_config["expected_count"]),
        seed=str(target_config["selection_seed"]),
    )
    formal_tokenizer = tokenizer or load_formal_tokenizer(budget_file, root)
    controller = BudgetController(formal_tokenizer)

    visible_root = _resolve(root, paths["target_visible"])
    evidence_root = _resolve(root, paths["target_evidence"])
    gold_root = _resolve(root, paths["target_gold"])
    packs_root = _resolve(root, paths["evidence_packs"])
    selected_entries: list[dict[str, Any]] = []
    for raw in selected:
        adapted = adapt_nc_physics_record(raw)
        visible, evidence, gold = normalize_target_record(adapted)
        write_json(visible_root / f"{visible.target_id}.json", visible.to_dict())
        write_json(evidence_root / f"{evidence.target_id}.json", evidence.to_dict())
        write_json(gold_root / f"{gold.target_id}.json", gold.to_dict())
        pack = build_target_evidence_pack(
            visible,
            evidence,
            budget_tokens=int(budget["target_evidence_tokens"]),
            budget_controller=controller,
        )
        write_json(packs_root / f"{pack.target_id}.json", pack.to_dict())
        selected_entries.append(
            {
                "target_id": visible.target_id,
                "unique_id": raw["unique_id"],
                "subfield": raw.get("subfield"),
                "visible_hash": sha256_json(visible.to_dict()),
                "evidence_hash": sha256_json(evidence.to_dict()),
                "gold_hash": sha256_json(gold.to_dict()),
                "evidence_pack_hash": sha256_json(pack.to_dict()),
                "evidence_pack_tokens": pack.post_truncation_tokens,
            }
        )

    adapted_pool = [adapt_nc_physics_record(record) for record in validation_pool]
    length_stats = derive_development_length_statistics(
        (record["gold_introduction"] for record in adapted_pool), formal_tokenizer
    )
    length_stats.update(
        {
            "run_mode": "formal",
            "dataset_revision": target_config["dataset_revision"],
            "official_test_accessed": False,
        }
    )
    audits_root = _resolve(root, paths["audits"])
    write_json(audits_root / "length_stats.json", length_stats)
    manifest = {
        "manifest_version": "nc-physics-formal-pilot-2",
        "run_mode": "formal",
        "dataset_config_version": config["config_version"],
        "dataset": target_config["provider"],
        "dataset_license_identifier": target_config["dataset_license_identifier"],
        "license_review_url": target_config["license_review_url"],
        "source_article_license_confirmation": True,
        "dataset_revision": target_config["dataset_revision"],
        "resolved_dataset_revision": target_config["dataset_revision"],
        "source_split": "train",
        "validation_start": int(target_config["validation_start"]),
        "validation_count": int(target_config["validation_count"]),
        "selection_seed": target_config["selection_seed"],
        "expected_count": int(target_config["expected_count"]),
        "selected_count": len(selected_entries),
        "official_test_accessed": False,
        "source_url": target_config["trainval_source_url"],
        "source_file_sha256": sha256_file(source_path),
        "source_record_count": source_record_count,
        "snapshot_sha256": sha256_file(snapshot_path),
        "snapshot_metadata_hash": snapshot_metadata["metadata_hash"],
        "tokenizer_version": formal_tokenizer.version,
        "length_stats_hash": sha256_json(length_stats),
        "entries": selected_entries,
    }
    manifest["manifest_hash"] = sha256_json(manifest)
    manifest_path = _resolve(root, paths["manifests"]) / "nc_physics_pilot.json"
    write_json(manifest_path, manifest)
    return manifest, manifest_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare real NC_Physics development targets")
    parser.add_argument("--profile", choices=("pilot",), default="pilot")
    parser.add_argument("--config", default="configs/dataset_formal_pilot.yaml")
    parser.add_argument("--budget-config", default="configs/budget_formal.yaml")
    parser.add_argument("--accept-source-licenses", action="store_true")
    args = parser.parse_args(argv)
    try:
        manifest, path = prepare_nc_physics_targets(
            args.config,
            args.budget_config,
            accept_source_licenses=args.accept_source_licenses,
        )
    except (PermissionError, FileNotFoundError) as exc:
        print("NC_PHYSICS_TARGETS=BLOCKED")
        print(f"BLOCKER={exc}")
        return 2
    passed = manifest["selected_count"] == manifest["expected_count"]
    print(f"NC_PHYSICS_TARGETS={'PASS' if passed else 'FAIL'}")
    print(f"MANIFEST={path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
