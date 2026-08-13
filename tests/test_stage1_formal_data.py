from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.budget.formal_tokenizer import DeepSeekFormalTokenizer, tokenizer_asset_hash
from src.audits.gate1_formal import build_gate1_formal_report
from src.cli.prepare_acl_corpus import prepare_acl_corpus
from src.cli.prepare_nc_physics_targets import prepare_nc_physics_targets
from src.common.jsonio import read_json, sha256_json, write_json
from src.ingest.formal_data import (
    adapt_nc_physics_record,
    derive_development_length_statistics,
    deterministic_select,
    extract_introduction,
    parse_acl_anthology_metadata,
    select_nc_physics_development_records,
)


class FakeTokenizerBackend:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(character) for character in text]

    def decode(self, tokens: list[int], skip_special_tokens: bool = False) -> str:
        return "".join(chr(token) for token in tokens)


def formal_tokenizer() -> DeepSeekFormalTokenizer:
    return DeepSeekFormalTokenizer(
        FakeTokenizerBackend(),
        model_repo="deepseek-ai/DeepSeek-V4-Flash",
        revision="test-revision",
        asset_hash="a" * 64,
    )


def nc_record(index: int) -> dict:
    introduction = (
        f"This is the development Introduction for physics paper {index}. "
        "It motivates the problem, establishes the gap, and states the contribution. " * 4
    )
    return {
        "unique_id": f"paper-{index}",
        "subfield": "Physics",
        "title": f"Physics paper {index}",
        "abstract": f"Abstract evidence for paper {index}.",
        "sections": [
            {"section": "1 Introduction", "content": introduction},
            {"section": "2 Methods", "content": "Method evidence and experimental setup."},
            {"section": "3 Results", "content": "Measured results from the experiment."},
        ],
        "references": [{"idx": "1.", "title": "Reference title", "link": "https://doi.org/x"}],
        "core_idea": "derived annotation that must not enter evidence",
        "entities": ["derived entity"],
    }


ACL_XML = b"""<?xml version='1.0' encoding='UTF-8'?>
<collection id="2024.acl"><volume id="long"><paper id="1">
<title>First <fixed-case>ACL</fixed-case> Paper</title>
<author><first>Ada</first><last>One</last></author><url>2024.acl-long.1</url><doi>10.1/one</doi>
</paper><paper id="2"><title>Second Paper</title>
<author><first>Ben</first><last>Two</last></author><url>2024.acl-long.2</url><doi>10.1/two</doi>
</paper><paper id="3"><title>Third Paper</title>
<author><first>Cy</first><last>Three</last></author><url>2024.acl-long.3</url><doi>10.1/three</doi>
</paper></volume></collection>"""


class Stage1FormalDataTests(unittest.TestCase):
    def test_acl_metadata_parse_and_deterministic_selection(self) -> None:
        papers = parse_acl_anthology_metadata(ACL_XML)
        self.assertEqual(len(papers), 3)
        self.assertEqual(papers[0]["title"], "First ACL Paper")
        first = deterministic_select(papers, id_field="source_id", count=2, seed="fixed")
        second = deterministic_select(list(reversed(papers)), id_field="source_id", count=2, seed="fixed")
        self.assertEqual([item["source_id"] for item in first], [item["source_id"] for item in second])

    def test_introduction_extraction_stops_at_next_peer_heading(self) -> None:
        markdown = """# Paper title

## 1 Introduction

""" + ("Introduction evidence sentence. " * 12) + """

## 2 Related Work

This must not be included.
"""
        introduction = extract_introduction(markdown)
        self.assertIn("Introduction evidence", introduction)
        self.assertNotIn("must not be included", introduction)

    def test_plain_two_column_text_dehyphenates_and_stops_at_next_section(self) -> None:
        text = (
            "1 Introduction\n"
            + ("This formal introduc-\ntion contains grounded evidence. " * 6)
            + "\n*Corresponding authors.\n181\n"
            + ("Further grounded evidence remains. " * 6)
            + "\n2 Related Work\nThis section must not be included."
        )
        introduction = extract_introduction(text)
        self.assertIn("formal introduction", introduction)
        self.assertNotIn("Corresponding authors", introduction)
        self.assertNotIn("181", introduction)
        self.assertNotIn("must not be included", introduction)

    def test_plain_section_boundary_ignores_year_and_numbered_url_footnotes(self) -> None:
        text = (
            "1 Introduction\n"
            + ("Prior work provides evidence. " * 5)
            + "\n2023) demonstrated a relevant result.\n"
            + "1 UKPLab/Triple-Encoders\n"
            + ("The introduction continues with grounded evidence. " * 6)
            + "\n2 Related Work\nThis section must not be included."
        )
        introduction = extract_introduction(text)
        self.assertIn("2023) demonstrated", introduction)
        self.assertIn("introduction continues", introduction)
        self.assertNotIn("UKPLab/Triple-Encoders", introduction)
        self.assertNotIn("must not be included", introduction)

    def test_nc_adapter_separates_intro_and_excludes_derived_annotations(self) -> None:
        adapted = adapt_nc_physics_record(nc_record(1))
        self.assertIn("gold_introduction", adapted)
        self.assertNotIn("Introduction", adapted["non_intro_sections"])
        serialized_evidence = json.dumps(adapted["non_intro_sections"])
        self.assertNotIn("derived annotation", serialized_evidence)
        self.assertEqual(
            adapted["acquisition_metadata"]["excluded_dataset_fields"],
            ["core_idea", "entities"],
        )

    def test_nc_development_selector_refuses_test_split(self) -> None:
        records = [nc_record(index) for index in range(5)]
        with self.assertRaisesRegex(ValueError, "train split"):
            select_nc_physics_development_records(
                records,
                split_name="test",
                validation_start=0,
                validation_count=5,
                target_count=2,
                seed="fixed",
            )

    def test_development_length_stats_contain_only_aggregates(self) -> None:
        introductions = [adapt_nc_physics_record(nc_record(index))["gold_introduction"] for index in range(4)]
        stats = derive_development_length_statistics(introductions, formal_tokenizer())
        self.assertEqual(stats["count"], 4)
        self.assertIn("median_words", stats)
        self.assertIn("p25_tokens", stats)
        self.assertNotIn(introductions[0], json.dumps(stats))
        self.assertTrue(stats["tokenizer_version"].startswith("deepseek_formal:"))

    def test_formal_tokenizer_records_asset_hash_and_revision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "tokenizer.json").write_text('{"version":"test"}', encoding="utf-8")
            asset_hash = tokenizer_asset_hash(root)
            tokenizer = DeepSeekFormalTokenizer(
                FakeTokenizerBackend(),
                model_repo="deepseek-ai/DeepSeek-V4-Flash",
                revision="abc123",
                asset_hash=asset_hash,
            )
            text = "formal budget"
            self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)
            self.assertIn("@abc123", tokenizer.version)
            self.assertIn(asset_hash, tokenizer.version)

    def test_acl_prepare_writes_real_namespace_manifest_without_pdf_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = formal_dataset_config(expected_sources=2, expected_targets=2, validation_count=4)
            config_path = write_config(root, "dataset_formal_pilot.yaml", config)

            def fetcher(url: str) -> bytes:
                return ACL_XML if url.endswith("2024.acl.xml") else b"%PDF-fake"

            markdown = "## 1 Introduction\n\n" + ("Grounded ACL introduction sentence. " * 12) + "\n\n## 2 Method\nNot intro."
            manifest, manifest_path = prepare_acl_corpus(
                config_path,
                project_root=root,
                accept_source_licenses=True,
                fetcher=fetcher,
                pdf_to_markdown=lambda _: markdown,
            )
            self.assertEqual(manifest["ready_count"], 2)
            self.assertEqual(manifest["run_mode"], "formal")
            self.assertTrue(manifest_path.exists())
            raw_manifest = manifest_path.read_text(encoding="utf-8")
            self.assertNotIn("Grounded ACL introduction sentence", raw_manifest)
            for entry in manifest["entries"]:
                normalized = read_json(root / "data/formal/source/normalized" / f"{entry['source_id']}.json")
                self.assertEqual(normalized["source_id"], entry["source_id"])

    def test_nc_prepare_uses_only_train_validation_and_writes_isolated_targets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = formal_dataset_config(expected_sources=2, expected_targets=2, validation_count=4)
            config_path = write_config(root, "dataset_formal_pilot.yaml", config)
            budget_path = write_config(
                root,
                "budget_formal.yaml",
                {
                    "config_version": "test-formal-budget-1",
                    "run_mode": "formal",
                    "tokenizer": {},
                    "target_evidence_tokens": 8000,
                    "writing_condition_tokens": 4000,
                },
            )
            requested: list[str] = []

            def file_fetcher(url: str, destination: Path) -> None:
                requested.append(url)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(
                    "\n".join(json.dumps(nc_record(index)) for index in range(4)) + "\n",
                    encoding="utf-8",
                )

            manifest, _ = prepare_nc_physics_targets(
                config_path,
                budget_path,
                project_root=root,
                accept_source_licenses=True,
                tokenizer=formal_tokenizer(),
                file_fetcher=file_fetcher,
            )
            self.assertEqual(manifest["selected_count"], 2)
            self.assertFalse(manifest["official_test_accessed"])
            self.assertEqual(manifest["resolved_dataset_revision"], "dataset-revision")
            self.assertEqual(
                requested,
                [
                    "https://example.invalid/resolve/dataset-revision/"
                    "NC_Physics_trainval.jsonl"
                ],
            )
            for entry in manifest["entries"]:
                target_id = entry["target_id"]
                visible = read_json(root / "data/formal/target/visible" / f"{target_id}.json")
                evidence = read_json(root / "data/formal/target/evidence" / f"{target_id}.json")
                gold = read_json(root / "data/formal/target/gold" / f"{target_id}.json")
                self.assertNotIn("introduction", visible)
                self.assertNotIn("introduction", evidence)
                self.assertIn("introduction", gold)
            stats = read_json(
                root / "artifacts/formal_pilot/formal-pilot-v1/audits/length_stats.json"
            )
            self.assertEqual(stats["count"], 4)
            self.assertFalse(stats["official_test_accessed"])

    def test_gate1r_revalidates_revision_snapshot_and_all_derived_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = formal_dataset_config(
                expected_sources=2, expected_targets=2, validation_count=4
            )
            config_path = write_config(root, "dataset_formal_pilot.yaml", config)
            budget_path = write_config(
                root,
                "budget_formal.yaml",
                {
                    "config_version": "test-formal-budget-1",
                    "run_mode": "formal",
                    "tokenizer": {},
                    "target_evidence_tokens": 8000,
                    "writing_condition_tokens": 4000,
                },
            )

            def acl_fetcher(url: str) -> bytes:
                return ACL_XML if url.endswith("2024.acl.xml") else b"%PDF-fake"

            markdown = (
                "## 1 Introduction\n\n"
                + ("Grounded ACL introduction sentence. " * 12)
                + "\n\n## 2 Method\nNot intro."
            )
            prepare_acl_corpus(
                config_path,
                project_root=root,
                accept_source_licenses=True,
                fetcher=acl_fetcher,
                pdf_to_markdown=lambda _: markdown,
            )

            def nc_file_fetcher(url: str, destination: Path) -> None:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(
                    "\n".join(json.dumps(nc_record(index)) for index in range(4)) + "\n",
                    encoding="utf-8",
                )

            prepare_nc_physics_targets(
                config_path,
                budget_path,
                project_root=root,
                accept_source_licenses=True,
                tokenizer=formal_tokenizer(),
                file_fetcher=nc_file_fetcher,
            )
            report = build_gate1_formal_report(root, config_path, budget_path)
            self.assertEqual(report["status"], "PASS", report.get("checks"))
            self.assertTrue(report["checks"]["acquisition_and_derived_hashes"]["passed"])

            target_id = read_json(root / "data/manifests/nc_physics_pilot.json")["entries"][0][
                "target_id"
            ]
            visible_path = root / "data/formal/target/visible" / f"{target_id}.json"
            original_visible = visible_path.read_text(encoding="utf-8")
            visible = read_json(visible_path)
            visible["title"] = "tampered"
            visible_path.write_text(json.dumps(visible), encoding="utf-8")
            tampered = build_gate1_formal_report(root, config_path, budget_path)
            self.assertEqual(tampered["status"], "FAIL")
            self.assertFalse(tampered["checks"]["acquisition_and_derived_hashes"]["passed"])

            visible_path.write_text(original_visible, encoding="utf-8")
            source_path = root / "data/raw/nc_physics/trainval.jsonl"
            original_source = source_path.read_text(encoding="utf-8")
            source_lines = original_source.splitlines()
            source_lines[0] = json.dumps({**json.loads(source_lines[0]), "title": "tampered"})
            source_path.write_text("\n".join(source_lines) + "\n", encoding="utf-8")
            source_tampered = build_gate1_formal_report(root, config_path, budget_path)
            self.assertEqual(source_tampered["status"], "FAIL")
            self.assertFalse(
                source_tampered["checks"]["acquisition_and_derived_hashes"]["passed"]
            )

            source_path.write_text(original_source, encoding="utf-8")
            target_manifest_path = root / "data/manifests/nc_physics_pilot.json"
            target_manifest = read_json(target_manifest_path)
            target_manifest["source_article_license_confirmation"] = False
            target_manifest.pop("manifest_hash")
            target_manifest["manifest_hash"] = sha256_json(target_manifest)
            write_json(target_manifest_path, target_manifest)
            unconfirmed = build_gate1_formal_report(root, config_path, budget_path)
            self.assertEqual(unconfirmed["status"], "FAIL")
            self.assertFalse(unconfirmed["checks"]["real_nc_development_targets"]["passed"])

    def test_nc_acquisition_rejects_unpinned_source_url(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = formal_dataset_config(
                expected_sources=2, expected_targets=2, validation_count=4
            )
            config["target"]["trainval_source_url"] = (
                "https://example.invalid/resolve/main/NC_Physics_trainval.jsonl"
            )
            config_path = write_config(root, "dataset_formal_pilot.yaml", config)
            budget_path = write_config(
                root,
                "budget_formal.yaml",
                {
                    "config_version": "test-formal-budget-1",
                    "run_mode": "formal",
                    "tokenizer": {},
                    "target_evidence_tokens": 8000,
                    "writing_condition_tokens": 4000,
                },
            )
            def file_fetcher(url: str, destination: Path) -> None:
                self.fail("an unpinned source URL must be rejected before download")

            with self.assertRaisesRegex(ValueError, "not pinned"):
                prepare_nc_physics_targets(
                    config_path,
                    budget_path,
                    project_root=root,
                    accept_source_licenses=True,
                    tokenizer=formal_tokenizer(),
                    file_fetcher=file_fetcher,
                )

    def test_real_data_preparation_requires_license_confirmation_before_fetch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = write_config(
                root,
                "dataset_formal_pilot.yaml",
                formal_dataset_config(expected_sources=2, expected_targets=2, validation_count=4),
            )
            with self.assertRaises(PermissionError):
                prepare_acl_corpus(
                    config_path,
                    project_root=root,
                    accept_source_licenses=False,
                    fetcher=lambda _: self.fail("network must not be called"),
                )


def formal_dataset_config(
    *, expected_sources: int, expected_targets: int, validation_count: int
) -> dict:
    return {
        "config_version": "test-formal-dataset-1",
        "run_mode": "formal",
        "run_id": "formal-pilot-v1",
        "source": {
            "provider": "ACL Anthology",
            "metadata_url": "https://example.invalid/2024.acl.xml",
            "license_identifier": "CC-BY-4.0",
            "license_url": "https://aclanthology.org/faq/copyright/",
            "volume": "long",
            "selection_seed": "acl-seed",
            "expected_count": expected_sources,
        },
        "target": {
            "provider": "Xiao-Youth/NC_Physics",
            "dataset_revision": "dataset-revision",
            "dataset_license_identifier": "other",
            "license_review_url": "https://example.invalid/NC_Physics",
            "trainval_source_url": (
                "https://example.invalid/resolve/dataset-revision/NC_Physics_trainval.jsonl"
            ),
            "trainval_expected_count": validation_count,
            "config": "default",
            "split": "train",
            "validation_start": 0,
            "validation_count": validation_count,
            "selection_seed": "target-seed",
            "expected_count": expected_targets,
            "official_test_access": "forbidden",
        },
        "paths": {
            "acl_metadata_raw": "data/raw/acl/2024.acl.xml",
            "acl_pdf_root": "data/raw/acl/pdfs",
            "acl_markdown_root": "data/raw/acl/markdown",
            "nc_trainval_raw": "data/raw/nc_physics/trainval.jsonl",
            "nc_validation_snapshot": "data/raw/nc_physics/validation.jsonl",
            "nc_validation_snapshot_metadata": "data/raw/nc_physics/validation.meta.json",
            "source_normalized": "data/formal/source/normalized",
            "target_visible": "data/formal/target/visible",
            "target_evidence": "data/formal/target/evidence",
            "target_gold": "data/formal/target/gold",
            "evidence_packs": "artifacts/formal_pilot/formal-pilot-v1/evidence_packs",
            "representations": "artifacts/formal_pilot/formal-pilot-v1/representations",
            "experiences": "artifacts/formal_pilot/formal-pilot-v1/experiences",
            "generations": "artifacts/formal_pilot/formal-pilot-v1/generations",
            "costs": "artifacts/formal_pilot/formal-pilot-v1/costs",
            "evaluations": "artifacts/formal_pilot/formal-pilot-v1/evaluations",
            "audits": "artifacts/formal_pilot/formal-pilot-v1/audits",
            "manifests": "data/manifests",
        },
    }


def write_config(root: Path, name: str, payload: dict) -> Path:
    path = root / "configs" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


if __name__ == "__main__":
    unittest.main()
