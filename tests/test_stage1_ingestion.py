from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.audits.gate1 import CONDITIONS, audit_gold_import_isolation
from src.common.jsonio import read_json
from src.ingest.text import normalize_introduction
from src.cli.prepare_pilot import prepare_pilot


ROOT = Path(__file__).resolve().parents[1]


class Stage1IngestionTests(unittest.TestCase):
    def test_sentence_coordinates_address_exact_text(self) -> None:
        introduction = normalize_introduction("One sentence. Two?\r\n\r\nThree! Final fragment")
        for paragraph in introduction.paragraphs:
            for sentence in paragraph.sentences:
                self.assertEqual(
                    introduction.normalized_text[sentence.char_start:sentence.char_end],
                    sentence.text,
                )

    def test_compiler_writer_packages_do_not_reference_gold(self) -> None:
        result = audit_gold_import_isolation(ROOT)
        self.assertTrue(result["passed"], result["violations"])

    def test_end_to_end_gate1_for_20_plus_10_fixture(self) -> None:
        with TemporaryDirectory() as directory:
            temp_root = Path(directory)
            dataset = load_fixture_config(temp_root)
            report, audit_path = prepare_pilot(dataset, ROOT / "configs" / "budget.yaml", temp_root)
            self.assertTrue(report["passed"], report)
            self.assertEqual(report["checks"]["pilot_counts"]["actual_source_count"], 20)
            self.assertEqual(report["checks"]["pilot_counts"]["actual_target_count"], 10)
            stored = read_json(audit_path)
            self.assertEqual(stored, report)
            for target in report["checks"]["evidence_pack_reuse"]["targets"].values():
                self.assertEqual(set(target["condition_assignments"]), set(CONDITIONS))
                self.assertEqual(len(set(target["condition_assignments"].values())), 1)
            for target_id in report["checks"]["evidence_pack_reuse"]["targets"]:
                visible = read_json(temp_root / "data/target/visible" / f"{target_id}.json")
                evidence = read_json(temp_root / "data/target/evidence" / f"{target_id}.json")
                gold = read_json(temp_root / "data/target/gold" / f"{target_id}.json")
                self.assertNotIn("introduction", visible)
                self.assertNotIn("introduction", evidence)
                self.assertIn("introduction", gold)

            audit_bytes = audit_path.read_bytes()
            pack_path = temp_root / "artifacts/evidence_packs/ncphysics_fixture_001.json"
            pack_bytes = pack_path.read_bytes()
            repeated_report, repeated_audit_path = prepare_pilot(
                dataset, ROOT / "configs" / "budget.yaml", temp_root
            )
            self.assertEqual(repeated_report, report)
            self.assertEqual(repeated_audit_path.read_bytes(), audit_bytes)
            self.assertEqual(pack_path.read_bytes(), pack_bytes)


def load_fixture_config(temp_root: Path) -> Path:
    import json
    import shutil

    source = temp_root / "data/pilot/input/source/pilot_sources.jsonl"
    target = temp_root / "data/pilot/input/target/pilot_targets.jsonl"
    source.parent.mkdir(parents=True)
    target.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / "data/pilot/input/source/pilot_sources.jsonl", source)
    shutil.copyfile(ROOT / "data/pilot/input/target/pilot_targets.jsonl", target)
    config = json.loads((ROOT / "configs/dataset.yaml").read_text(encoding="utf-8"))
    path = temp_root / "configs/dataset.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
    return path


if __name__ == "__main__":
    unittest.main()
