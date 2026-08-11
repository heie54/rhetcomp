from __future__ import annotations

import unittest
from dataclasses import asdict
from json import loads
from pathlib import Path
from tempfile import TemporaryDirectory

from src.audits.gate1 import CONDITIONS, audit_gold_import_isolation
from src.budget.tokenizer import BudgetController, DeterministicRegexTokenizer
from src.common.jsonio import canonical_json, read_json
from src.domain.models import TargetEvidence, TargetVisible
from src.evidence_pack.builder import build_target_evidence_pack
from src.ingest.text import normalize_introduction
from src.cli.prepare_pilot import prepare_pilot
from src.runtime import (
    CompilerDataAccess,
    EvaluatorDataAccess,
    WriterDataAccess,
    load_compiler_runtime_config,
    load_evaluator_runtime_config,
    load_writer_runtime_config,
)


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

    def test_gold_audit_rejects_evaluator_capability_import(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            compiler = root / "src/compilers/forbidden.py"
            compiler.parent.mkdir(parents=True)
            (root / "src/writer").mkdir(parents=True)
            compiler.write_text(
                "from src.runtime import EvaluatorDataAccess\n",
                encoding="utf-8",
            )
            result = audit_gold_import_isolation(root)
            self.assertFalse(result["passed"])
            self.assertEqual(result["violations"][0]["reason"], "gold_capability_import")

    def test_runtime_gold_capability_is_evaluator_only(self) -> None:
        with TemporaryDirectory() as directory:
            temp_root = Path(directory)
            dataset = load_fixture_config(temp_root)
            compiler = CompilerDataAccess(load_compiler_runtime_config(dataset, temp_root))
            writer = WriterDataAccess(load_writer_runtime_config(dataset, temp_root))
            evaluator_config = load_evaluator_runtime_config(dataset, temp_root)
            evaluator = EvaluatorDataAccess(evaluator_config)
            gold_root = (temp_root / "data/target/gold").resolve()

            for restricted in (compiler, writer):
                with self.subTest(role=type(restricted).__name__):
                    self.assertFalse(
                        any("gold" in name.lower() for name in restricted.configured_roots())
                    )
                    self.assertNotIn(gold_root, map(Path.resolve, restricted.configured_roots().values()))
                    self.assertFalse(
                        any(
                            "gold" in name.lower()
                            for name in dir(restricted)
                            if not name.startswith("_")
                        )
                    )
                    self.assertNotIn("target_gold_root", asdict(restricted._config))
                    with self.assertRaises(AttributeError):
                        restricted.target_gold_path("ncphysics_fixture_001")

            self.assertEqual(evaluator.configured_roots(), {"target_gold": gold_root})
            self.assertEqual(
                evaluator.target_gold_path("ncphysics_fixture_001"),
                gold_root / "ncphysics_fixture_001.json",
            )
            with self.assertRaises(TypeError):
                CompilerDataAccess(evaluator_config)  # type: ignore[arg-type]
            with self.assertRaises(ValueError):
                evaluator.target_gold_path("../visible/ncphysics_fixture_001")

    def test_oversized_evidence_pack_uses_structured_truncation(self) -> None:
        fixture = read_json(ROOT / "tests/fixtures/oversized_target_evidence.json")
        sections = {
            name: " ".join([spec["text"]] * spec["repeat"])
            for name, spec in fixture["non_intro_sections"].items()
        }
        visible = TargetVisible(
            target_id=fixture["target_id"],
            title=fixture["title"],
            abstract=fixture["abstract"],
        )
        evidence = TargetEvidence(
            target_id=fixture["target_id"],
            non_intro_sections=sections,
            reference_metadata=tuple(fixture["reference_metadata"]),
        )
        tokenizer = DeterministicRegexTokenizer()
        controller = BudgetController(tokenizer)

        first = build_target_evidence_pack(visible, evidence, 8000, controller)
        second = build_target_evidence_pack(visible, evidence, 8000, controller)
        content = loads(first.content)
        direct = controller.apply(
            canonical_json(
                {
                    "target_id": visible.target_id,
                    "title": visible.title,
                    "abstract": visible.abstract,
                    "non_intro_body": evidence.non_intro_sections,
                    "reference_metadata": evidence.reference_metadata,
                }
            ),
            8000,
        )

        self.assertGreater(first.pre_truncation_tokens, 8000)
        self.assertLessEqual(first.post_truncation_tokens, 8000)
        self.assertEqual(first.post_truncation_tokens, len(tokenizer.encode(first.content)))
        self.assertEqual(content["title"], fixture["title"])
        self.assertEqual(content["abstract"], fixture["abstract"])
        self.assertEqual(content["target_id"], fixture["target_id"])
        self.assertEqual(first.content, canonical_json(content))
        self.assertEqual(first, second)
        self.assertEqual(direct.content, first.content)
        self.assertEqual(loads(direct.content), content)

    def test_end_to_end_gate1_for_20_plus_10_fixture(self) -> None:
        with TemporaryDirectory() as directory:
            temp_root = Path(directory)
            dataset = load_fixture_config(temp_root)
            report, audit_path = prepare_pilot(dataset, ROOT / "configs" / "budget.yaml", temp_root)
            self.assertTrue(report["passed"], report)
            self.assertTrue(report["checks"]["runtime_gold_isolation"]["passed"])
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
