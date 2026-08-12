from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path

from src.audits.gate5 import build_gate5_report
from src.budget.tokenizer import BudgetController, DeterministicRegexTokenizer
from src.domain.models import RepresentationArtifact, TargetEvidence, TargetVisible
from src.evaluation.review import (
    check_experience_strategies_actionable,
    check_gold_leakage,
    check_guideline_experience_distinct,
    check_output_format_and_citations,
    check_source_domain_leakage,
    run_pilot_review,
)
from src.evidence_pack.builder import build_target_evidence_pack
from src.ingest.source import normalize_source_record
from src.writer.config import WRITER_CONDITIONS


ROOT = Path(__file__).resolve().parents[1]


def _tokenizer():
    return DeterministicRegexTokenizer()


def _source(source_id: str) -> object:
    return normalize_source_record(
        {
            "source_id": source_id,
            "title": f"Paper {source_id}",
            "authors": ["A. Author"],
            "venue": "ACL 2024",
            "track": "main",
            "introduction": "Language systems must generalize beyond memorized examples.",
        }
    )


def _pack(target_id: str = "ncphysics_t001") -> object:
    visible = TargetVisible(
        target_id=target_id,
        title="Synthetic Physics Target",
        abstract="We study a deterministic oscillator under fixed parameters.",
    )
    evidence = TargetEvidence(
        target_id=target_id,
        non_intro_sections={
            "Methods": "The fixture uses a fixed integration interval.",
            "Results": "The trajectory remains bounded.",
        },
        reference_metadata=({"title": "Ref", "year": 2024, "doi": "10.0/x"},),
    )
    return build_target_evidence_pack(visible, evidence, 8000, BudgetController(_tokenizer()))


def _representation(name: str, content: str) -> RepresentationArtifact:
    return RepresentationArtifact(
        representation_id=f"rep_{name}_test",
        type=name,
        source_corpus_hash="a" * 64,
        compiler_model="deterministic-mock-1",
        compiler_prompt_version="test",
        compiler_input_tokens=100,
        compiler_output_tokens=len(_tokenizer().encode(content)),
        compiler_calls=0,
        content=content,
        content_tokens=len(_tokenizer().encode(content)),
        content_hash="b" * 64,
    )


def _generation(target_id: str, condition: str, text: str) -> dict:
    from src.common.jsonio import sha256_text

    digest = sha256_text(f"{target_id}:{condition}:{text}")
    return {
        "generation_id": f"gen_{digest[:20]}",
        "target_id": target_id,
        "condition": condition,
        "writer_model": "deterministic-mock-1",
        "writer_prompt_hash": "c" * 64,
        "input_tokens": 100,
        "output_tokens": len(_tokenizer().encode(text)),
        "latency_ms": 0,
        "text": text,
    }


def _representations() -> dict[str, RepresentationArtifact]:
    return {
        "raw": _representation("raw", "Raw exemplar introduction text."),
        "summary": _representation("summary", "A compressed corpus summary."),
        "guideline": _representation("guideline", "Guideline: open with a problem statement."),
        "experience": _representation(
            "experience",
            json.dumps(
                [
                    {
                        "experience_id": "exp_1",
                        "tier": "stable_core",
                        "distinct_source_count": 2,
                        "observed_pattern": "The cited span states a claim.",
                        "strategy": "State the core claim directly in the opening.",
                        "applicable_when": "When an Introduction needs to present a claim.",
                        "evidence": [],
                    }
                ]
            ),
        ),
    }


class Stage5PilotTests(unittest.TestCase):
    def test_pilot_review_passes_on_clean_synthetic_run(self) -> None:
        reps = _representations()
        packs = {"ncphysics_t001": _pack()}
        gold = {"ncphysics_t001": "Withheld synthetic gold introduction for t001."}
        sources = [_source("acl2024_a"), _source("acl2024_b")]
        generations = [
            _generation("ncphysics_t001", condition, f"Introduction for {condition}.")
            for condition in WRITER_CONDITIONS
        ]
        review = run_pilot_review(
            generations, gold, packs, reps, sources, writing_condition_tokens=4000,
            writer_max_output_tokens=600,
        )
        self.assertTrue(review["passed"], review["checks"])
        self.assertEqual(review["diagnostics"]["generation_count"], 5)

    def test_gold_leakage_is_detected(self) -> None:
        reps = _representations()
        gold_text = "Withheld synthetic gold introduction for t001."
        # Leak the gold into a generation.
        generations = [
            _generation("ncphysics_t001", "raw", f"Contained {gold_text} verbatim.")
        ]
        check = check_gold_leakage(
            generations,
            {"ncphysics_t001": gold_text},
            {"ncphysics_t001": _pack()},
            reps,
        )
        self.assertFalse(check["passed"])
        self.assertEqual(check["violations"][0]["source"], "generation_raw")

    def test_source_domain_leakage_is_detected(self) -> None:
        source = _source("acl2024_a")
        source_sentence = source.introduction.paragraphs[0].sentences[0].text
        generations = [
            _generation("ncphysics_t001", "experience", f"We write: {source_sentence}")
        ]
        check = check_source_domain_leakage(generations, [source])
        self.assertFalse(check["passed"])
        self.assertEqual(check["violations"][0]["condition"], "experience")

    def test_guideline_experience_distinct_detects_identical_generations(self) -> None:
        reps = _representations()
        identical = "Same generated text for both conditions."
        generations = [
            _generation("ncphysics_t001", "guideline", identical),
            _generation("ncphysics_t001", "experience", identical),
        ]
        check = check_guideline_experience_distinct(
            reps["guideline"], reps["experience"], generations
        )
        self.assertFalse(check["passed"])
        self.assertEqual(check["identical_generation_targets"], 1)

    def test_experience_strategies_actionable(self) -> None:
        check = check_experience_strategies_actionable(
            json.dumps(
                [
                    {
                        "strategy": "State the core claim directly.",
                        "observed_pattern": "The cited span states a claim.",
                        "applicable_when": "When introducing a claim.",
                    },
                    {
                        "strategy": "The text describes prior work",  # not imperative
                        "observed_pattern": "The cited span describes prior work.",
                        "applicable_when": "When surveying prior work.",
                    },
                ]
            )
        )
        self.assertEqual(check["actionable_ratio"], 0.5)

    def test_output_format_detects_malformed_citation(self) -> None:
        generations = [
            _generation("ncphysics_t001", "raw", "Claim without citation [12x]."),
            _generation("ncphysics_t001", "summary", "Proper citation [1] and [2]."),
        ]
        check = check_output_format_and_citations(generations, writer_max_output_tokens=600)
        self.assertFalse(check["passed"])
        self.assertEqual(check["problems"][0]["issue"], "non_numeric_bracket_citation")

    def test_gate5_report(self) -> None:
        reps = _representations()
        packs = {"ncphysics_t001": _pack()}
        sources = [_source("acl2024_a")]
        generations = [
            _generation("ncphysics_t001", condition, f"Introduction for {condition}.")
            for condition in WRITER_CONDITIONS
        ]
        review = run_pilot_review(
            generations,
            {"ncphysics_t001": "gold"},
            packs,
            reps,
            sources,
            writing_condition_tokens=4000,
            writer_max_output_tokens=600,
        )
        report = build_gate5_report(
            review,
            config_versions={
                "dataset": "v1", "compiler": "v1", "budget": "v1", "writer": "v1"
            },
            prompt_versions={"writer_system": "p1", "writer_task": "p2"},
            expected_generations=5,
        )
        self.assertTrue(report["passed"], report["checks"])
        self.assertEqual(report["freeze_status"], "frozen")

    def test_generation_ids_are_anonymized(self) -> None:
        from src.writer.config import load_writer_settings
        from src.writer.writer import Writer

        settings = load_writer_settings(ROOT / "configs" / "writer.yaml")
        writer = Writer(settings, _tokenizer(), adapter=None)
        pack = _pack()
        reps = _representations()
        for condition in WRITER_CONDITIONS:
            artifact = writer.generate(pack, condition, reps.get(condition))
            self.assertNotIn(condition, artifact.generation_id)
            self.assertTrue(artifact.generation_id.startswith("gen_"))

    def test_desired_length_derivation_from_gold_distribution(self) -> None:
        from src.evaluation.derive import derive_desired_introduction_length

        derivation = derive_desired_introduction_length(
            ["One short.", "A substantially longer withheld gold introduction here."],
            _tokenizer(),
        )
        self.assertTrue(derivation["derived"])
        self.assertEqual(derivation["count"], 2)
        self.assertGreater(derivation["max_tokens"], derivation["min_tokens"])
        self.assertGreaterEqual(derivation["mean_tokens"], derivation["min_tokens"])

    def test_end_to_end_pilot_on_fixture(self) -> None:
        import tempfile

        from src.cli.prepare_pilot import prepare_pilot
        from src.cli.run_pilot import run_pilot

        with tempfile.TemporaryDirectory() as directory:
            temp_root = Path(directory)
            _copy_fixture(temp_root)
            gate1, _ = prepare_pilot(
                temp_root / "configs/dataset.yaml",
                temp_root / "configs/budget.yaml",
                project_root=temp_root,
            )
            self.assertTrue(gate1["passed"])
            report, audit_path, review_path = run_pilot(
                temp_root / "configs/dataset.yaml",
                temp_root / "configs/compiler.yaml",
                temp_root / "configs/budget.yaml",
                temp_root / "configs/writer.yaml",
                project_root=temp_root,
            )
            self.assertTrue(report["passed"], report)
            self.assertTrue(report["checks"]["pilot_scale_complete"]["passed"])
            self.assertEqual(report["freeze_status"], "frozen")
            self.assertTrue(audit_path.exists())
            self.assertTrue(review_path.exists())
            manifest = json.loads(
                (temp_root / "artifacts/generations/manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(manifest), 10)
            for condition in WRITER_CONDITIONS:
                self.assertIn(condition, manifest["ncphysics_fixture_001"])


def _copy_fixture(temp_root: Path) -> None:
    for relative in (
        "data/pilot/input/source/pilot_sources.jsonl",
        "data/pilot/input/target/pilot_targets.jsonl",
    ):
        destination = temp_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    for name in ("dataset", "budget", "compiler", "writer", "evaluation"):
        source = ROOT / "configs" / f"{name}.yaml"
        destination = temp_root / "configs" / f"{name}.yaml"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)


if __name__ == "__main__":
    unittest.main()
