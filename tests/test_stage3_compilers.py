from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.adapters.model import ModelAdapter, ModelRequest, ModelResponse
from src.audits.gate3 import build_cost_record, build_gate3_report
from src.budget.tokenizer import DeterministicRegexTokenizer
from src.compilers.config import load_compiler_settings
from src.compilers.experience.representation import compile_experience_representation
from src.compilers.guideline import compile_guideline
from src.compilers.raw import compile_raw
from src.compilers.summary import compile_summary
from src.domain.models import SourcePaper
from src.ingest.source import normalize_source_record


ROOT = Path(__file__).resolve().parents[1]
COMPILER_CONFIG = ROOT / "configs" / "compiler.yaml"


def _settings():
    return load_compiler_settings(COMPILER_CONFIG)


def _tokenizer():
    return DeterministicRegexTokenizer()


def _sources(count: int = 6) -> list[SourcePaper]:
    return [
        normalize_source_record(
            {
                "source_id": f"acl2024_{index}",
                "title": f"Paper {index}",
                "authors": ["A. Author"],
                "venue": "ACL 2024",
                "track": "main",
                "introduction": (
                    f"Domain {index} systems must generalize beyond memorized examples. "
                    f"Prior work often hides domain shifts in evaluation {index}.\n\n"
                    f"We introduce method {index} and state its scope."
                ),
            }
        )
        for index in range(count)
    ]


class _EchoAdapter(ModelAdapter):
    """Returns a fixed text so the model path mechanics are deterministic."""

    model_name = "echo"

    def __init__(self, text: str) -> None:
        self._text = text

    def generate(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(
            text=self._text,
            input_tokens=100,
            output_tokens=50,
            latency_ms=10,
            metadata={},
        )


class Stage3CompilerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = _settings()
        self.tokenizer = _tokenizer()
        self.sources = _sources()
        self.budget = 4000

    def _compile_all(self, adapter=None):
        raw = compile_raw(self.sources, self.settings, self.tokenizer, self.budget)
        summary = compile_summary(
            self.sources, self.settings, self.tokenizer, self.budget, adapter
        )
        guideline = compile_guideline(
            self.sources, self.settings, self.tokenizer, self.budget, adapter
        )
        experience, _ = compile_experience_representation(
            self.sources, self.settings, self.tokenizer, self.budget, adapter
        )
        return {"raw": raw, "summary": summary, "guideline": guideline, "experience": experience}

    def test_shared_source_corpus_hash(self) -> None:
        reps = self._compile_all()
        hashes = {reps[name].source_corpus_hash for name in reps}
        self.assertEqual(len(hashes), 1)

    def test_budget_respected_and_types_valid(self) -> None:
        reps = self._compile_all()
        for name, artifact in reps.items():
            with self.subTest(type=name):
                self.assertEqual(artifact.type, name)
                self.assertLessEqual(artifact.content_tokens, self.budget)
                self.assertTrue(artifact.content)
                self.assertEqual(len(artifact.content_hash), 64)
                self.assertGreater(len(artifact.representation_id), 10)

    def test_raw_contains_exemplar_introductions(self) -> None:
        raw = compile_raw(self.sources, self.settings, self.tokenizer, self.budget)
        self.assertIn("Domain 0 systems must generalize beyond memorized examples", raw.content)

    def test_summary_is_compressed_and_distinct_from_raw(self) -> None:
        reps = self._compile_all()
        self.assertLess(reps["summary"].content_tokens, reps["raw"].content_tokens)
        self.assertNotEqual(reps["summary"].content, reps["raw"].content)

    def test_guideline_is_source_derived_but_distinct_from_experience(self) -> None:
        reps = self._compile_all()
        self.assertIn("Guideline:", reps["guideline"].content)
        self.assertNotEqual(reps["guideline"].content, reps["experience"].content)
        # Guideline must not reuse the Experience library serialization.
        self.assertNotIn("observed_pattern", reps["guideline"].content)

    def test_experience_content_is_valid_json_library(self) -> None:
        experience, result = compile_experience_representation(
            self.sources, self.settings, self.tokenizer, self.budget
        )
        entries = json.loads(experience.content)
        self.assertIsInstance(entries, list)
        self.assertEqual(result.library.content_tokens, experience.content_tokens)
        self.assertIn("strategy", entries[0])

    def test_model_path_records_costs(self) -> None:
        adapter = _EchoAdapter("A compressed summary of the corpus.")
        reps = self._compile_all(adapter)
        self.assertGreater(reps["summary"].compiler_calls, 0)
        self.assertGreater(reps["guideline"].compiler_calls, 0)
        self.assertEqual(reps["summary"].compiler_model, "echo")
        self.assertGreater(reps["summary"].compiler_input_tokens, 0)
        self.assertGreater(reps["guideline"].compiler_output_tokens, 0)

    def test_gate3_report_passes(self) -> None:
        reps = self._compile_all()
        costs = {
            name: build_cost_record(artifact, mode="deterministic")
            for name, artifact in reps.items()
        }
        report = build_gate3_report(reps, costs, self.budget)
        self.assertTrue(report["passed"], report["checks"])
        self.assertEqual(report["checks"]["shared_source_corpus_hash"]["distinct_hash_count"], 1)
        self.assertTrue(report["checks"]["guideline_experience_compute_comparable"]["passed"])

    def test_representations_are_deterministic(self) -> None:
        first = self._compile_all()
        second = self._compile_all()
        for name in first:
            self.assertEqual(first[name].content_hash, second[name].content_hash)


if __name__ == "__main__":
    unittest.main()
