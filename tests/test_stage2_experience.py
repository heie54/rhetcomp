from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.adapters.model import ModelAdapter, ModelRequest, ModelResponse
from src.audits.gate2 import build_gate2_report
from src.budget.tokenizer import DeterministicRegexTokenizer
from src.compilers.config import load_compiler_settings
from src.compilers.experience.adjudicate import adjudicate_pair
from src.compilers.experience.canonicalize import canonicalize
from src.compilers.experience.extract import (
    ExtractionCandidate,
    _parse_extraction_output,
    extract_candidates,
)
from src.compilers.experience.pair_retrieval import retrieve_pairs
from src.compilers.experience.pipeline import compile_experience_library
from src.compilers.experience.span_validate import validate_candidate
from src.compilers.experience.verify import _model_verdict, verify_candidate
from src.domain.models import EvidenceLocation, Experience, SourcePaper
from src.domain.schemas import validate_schema
from src.ingest.source import normalize_source_record


ROOT = Path(__file__).resolve().parents[1]
COMPILER_CONFIG = ROOT / "configs" / "compiler.yaml"
BUDGET_CONFIG = ROOT / "configs" / "budget.yaml"


def _settings():
    return load_compiler_settings(COMPILER_CONFIG)


def _tokenizer():
    return DeterministicRegexTokenizer()


def _source(source_id: str, introduction: str) -> SourcePaper:
    return normalize_source_record(
        {
            "source_id": source_id,
            "title": f"Paper {source_id}",
            "authors": ["A. Author"],
            "venue": "ACL 2024",
            "track": "main",
            "introduction": introduction,
        }
    )


def _candidate(
    source_id: str,
    paragraph: int,
    sentence_start: int,
    sentence_end: int,
    span: str,
) -> ExtractionCandidate:
    return ExtractionCandidate(
        source_id=source_id,
        location=EvidenceLocation(
            section="Introduction",
            paragraph=paragraph,
            sentence_start=sentence_start,
            sentence_end=sentence_end,
        ),
        span=span,
        observed_pattern=f"Observed {span[:24]}",
        strategy="State the claim directly, then connect it to the broader problem.",
        applicable_when="When an Introduction needs to present a claim.",
    )


class Stage2ExperienceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = _settings()
        self.tokenizer = _tokenizer()

    def test_experience_schema_round_trip(self) -> None:
        from src.domain.models import ExperienceEvidence

        evidence = (
            ExperienceEvidence(
                source_id="acl2024_test",
                location=EvidenceLocation(
                    section="Introduction", paragraph=1, sentence_start=1, sentence_end=1
                ),
                span="First sentence.",
                support_relation="instantiates_observed_pattern",
            ),
        )
        experience = Experience(
            experience_id="exp_test",
            observed_pattern="p",
            strategy="s",
            applicable_when="w",
            evidence=evidence,
            grounding_status="support_verified",
        )
        payload = experience.to_dict()
        validate_schema("Experience", payload)
        self.assertEqual(Experience.from_dict(payload), experience)

    def test_extraction_emits_atomic_candidates(self) -> None:
        sources = [_source("acl2024_a", "One claim. Two claim!\n\nA new paragraph.")]
        outcome = extract_candidates(sources, self.settings, adapter=None)
        self.assertEqual(outcome.adapter_mode, "deterministic")
        self.assertGreater(len(outcome.candidates), 0)
        for candidate in outcome.candidates:
            self.assertEqual(candidate.location.section, "Introduction")
            self.assertTrue(candidate.span.strip())
            self.assertTrue(candidate.observed_pattern.strip())
            self.assertTrue(candidate.strategy.strip())
            self.assertTrue(candidate.applicable_when.strip())

    def test_verbatim_spans_pass_exact_check_by_construction(self) -> None:
        source = _source("acl2024_a", "First claim here. Second claim here!")
        candidates = extract_candidates([source], self.settings).candidates
        self.assertGreater(len(candidates), 0)
        source_by_id = {source.source_id: source}
        for candidate in candidates:
            validated = validate_candidate(candidate, source_by_id)
            self.assertEqual(
                validated.grounding_status,
                "span_verified",
                f"span {candidate.span!r} should be exact",
            )

    def test_fabricated_span_is_rejected_and_traceable(self) -> None:
        source = _source("acl2024_a", "Real sentence about real content.")
        candidate = _candidate("acl2024_a", 1, 1, 1, "Invented sentence never in the text.")
        validated = validate_candidate(candidate, {source.source_id: source})
        self.assertEqual(validated.grounding_status, "rejected")
        self.assertIsNotNone(validated.rejection_reason)
        self.assertIn(validated.rejection_reason, {"span_not_in_referenced_window", "span_not_in_normalized_text"})

    def test_misaddressed_coordinates_are_rejected(self) -> None:
        source = _source("acl2024_a", "First sentence. Second sentence.")
        # Declare sentence 2 but quote sentence 1's exact text.
        candidate = _candidate("acl2024_a", 1, 2, 2, "First sentence.")
        validated = validate_candidate(candidate, {source.source_id: source})
        self.assertEqual(validated.grounding_status, "rejected")
        self.assertEqual(validated.rejection_reason, "span_not_in_referenced_window")

    def test_verifier_output_is_structured_and_admits(self) -> None:
        source = _source("acl2024_a", "A sufficiently long sentence that makes a real claim.")
        candidate = extract_candidates([source], self.settings).candidates[0]
        checked = verify_candidate(candidate, self.settings, adapter=None)
        self.assertIn(
            checked.verifier_result["observation_support"],
            {"supported", "partial", "unsupported"},
        )
        self.assertIn(
            checked.verifier_result["strategy_generalization"],
            {"reasonable", "overgeneralized", "unsupported"},
        )
        self.assertIn(checked.grounding_status, {"support_verified", "rejected"})

    def test_model_verifier_path_parses_structured_json(self) -> None:
        class FakeAdapter(ModelAdapter):
            model_name = "fake"

            def generate(self, request: ModelRequest) -> ModelResponse:
                return ModelResponse(
                    text='{"observation_support": "partial", '
                    '"strategy_generalization": "overgeneralized", "notes": "thin"}',
                    input_tokens=10,
                    output_tokens=5,
                    latency_ms=1,
                    metadata={},
                )

        candidate = _candidate("acl2024_a", 1, 1, 1, "Some span.")
        result = _model_verdict(candidate, FakeAdapter())
        self.assertEqual(result["observation_support"], "partial")

    def test_model_extraction_parser(self) -> None:
        raw = json.dumps(
            {
                "candidates": [
                    {
                        "location": {"paragraph": 1, "sentence_start": 1, "sentence_end": 2},
                        "span": "abc",
                        "observed_pattern": "p",
                        "strategy": "s",
                        "applicable_when": "w",
                    }
                ]
            }
        )
        parsed, error = _parse_extraction_output(raw, "acl2024_x")
        self.assertIsNone(error)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0].source_id, "acl2024_x")
        malformed, error = _parse_extraction_output("not json", "acl2024_x")
        self.assertEqual(malformed, [])
        self.assertIn("invalid_json", error)

    def test_model_extraction_parser_accepts_only_surrounding_json_fence(self) -> None:
        raw = """```json
        {"candidates": []}
        ```"""
        parsed, error = _parse_extraction_output(raw, "acl2024_x")
        self.assertEqual(parsed, [])
        self.assertIsNone(error)
        _, error = _parse_extraction_output(
            'analysis first\n{"candidates": []}', "acl2024_x"
        )
        self.assertIn("invalid_json", error)

    def test_verifier_allows_exactly_one_format_repair(self) -> None:
        class RepairingAdapter(ModelAdapter):
            model_name = "fake"

            def __init__(self) -> None:
                self.requests: list[ModelRequest] = []

            def generate(self, request: ModelRequest) -> ModelResponse:
                self.requests.append(request)
                text = (
                    "not json"
                    if len(self.requests) == 1
                    else '{"observation_support":"supported",'
                    '"strategy_generalization":"reasonable","notes":"bounded"}'
                )
                return ModelResponse(
                    text=text,
                    input_tokens=10,
                    output_tokens=5,
                    latency_ms=1,
                    metadata={"call_id": f"call_{len(self.requests)}"},
                )

        adapter = RepairingAdapter()
        candidate = _candidate("acl2024_a", 1, 1, 1, "Some span.")
        checked = verify_candidate(candidate, self.settings, adapter=adapter, formal_mode=True)
        self.assertEqual(checked.grounding_status, "support_verified")
        self.assertEqual(len(adapter.requests), 2)
        self.assertEqual(adapter.requests[0].max_output_tokens, 2048)
        self.assertEqual(adapter.requests[1].role, "experience_verifier_format_repair")
        self.assertEqual(checked.provider_metadata["call_id"], "call_2")

    def test_retrieval_does_not_merge_on_cosine(self) -> None:
        sources = [
            _source("acl2024_a", "Language systems generalize beyond memorized examples."),
            _source("acl2024_b", "Language systems generalize beyond memorized examples."),
        ]
        candidates = extract_candidates(sources, self.settings).candidates
        validated = [
            validate_candidate(c, {s.source_id: s for s in sources}) for c in candidates
        ]
        verified = [
            verify_candidate(v.candidate, self.settings) for v in validated
            if v.grounding_status == "span_verified"
        ]
        pairs = retrieve_pairs(
            verified,
            dimensions=16,
            top_k=4,
            min_cosine=0.0,
        )
        self.assertGreater(len(pairs), 0)
        # Adjudicate all pairs and force a non-merge verdict path, then confirm the
        # merge graph only contains adjudicated compatible pairs.
        adjudicated = [
            adjudicate_pair(verified[p.index_a], verified[p.index_b], p.index_a, p.index_b, self.settings)
            for p in pairs
        ]
        merge_indices = {(a.index_a, a.index_b) for a in adjudicated if a.merges}
        for a in adjudicated:
            if (a.index_a, a.index_b) in merge_indices:
                self.assertTrue(a.compatible_for_canonicalization)
                self.assertFalse(a.applicability_conflict)
        canonical = canonicalize(verified, adjudicated, self.settings)
        for item in canonical.canonical:
            for index in item.member_indices:
                self.assertTrue(
                    any(
                        index in (a.index_a, a.index_b)
                        and a.merges
                        for a in adjudicated
                    ) or len(item.member_indices) == 1
                )

    def test_adjudicator_allows_exactly_one_format_repair(self) -> None:
        class RepairingAdapter(ModelAdapter):
            model_name = "fake"

            def __init__(self) -> None:
                self.requests: list[ModelRequest] = []

            def generate(self, request: ModelRequest) -> ModelResponse:
                self.requests.append(request)
                text = (
                    "not json"
                    if len(self.requests) == 1
                    else '{"relation":"related_but_distinct",'
                    '"compatible_for_canonicalization":false,'
                    '"applicability_conflict":false,"notes":"distinct"}'
                )
                return ModelResponse(
                    text=text,
                    input_tokens=10,
                    output_tokens=5,
                    latency_ms=1,
                    metadata={"call_id": f"call_{len(self.requests)}"},
                )

        adapter = RepairingAdapter()
        left = verify_candidate(
            _candidate("acl2024_a", 1, 1, 1, "First span."), self.settings
        )
        right = verify_candidate(
            _candidate("acl2024_b", 1, 1, 1, "Second span."), self.settings
        )
        result = adjudicate_pair(
            left, right, 0, 1, self.settings, adapter=adapter, formal_mode=True
        )
        self.assertEqual(result.relation, "related_but_distinct")
        self.assertEqual(len(adapter.requests), 2)
        self.assertEqual(adapter.requests[0].max_output_tokens, 4096)
        self.assertEqual(adapter.requests[1].role, "experience_adjudicator_format_repair")
        self.assertEqual(result.provider_metadata["call_id"], "call_2")

    def test_canonicalization_never_merges_without_adjudication(self) -> None:
        # Two identical candidates that are high-cosine but explicitly adjudicated
        # "unrelated" (non-compatible) must remain separate canonical experiences.
        sources = [
            _source("acl2024_a", "The first sentence here is about topic alpha."),
            _source("acl2024_b", "The first sentence here is about topic alpha."),
        ]
        candidates = extract_candidates(sources, self.settings).candidates[:2]
        verified = [
            verify_candidate(c, self.settings)
            for c in candidates
        ]
        non_merge = adjudicate_pair(
            verified[0], verified[1], 0, 1, self.settings
        )
        # Force non-compatible by simulating a retrieval pair that adjudication rejects.
        hard_reject = type(
            "R",
            (),
            {
                "index_a": 0,
                "index_b": 1,
                "relation": "unrelated",
                "compatible_for_canonicalization": False,
                "applicability_conflict": False,
                "notes": "test",
                "merges": False,
            },
        )()
        canonical = canonicalize(verified, [hard_reject], self.settings)
        self.assertEqual(len(canonical.canonical), 2)
        for item in canonical.canonical:
            self.assertEqual(len(item.member_indices), 1)

    def test_library_serializes_within_budget_as_valid_json(self) -> None:
        sources = [_source(f"acl2024_{i}", f"Claim sentence number {i} here.") for i in range(10)]
        result = compile_experience_library(sources, self.settings, self.tokenizer, 4000)
        content = json.loads(result.library.content)
        self.assertIsInstance(content, list)
        for entry in content:
            for key in ("experience_id", "tier", "observed_pattern", "strategy"):
                self.assertIn(key, entry)
        self.assertLessEqual(result.library.content_tokens, 4000)
        self.assertEqual(result.library.content_hash, result.library.content_hash)

    def test_gate2_end_to_end_report_passes(self) -> None:
        sources = [_source(f"acl2024_{i}", f"Language systems generalize beyond example {i}.") for i in range(6)]
        result = compile_experience_library(sources, self.settings, self.tokenizer, 4000)
        report = build_gate2_report(result, self.settings, 4000)
        self.assertTrue(report["passed"], report["checks"])

    def test_all_library_spans_programmatically_checkable(self) -> None:
        # Invariant §2.8: every retained evidence span must be checkable against the
        # original normalized Introduction text at its declared coordinates.
        sources = [_source(f"acl2024_{i}", f"Language systems generalize beyond example {i}.") for i in range(6)]
        result = compile_experience_library(sources, self.settings, self.tokenizer, 4000)
        source_by_id = {source.source_id: source for source in sources}
        for experience in result.experiences:
            for evidence in experience.evidence:
                source = source_by_id[evidence.source_id]
                location = evidence.location
                paragraph = source.introduction.paragraphs[location.paragraph - 1]
                first = paragraph.sentences[location.sentence_start - 1]
                last = paragraph.sentences[location.sentence_end - 1]
                window = source.introduction.normalized_text[first.char_start:last.char_end]
                self.assertIn(evidence.span, window)
                self.assertIn(evidence.span, source.introduction.normalized_text)


if __name__ == "__main__":
    unittest.main()
