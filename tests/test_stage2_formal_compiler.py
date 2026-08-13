from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path

from src.adapters.embedding.base import EmbeddingRequest, EmbeddingResponse
from src.adapters.model import ModelRequest, ModelResponse
from src.budget.tokenizer import DeterministicRegexTokenizer
from src.cli.compile_experience_formal import compile_experience_formal
from src.compilers.config import load_compiler_settings
from src.compilers.experience.pipeline import compile_experience_library
from src.domain.models import SourcePaper
from src.ingest.source import normalize_source_record


ROOT = Path(__file__).resolve().parents[1]
SETTINGS = load_compiler_settings(ROOT / "configs" / "compiler_formal.yaml")
CONFIG_HASH = "c" * 64
DATA_MANIFEST_HASH = "d" * 64


def _source(source_id: str, sentence: str) -> SourcePaper:
    return normalize_source_record(
        {
            "source_id": source_id,
            "title": f"Paper {source_id}",
            "authors": ["A. Author"],
            "venue": "ACL 2024",
            "track": "main-long",
            "introduction": sentence,
        }
    )


def _extraction(span: str, strategy: str = "State the scope directly.") -> str:
    return json.dumps(
        {
            "candidates": [
                {
                    "location": {"paragraph": 1, "sentence_start": 1, "sentence_end": 1},
                    "span": span,
                    "observed_pattern": "The opening establishes the study scope.",
                    "strategy": strategy,
                    "applicable_when": "When a scientific Introduction opens the problem.",
                }
            ]
        }
    )


VERDICT = json.dumps(
    {
        "observation_support": "supported",
        "strategy_generalization": "reasonable",
        "notes": "The span directly supports this bounded strategy.",
    }
)
ADJUDICATION = json.dumps(
    {
        "relation": "equivalent",
        "compatible_for_canonicalization": True,
        "applicability_conflict": False,
        "notes": "Same strategy and applicability.",
    }
)


@dataclass
class ScriptedFormalChat:
    responses: list[str]
    requests: list[ModelRequest] = field(default_factory=list)

    @property
    def model_name(self) -> str:
        return "deepseek-v4-flash"

    def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if not self.responses:
            raise AssertionError("Unexpected chat call")
        index = len(self.requests)
        return ModelResponse(
            text=self.responses.pop(0),
            input_tokens=10,
            output_tokens=5,
            latency_ms=1,
            metadata={
                "provider": "mock",
                "formal": False,
                "call_id": f"mock_chat_{index}",
                "returned_model": "deepseek-v4-flash",
                "system_fingerprint": "mock-only",
            },
        )


@dataclass
class ScriptedFormalEmbedding:
    requests: list[EmbeddingRequest] = field(default_factory=list)

    @property
    def model_name(self) -> str:
        return "qwen3.7-text-embedding"

    def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        self.requests.append(request)
        vectors = tuple(
            tuple(1.0 if index == 0 else 0.0 for index in range(request.dimensions))
            for _ in request.inputs
        )
        return EmbeddingResponse(
            vectors=vectors,
            model="qwen3.7-text-embedding",
            dimensions=request.dimensions,
            input_tokens=12,
            request_id="mock-request",
            latency_ms=1,
            metadata={
                "provider": "mock",
                "formal": False,
                "call_id": "mock_embedding_1",
                "cache_hits": 0,
                "cache_misses": len(request.inputs),
            },
        )


class Stage2FormalCompilerTests(unittest.TestCase):
    def test_formal_mode_requires_both_real_adapter_interfaces(self) -> None:
        source = _source("2024.acl-long.1", "The opening establishes the study scope.")
        with self.assertRaisesRegex(ValueError, "chat and embedding"):
            compile_experience_library(
                [source],
                SETTINGS,
                DeterministicRegexTokenizer(),
                4000,
                run_mode="formal",
            )

    def test_formal_pipeline_uses_thinking_and_qwen_only_for_recall(self) -> None:
        spans = (
            "The opening establishes the study scope.",
            "This opening establishes the evaluation scope.",
        )
        sources = [
            _source("2024.acl-long.1", spans[0]),
            _source("2024.acl-long.2", spans[1]),
        ]
        chat = ScriptedFormalChat(
            [_extraction(spans[0]), _extraction(spans[1]), VERDICT, VERDICT, ADJUDICATION]
        )
        embedding = ScriptedFormalEmbedding()
        result = compile_experience_library(
            sources,
            SETTINGS,
            DeterministicRegexTokenizer(),
            4000,
            adapter=chat,
            run_mode="formal",
            run_id="formal-contract-test",
            embedding_adapter=embedding,
            config_hash=CONFIG_HASH,
            data_manifest_hash=DATA_MANIFEST_HASH,
        )

        self.assertEqual(result.adapter_mode, "model:deepseek-v4-flash")
        self.assertEqual(result.embedding_backend, "qwen3_7_text_embedding")
        self.assertEqual(result.embedding_dimensions, 1024)
        self.assertEqual(result.deterministic_fallback_count, 0)
        self.assertEqual(result.retrieved_pair_count, 1)
        self.assertEqual(result.adjudicated_pair_count, 1)
        self.assertEqual(len(embedding.requests), 1)
        embedding_request = embedding.requests[0]
        self.assertEqual(embedding_request.role, "experience_candidate_retrieval")
        self.assertEqual(embedding_request.dimensions, 1024)
        for text in embedding_request.inputs:
            self.assertEqual(
                text,
                "State the scope directly.\nWhen a scientific Introduction opens the problem.",
            )
            self.assertNotIn("The opening establishes", text)
            self.assertNotIn("2024.acl", text)

        self.assertEqual(
            [request.role for request in chat.requests],
            [
                "experience_extractor",
                "experience_extractor",
                "experience_verifier",
                "experience_verifier",
                "experience_adjudicator",
            ],
        )
        for request in chat.requests:
            self.assertTrue(request.thinking_enabled)
            self.assertEqual(request.run_mode, "formal")
            self.assertEqual(request.config_hash, CONFIG_HASH)
            self.assertEqual(request.data_manifest_hash, DATA_MANIFEST_HASH)
            self.assertEqual(request.reasoning_effort, "high")
            self.assertIsNone(request.temperature)
            self.assertIsNone(request.top_p)
            self.assertIsNone(request.seed)
            self.assertEqual(request.response_format, "json_object")
        self.assertEqual(embedding_request.run_mode, "formal")
        self.assertEqual(embedding_request.config_hash, CONFIG_HASH)
        self.assertEqual(embedding_request.data_manifest_hash, DATA_MANIFEST_HASH)

    def test_extraction_gets_at_most_one_format_repair_and_no_fallback(self) -> None:
        span = "The opening establishes the study scope."
        source = _source("2024.acl-long.1", span)
        chat = ScriptedFormalChat(["not json", _extraction(span), VERDICT])
        embedding = ScriptedFormalEmbedding()
        result = compile_experience_library(
            [source],
            SETTINGS,
            DeterministicRegexTokenizer(),
            4000,
            adapter=chat,
            run_mode="formal",
            run_id="repair-test",
            embedding_adapter=embedding,
            config_hash=CONFIG_HASH,
            data_manifest_hash=DATA_MANIFEST_HASH,
        )
        self.assertEqual(result.format_repair_count, 1)
        self.assertEqual(result.deterministic_fallback_count, 0)
        self.assertEqual(
            [request.role for request in chat.requests],
            [
                "experience_extractor",
                "experience_extractor_format_repair",
                "experience_verifier",
            ],
        )
        self.assertEqual(result.support_verified_count, 1)

    def test_invalid_verifier_output_is_rejected_and_traced(self) -> None:
        span = "The opening establishes the study scope."
        source = _source("2024.acl-long.1", span)
        chat = ScriptedFormalChat([_extraction(span), "not json", "still not json"])
        result = compile_experience_library(
            [source],
            SETTINGS,
            DeterministicRegexTokenizer(),
            4000,
            adapter=chat,
            run_mode="formal",
            run_id="invalid-verifier-test",
            embedding_adapter=ScriptedFormalEmbedding(),
            config_hash=CONFIG_HASH,
            data_manifest_hash=DATA_MANIFEST_HASH,
        )
        self.assertEqual(result.verifier_rejected_count, 1)
        rejection = next(item for item in result.trace if item.get("stage") == "verify")
        self.assertEqual(rejection["rejection_reason"], "verifier_output_invalid")
        self.assertEqual(result.deterministic_fallback_count, 0)
        self.assertEqual(
            [request.role for request in chat.requests],
            [
                "experience_extractor",
                "experience_verifier",
                "experience_verifier_format_repair",
            ],
        )

    def test_formal_cli_blocks_before_credentials_when_gate1r_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report, audit_path = compile_experience_formal(
                ROOT / "configs" / "dataset_formal_pilot.yaml",
                ROOT / "configs" / "compiler_formal.yaml",
                ROOT / "configs" / "budget_formal.yaml",
                ROOT / "configs" / "providers.yaml",
                project_root=directory,
                execute_live=True,
            )
            self.assertEqual(report["status"], "BLOCKED")
            self.assertEqual(report["blockers"][0]["reason"], "gate1r_not_passed")
            self.assertTrue(audit_path.exists())


if __name__ == "__main__":
    unittest.main()
