from __future__ import annotations

import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path

from src.adapters.model import ModelRequest, ModelResponse
from src.adapters.records import ProviderCallArtifact
from src.audits.gate3_formal import build_gate3_formal_report
from src.budget.tokenizer import DeterministicRegexTokenizer
from src.cli.compile_representations_formal import compile_representations_formal
from src.compilers.base import build_representation
from src.compilers.config import load_compiler_settings
from src.compilers.formal_representations import (
    ComputeEnvelope,
    RepresentationBudgetMetrics,
    compile_guideline_formal,
    compile_raw_formal,
    compile_summary_formal,
)
from src.domain.models import SourcePaper
from src.formal_metadata import FormalArtifactMetadata
from src.ingest.source import normalize_source_record


ROOT = Path(__file__).resolve().parents[1]
SETTINGS = load_compiler_settings(ROOT / "configs" / "compiler_formal.yaml")
METADATA = FormalArtifactMetadata("formal-test", "c" * 64, "d" * 64)
PROFILE_HASH = "b" * 64


class FakeFormalTokenizer(DeterministicRegexTokenizer):
    @property
    def version(self) -> str:
        return "deepseek_formal:test@revision:" + "a" * 64


def _source(source_id: str, text: str) -> SourcePaper:
    return normalize_source_record(
        {
            "source_id": source_id,
            "title": source_id,
            "authors": ["A. Author"],
            "venue": "ACL 2024",
            "track": "main-long",
            "introduction": text,
        }
    )


@dataclass
class RecordingCompilerAdapter:
    requests: list[ModelRequest] = field(default_factory=list)

    @property
    def model_name(self) -> str:
        return "deepseek-v4-flash"

    def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        text = (
            "Compressed source-corpus findings without actionable instructions."
            if request.role == "summary_compiler"
            else "When introducing a problem, state its scope before the contribution."
        )
        return ModelResponse(
            text=text,
            input_tokens=max(1, len(request.user_prompt.split())),
            output_tokens=max(1, len(text.split())),
            latency_ms=1,
            metadata={"provider": "mock", "formal": False},
        )


def _call(role: str, index: int) -> ProviderCallArtifact:
    return ProviderCallArtifact(
        call_id=f"call_{index}",
        run_id="formal-test",
        role=role,
        provider="deepseek",
        gateway="opencode_go",
        requested_model="deepseek-v4-flash",
        returned_model="deepseek-v4-flash",
        provider_profile_hash="b" * 64,
        thinking_mode="enabled",
        reasoning_effort="high",
        prompt_hash="c" * 64,
        input_hash="d" * 64,
        response_hash="e" * 64,
        input_tokens=10,
        output_tokens=5,
        latency_ms=1,
        system_fingerprint=None,
        provider_request_id=f"req_{index}",
        retry_count=0,
        status="success",
        run_mode="formal",
        config_hash=METADATA.config_hash,
        data_manifest_hash=METADATA.data_manifest_hash,
    )


def _embedding_call(index: int, returned_model: str = "qwen3.7-text-embedding") -> ProviderCallArtifact:
    return ProviderCallArtifact(
        call_id=f"call_{index}",
        run_id="formal-test",
        role="experience_candidate_retrieval",
        provider="qwen",
        gateway="alibaba_model_studio",
        requested_model="qwen3.7-text-embedding",
        returned_model=returned_model,
        provider_profile_hash="q" * 64,
        thinking_mode="not_applicable",
        reasoning_effort=None,
        prompt_hash="0" * 64,
        input_hash="d" * 64,
        response_hash="e" * 64,
        input_tokens=10,
        output_tokens=0,
        latency_ms=1,
        system_fingerprint=None,
        provider_request_id=f"req_{index}",
        retry_count=0,
        status="success",
        run_mode="formal",
        config_hash=METADATA.config_hash,
        data_manifest_hash=METADATA.data_manifest_hash,
    )


class Stage3FormalRepresentationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = FakeFormalTokenizer()
        self.sources = [
            _source("2024.acl-long.1", "The paper establishes context and scope."),
            _source("2024.acl-long.2", "The study identifies a gap and contribution."),
        ]

    def test_formal_baselines_share_hash_budget_and_request_profile(self) -> None:
        adapter = RecordingCompilerAdapter()
        raw, raw_metrics = compile_raw_formal(
            self.sources, SETTINGS, self.tokenizer, 4000, METADATA, PROFILE_HASH
        )
        summary, summary_metrics = compile_summary_formal(
            self.sources,
            SETTINGS,
            self.tokenizer,
            4000,
            adapter,
            run_id="formal-test",
            metadata=METADATA,
            provider_profile_hash=PROFILE_HASH,
        )
        guideline, guideline_metrics = compile_guideline_formal(
            self.sources,
            SETTINGS,
            self.tokenizer,
            4000,
            adapter,
            ComputeEnvelope(calls=3, input_tokens=300, output_tokens=60),
            run_id="formal-test",
            metadata=METADATA,
            provider_profile_hash=PROFILE_HASH,
        )
        self.assertEqual(
            {raw.source_corpus_hash, summary.source_corpus_hash, guideline.source_corpus_hash},
            {raw.source_corpus_hash},
        )
        self.assertEqual(raw.compiler_calls, 0)
        self.assertEqual(summary.compiler_calls, 1)
        self.assertEqual(guideline.compiler_calls, 3)
        self.assertLessEqual(max(raw.content_tokens, summary.content_tokens, guideline.content_tokens), 4000)
        self.assertEqual(summary_metrics.post_budget_tokens, summary.content_tokens)
        self.assertEqual(guideline_metrics.post_budget_tokens, guideline.content_tokens)
        self.assertTrue(raw_metrics.included_items)
        for request in adapter.requests:
            self.assertTrue(request.thinking_enabled)
            self.assertEqual(request.reasoning_effort, "high")
            self.assertIsNone(request.temperature)
            self.assertIsNone(request.top_p)
            self.assertIsNone(request.seed)
        self.assertEqual(
            [request.role for request in adapter.requests],
            ["summary_compiler", "guideline_compiler", "guideline_compiler", "guideline_compiler"],
        )

    def test_gate3r_freezes_nonzero_compute_match_and_nonidentical_content(self) -> None:
        adapter = RecordingCompilerAdapter()
        raw, raw_metrics = compile_raw_formal(
            self.sources, SETTINGS, self.tokenizer, 4000, METADATA, PROFILE_HASH
        )
        summary, summary_metrics = compile_summary_formal(
            self.sources,
            SETTINGS,
            self.tokenizer,
            4000,
            adapter,
            run_id="formal-test",
            metadata=METADATA,
            provider_profile_hash=PROFILE_HASH,
        )
        guideline, guideline_metrics = compile_guideline_formal(
            self.sources,
            SETTINGS,
            self.tokenizer,
            4000,
            adapter,
            ComputeEnvelope(calls=3, input_tokens=300, output_tokens=60),
            run_id="formal-test",
            metadata=METADATA,
            provider_profile_hash=PROFILE_HASH,
        )
        experience = build_representation(
            "experience",
            '[{"strategy":"Use verified evidence."}]',
            len(self.tokenizer.encode('[{"strategy":"Use verified evidence."}]')),
            raw.source_corpus_hash,
            "deepseek-v4-flash",
            "stage2r",
            guideline.compiler_input_tokens,
            60,
            guideline.compiler_calls,
            run_id=METADATA.run_id,
            run_mode="formal",
            config_hash=METADATA.config_hash,
            data_manifest_hash=METADATA.data_manifest_hash,
            provider_profile_hash=PROFILE_HASH,
        )
        experience_metrics = RepresentationBudgetMetrics(
            pre_budget_tokens=experience.content_tokens,
            post_budget_tokens=experience.content_tokens,
            compression_ratio=1.0,
            included_items=("exp_1",),
            excluded_items=(),
            tokenizer_version=self.tokenizer.version,
        )
        representations = {
            "raw": raw,
            "summary": summary,
            "guideline": guideline,
            "experience": experience,
        }
        metrics = {
            "raw": raw_metrics,
            "summary": summary_metrics,
            "guideline": guideline_metrics,
            "experience": experience_metrics,
        }
        calls = [
            _call("summary_compiler", 1),
            _call("guideline_compiler", 2),
            _call("experience_extractor", 3),
            _call("experience_verifier", 4),
            _embedding_call(5),
        ]
        report = build_gate3_formal_report(
            representations,
            metrics,
            calls,
            gate2_report={
                "status": "PASS",
                "config_hash": METADATA.config_hash,
                "data_manifest_hash": METADATA.data_manifest_hash,
            },
            writing_condition_tokens=4000,
            run_id="formal-test",
        )
        self.assertEqual(report["status"], "PASS", report["checks"])
        self.assertTrue(report["checks"]["formal_metadata_chain"]["passed"])
        self.assertEqual(report["config_hash"], METADATA.config_hash)
        self.assertNotEqual(guideline.content_hash, experience.content_hash)
        missing_qwen = build_gate3_formal_report(
            representations,
            metrics,
            calls[:-1],
            gate2_report={
                "status": "PASS",
                "config_hash": METADATA.config_hash,
                "data_manifest_hash": METADATA.data_manifest_hash,
            },
            writing_condition_tokens=4000,
            run_id="formal-test",
        )
        self.assertFalse(
            missing_qwen["checks"]["provider_call_artifacts_complete"]["passed"]
        )
        self.assertEqual(missing_qwen["status"], "FAIL")
        stale = build_gate3_formal_report(
            representations,
            metrics,
            calls,
            gate2_report={
                "status": "PASS",
                "config_hash": METADATA.config_hash,
                "data_manifest_hash": "f" * 64,
            },
            writing_condition_tokens=4000,
            run_id="formal-test",
        )
        self.assertFalse(stale["checks"]["gate2r_precondition"]["passed"])
        self.assertEqual(stale["status"], "FAIL")

    def test_stage3r_cli_blocks_before_live_work_without_gate2r(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report, path = compile_representations_formal(
                ROOT / "configs" / "dataset_formal_pilot.yaml",
                ROOT / "configs" / "compiler_formal.yaml",
                ROOT / "configs" / "budget_formal.yaml",
                ROOT / "configs" / "providers.yaml",
                project_root=directory,
                execute_live=True,
            )
            self.assertEqual(report["status"], "BLOCKED")
            self.assertEqual(report["blockers"][0]["reason"], "gate2r_audit_missing")
            self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
