from __future__ import annotations

import tempfile
import unittest
from dataclasses import dataclass, field, replace
from pathlib import Path

from src.adapters.model import ModelRequest, ModelResponse
from src.adapters.records import ProviderCallArtifact
from src.audits.gate4_formal import build_gate4_formal_report
from src.budget.tokenizer import BudgetController, DeterministicRegexTokenizer
from src.cli.generate_writer_formal import generate_writer_formal
from src.common.jsonio import sha256_text
from src.domain.models import RepresentationArtifact
from src.evidence_pack.builder import build_target_evidence_pack
from src.ingest.target import normalize_target_record
from src.writer.config import WRITER_CONDITIONS, load_writer_settings
from src.writer.order import condition_order
from src.writer.prompts import build_system_prompt
from src.writer.writer import Writer


ROOT = Path(__file__).resolve().parents[1]
SETTINGS = load_writer_settings(ROOT / "configs" / "writer_formal.yaml")
CONFIG_HASH = "c" * 64
DATA_MANIFEST_HASH = "d" * 64
PROFILE_HASH = "p" * 64


def _pack():
    visible, evidence, _ = normalize_target_record(
        {
            "target_id": "ncphysics_real_dev_001",
            "title": "Physics Target",
            "abstract": "We study a bounded oscillator.",
            "non_intro_sections": {"Methods": "We integrate the equations."},
            "reference_metadata": [{"title": "Allowed Ref", "year": 2024}],
            "gold_introduction": "Withheld gold.",
        }
    )
    tokenizer = DeterministicRegexTokenizer()
    return build_target_evidence_pack(visible, evidence, 8000, BudgetController(tokenizer))


def _representation(kind: str) -> RepresentationArtifact:
    return RepresentationArtifact(
        representation_id=f"rep_{kind}",
        type=kind,
        source_corpus_hash="a" * 64,
        compiler_model="deepseek-v4-flash" if kind != "raw" else "none",
        compiler_prompt_version="formal",
        compiler_input_tokens=10,
        compiler_output_tokens=5,
        compiler_calls=1,
        content=f"Distinct {kind} rhetorical guidance only.",
        content_tokens=6,
        content_hash=(kind[0] * 64),
        run_id="formal-batch",
        run_mode="formal",
        config_hash=CONFIG_HASH,
        data_manifest_hash=DATA_MANIFEST_HASH,
        provider_profile_hash=PROFILE_HASH,
    )


@dataclass
class RecordingWriterAdapter:
    text: str = "The target study examines a bounded oscillator [1]."
    requests: list[ModelRequest] = field(default_factory=list)

    @property
    def model_name(self) -> str:
        return "deepseek-v4-flash"

    def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return ModelResponse(
            text=self.text,
            input_tokens=100,
            output_tokens=12,
            latency_ms=2,
            metadata={
                "provider": "deepseek",
                "gateway": "opencode_go",
                "requested_model": "deepseek-v4-flash",
                "formal": False,
                "call_id": f"mock_{len(self.requests)}",
                "returned_model": "deepseek-v4-flash",
                "provider_profile_hash": PROFILE_HASH,
                "provider_request_id": f"req_{len(self.requests)}",
                "system_fingerprint": "fp_consistent",
            },
        )


def _call(
    condition: str,
    index: int,
    *,
    call_id: str | None = None,
    response_text: str = RecordingWriterAdapter.text,
) -> ProviderCallArtifact:
    return ProviderCallArtifact(
        call_id=call_id or f"call_{index}",
        run_id="formal-batch",
        role=f"writer:{condition}",
        provider="deepseek",
        gateway="opencode_go",
        requested_model="deepseek-v4-flash",
        returned_model="deepseek-v4-flash",
        provider_profile_hash="p" * 64,
        thinking_mode="disabled",
        reasoning_effort=None,
        prompt_hash="h" * 64,
        input_hash="i" * 64,
        response_hash=sha256_text(response_text),
        input_tokens=100,
        output_tokens=12,
        latency_ms=2,
        system_fingerprint=None,
        provider_request_id=f"req_{index}",
        retry_count=0,
        status="success",
        run_mode="formal",
        config_hash=CONFIG_HASH,
        data_manifest_hash=DATA_MANIFEST_HASH,
    )


class Stage4FormalWriterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pack = _pack()
        self.representations = {kind: _representation(kind) for kind in ("raw", "summary", "guideline", "experience")}

    def test_formal_writer_disables_thinking_and_varies_only_conditioning(self) -> None:
        adapter = RecordingWriterAdapter()
        writer = Writer(SETTINGS, DeterministicRegexTokenizer(), adapter)
        outputs = []
        for condition in WRITER_CONDITIONS:
            outputs.append(
                writer.generate(
                    self.pack,
                    condition,
                    self.representations.get(condition),
                    formal_mode=True,
                    run_id="formal-batch",
                    config_hash=CONFIG_HASH,
                    data_manifest_hash=DATA_MANIFEST_HASH,
                    provider_profile_hash=PROFILE_HASH,
                )
            )
        self.assertEqual(len({item.target_evidence_hash for item in outputs}), 1)
        self.assertEqual(len({item.base_prompt_hash for item in outputs}), 1)
        self.assertEqual(len({item.prompt_template_hash for item in outputs}), 1)
        self.assertTrue(all(item.citation_valid for item in outputs))
        self.assertTrue(all(item.run_mode == "formal" for item in outputs))
        for request in adapter.requests:
            self.assertFalse(request.thinking_enabled)
            self.assertIsNone(request.reasoning_effort)
            self.assertEqual(request.temperature, 0.0)
            self.assertIsNone(request.top_p)
            self.assertIsNone(request.seed)
            self.assertEqual(request.max_output_tokens, 600)
        self.assertEqual(len({request.system_prompt for request in adapter.requests}), 1)
        self.assertIn("only factual source", build_system_prompt(SETTINGS))
        self.assertIn("reference_metadata", build_system_prompt(SETTINGS))

    def test_citation_outside_target_reference_metadata_fails_validation(self) -> None:
        writer = Writer(
            SETTINGS,
            DeterministicRegexTokenizer(),
            RecordingWriterAdapter(text="Unsupported citation [2]."),
        )
        output = writer.generate(
            self.pack,
            "raw",
            self.representations["raw"],
            formal_mode=True,
            run_id="formal-batch",
            config_hash=CONFIG_HASH,
            data_manifest_hash=DATA_MANIFEST_HASH,
            provider_profile_hash=PROFILE_HASH,
        )
        self.assertEqual(output.citation_indices, (2,))
        self.assertFalse(output.citation_valid)

    def test_local_condition_order_is_reproducible_and_not_model_seed(self) -> None:
        first = condition_order(self.pack.target_id, "order-seed")
        second = condition_order(self.pack.target_id, "order-seed")
        self.assertEqual(first, second)
        self.assertEqual(set(first), set(WRITER_CONDITIONS))
        self.assertNotEqual(first, WRITER_CONDITIONS)

    def test_gate4r_accepts_one_consistent_five_call_batch(self) -> None:
        adapter = RecordingWriterAdapter()
        writer = Writer(SETTINGS, DeterministicRegexTokenizer(), adapter)
        order = condition_order(self.pack.target_id, "order-seed")
        generations = [
            writer.generate(
                self.pack,
                condition,
                self.representations.get(condition),
                formal_mode=True,
                run_id="formal-batch",
                config_hash=CONFIG_HASH,
                data_manifest_hash=DATA_MANIFEST_HASH,
                provider_profile_hash=PROFILE_HASH,
            )
            for condition in order
        ]
        costs = {condition: {"logged": True} for condition in WRITER_CONDITIONS}
        calls = [
            _call(
                generation.condition,
                index,
                call_id=generation.provider_metadata["call_id"],
                response_text=generation.text,
            )
            for index, generation in enumerate(generations, start=1)
        ]
        order_manifest = {
            "condition_order": list(order),
            "ordering_kind": "local_seeded_shuffle",
            "model_generation_seed": None,
            "run_mode": "formal",
            "config_hash": CONFIG_HASH,
            "data_manifest_hash": DATA_MANIFEST_HASH,
        }
        report = build_gate4_formal_report(
            generations,
            costs,
            calls,
            gate3_report={"status": "PASS", "data_manifest_hash": DATA_MANIFEST_HASH},
            order_manifest=order_manifest,
            expected_profile_hash="p" * 64,
            batch_run_id="formal-batch",
        )
        self.assertEqual(report["status"], "PASS", report["checks"])
        self.assertTrue(report["checks"]["provider_fingerprint_consistent"]["passed"])
        self.assertFalse(report["checks"]["provider_fingerprint_consistent"]["required"])
        self.assertEqual(report["checks"]["provider_fingerprint_consistent"]["fingerprints"], [])
        self.assertTrue(report["checks"]["formal_metadata_chain"]["passed"])
        self.assertTrue(report["checks"]["generation_call_traceability"]["passed"])
        self.assertEqual(report["config_hash"], CONFIG_HASH)

        mismatched_calls = [replace(calls[0], response_hash="f" * 64), *calls[1:]]
        mismatched = build_gate4_formal_report(
            generations,
            costs,
            mismatched_calls,
            gate3_report={"status": "PASS", "data_manifest_hash": DATA_MANIFEST_HASH},
            order_manifest=order_manifest,
            expected_profile_hash=PROFILE_HASH,
            batch_run_id="formal-batch",
        )
        self.assertFalse(
            mismatched["checks"]["generation_call_traceability"]["passed"]
        )
        self.assertEqual(mismatched["status"], "FAIL")

        inconsistent_calls = [
            replace(calls[0], system_fingerprint="fp_a"),
            replace(calls[1], system_fingerprint="fp_b"),
            *calls[2:],
        ]
        inconsistent = build_gate4_formal_report(
            generations,
            costs,
            inconsistent_calls,
            gate3_report={"status": "PASS", "data_manifest_hash": DATA_MANIFEST_HASH},
            order_manifest=order_manifest,
            expected_profile_hash="p" * 64,
            batch_run_id="formal-batch",
        )
        self.assertFalse(
            inconsistent["checks"]["provider_fingerprint_consistent"]["passed"]
        )
        stale_upstream = build_gate4_formal_report(
            generations,
            costs,
            calls,
            gate3_report={"status": "PASS", "data_manifest_hash": "f" * 64},
            order_manifest=order_manifest,
            expected_profile_hash="p" * 64,
            batch_run_id="formal-batch",
        )
        self.assertFalse(stale_upstream["checks"]["gate3r_precondition"]["passed"])
        self.assertEqual(stale_upstream["status"], "FAIL")

    def test_stage4r_cli_blocks_before_live_work_without_gate3r(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report, path = generate_writer_formal(
                ROOT / "configs" / "dataset_formal_pilot.yaml",
                ROOT / "configs" / "writer_formal.yaml",
                ROOT / "configs" / "budget_formal.yaml",
                ROOT / "configs" / "providers.yaml",
                project_root=directory,
                execute_live=True,
            )
            self.assertEqual(report["status"], "BLOCKED")
            self.assertEqual(report["blockers"][0]["reason"], "gate3r_audit_missing")
            self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
