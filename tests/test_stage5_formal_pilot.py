from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from src.adapters.config import load_provider_profiles
from src.adapters.records import ProviderCallArtifact
from src.artifacts_freeze import build_formal_freeze_manifest, validate_freeze_manifest
from src.audits.gate5_formal import build_gate5_formal_report
from src.cli.run_formal_pilot import _accepted_upstream_artifacts, run_formal_pilot
from src.common.jsonio import sha256_text
from src.writer.config import WRITER_CONDITIONS
from src.writer.order import condition_order
from src.writer.writer import GenerationArtifact


ROOT = Path(__file__).resolve().parents[1]


def _generation(target_id: str, condition: str, index: int) -> GenerationArtifact:
    return GenerationArtifact(
        generation_id=f"gen_{index:03d}",
        target_id=target_id,
        condition=condition,
        writer_model="deepseek-v4-flash",
        writer_prompt_hash=f"prompt_{index}",
        prompt_template_hash=f"template_{target_id}",
        base_prompt_hash=f"base_{target_id}",
        target_evidence_hash=f"evidence_{target_id}",
        representation_hash=None if condition == "evidence_only" else f"rep_{condition}",
        input_tokens=100,
        output_tokens=100,
        latency_ms=2,
        text="A valid target-domain Introduction [1].",
        citation_indices=(1,),
        citation_valid=True,
        provider_metadata={
            "provider": "deepseek",
            "gateway": "opencode_go",
            "requested_model": "deepseek-v4-flash",
            "returned_model": "deepseek-v4-flash",
            "provider_profile_hash": "p" * 64,
            "provider_request_id": f"req_{index}",
            "call_id": f"call_{index}",
        },
        run_mode="formal",
        run_id=f"pilot:{target_id}",
        config_hash="c" * 64,
        data_manifest_hash="d" * 64,
        provider_profile_hash="p" * 64,
    )


def _call(target_id: str, condition: str, index: int) -> ProviderCallArtifact:
    return ProviderCallArtifact(
        call_id=f"call_{index}",
        run_id=f"pilot:{target_id}",
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
        response_hash=sha256_text("A valid target-domain Introduction [1]."),
        input_tokens=100,
        output_tokens=100,
        latency_ms=2,
        system_fingerprint=None,
        provider_request_id=f"req_{index}",
        retry_count=0,
        status="success",
        run_mode="formal",
        config_hash="c" * 64,
        data_manifest_hash="d" * 64,
    )


class Stage5FormalPilotTests(unittest.TestCase):
    def test_freeze_collects_only_upstream_accepted_attempts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            accepted = root / "artifacts" / "accepted" / "result.json"
            stale = root / "artifacts" / "stale" / "result.json"
            accepted.parent.mkdir(parents=True)
            stale.parent.mkdir(parents=True)
            accepted.write_text("{}\n", encoding="utf-8")
            stale.write_text("{}\n", encoding="utf-8")
            upstream = {
                "2R": {
                    "experience_artifact_root": "artifacts/accepted",
                },
                "3R": {},
                "4R": {},
            }
            collected = _accepted_upstream_artifacts(root, upstream)
            self.assertEqual(collected, [accepted.resolve()])
            self.assertNotIn(stale.resolve(), collected)

            upstream["2R"]["experience_artifact_root"] = "../outside"
            with self.assertRaisesRegex(ValueError, "escapes project root"):
                _accepted_upstream_artifacts(root, upstream)

    def _freeze(self, directory: str, generation_ids: list[str]) -> dict:
        root = Path(directory)
        frozen = root / "frozen.py"
        frozen.write_text("VALUE = 1\n", encoding="utf-8")

        def files(category: str, names: list[str]) -> list[Path]:
            paths = []
            for name in names:
                path = root / "freeze-fixtures" / category / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"{category}:{name}\n", encoding="utf-8")
                paths.append(path)
            return paths

        artifact_files = {
            "data_manifests": files("data_manifests", [f"manifest_{i}.json" for i in range(3)]),
            "upstream_audits": files("upstream_audits", [f"gate_{i}.json" for i in range(4)]),
            "compiled_artifacts": files("compiled_artifacts", ["compiled.json"]),
            "evidence_packs": files("evidence_packs", [f"pack_{i}.json" for i in range(10)]),
            "generation_artifacts": files(
                "generation_artifacts", [f"{generation_id}.json" for generation_id in generation_ids]
            ),
            "order_manifests": files("order_manifests", [f"order_{i}.json" for i in range(10)]),
            "writer_call_artifacts": files(
                "writer_call_artifacts", [f"call_{i}.json" for i in range(50)]
            ),
            "writer_cost_artifacts": files("writer_cost_artifacts", ["writer_costs.json"]),
            "run_manifests": files("run_manifests", ["manifest.json"]),
            "evaluation_artifacts": files("evaluation_artifacts", ["review.json"]),
        }
        return build_formal_freeze_manifest(
            root=root,
            run_id="formal-pilot-v1",
            attempt_id="attempt_test",
            providers=load_provider_profiles(ROOT / "configs" / "providers.yaml"),
            acl_manifest={"manifest_hash": "a" * 64, "source_corpus_hash": "b" * 64},
            target_manifest={
                "manifest_hash": "c" * 64,
                "source_split": "train",
                "official_test_accessed": False,
            },
            tokenizer_manifest={
                "model_repo": "deepseek-ai/DeepSeek-V4-Flash",
                "resolved_revision": "revision",
                "tokenizer_hash": "d" * 64,
                "tokenizer_version": "deepseek_formal:test",
            },
            frozen_files=[frozen],
            artifact_files=artifact_files,
            generation_ids=generation_ids,
            evaluation_id="evaluation_test",
        )

    def test_freeze_manifest_is_hash_complete_and_requires_50_generations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = self._freeze(directory, [f"gen_{index}" for index in range(50)])
            self.assertTrue(validate_freeze_manifest(manifest))
            self.assertFalse(manifest["official_test_accessed"])
            self.assertEqual(manifest["embedding"]["dimensions"], 1024)
            tampered = dict(manifest)
            tampered["budget_tokens"] = {"target_evidence": 1, "writing_condition": 1}
            self.assertFalse(validate_freeze_manifest(tampered))

    def test_gate5r_accepts_only_complete_real_dev_pilot(self) -> None:
        targets = [f"ncphysics_dev_{index:02d}" for index in range(10)]
        generations = []
        calls_by_target = {}
        orders = {}
        index = 0
        for target_id in targets:
            order = condition_order(target_id, "order-seed")
            orders[target_id] = {
                "condition_order": list(order),
                "ordering_kind": "local_seeded_shuffle",
                "run_mode": "formal",
                "config_hash": "c" * 64,
                "data_manifest_hash": "d" * 64,
            }
            calls = []
            for condition in order:
                generations.append(_generation(target_id, condition, index))
                calls.append(_call(target_id, condition, index))
                index += 1
            calls_by_target[target_id] = calls
        review = {
            "passed": True,
            "checks": {
                "gold_leakage": {"passed": True},
                "source_domain_leakage_in_generations": {"passed": True},
                "guideline_vs_experience_distinct": {"passed": True},
                "experience_strategies_actionable": {"passed": True},
                "token_compute_matching": {"passed": True},
                "output_format_and_citations": {"passed": True},
            },
        }
        upstream = {
            "1R": {"status": "PASS", "data_manifest_hash": "d" * 64},
            "2R": {
                "status": "PASS",
                "data_manifest_hash": "d" * 64,
                "summary": {
                    "format_repair_count": 0,
                    "candidate_count": 20,
                    "support_verified_count": 12,
                    "retrieved_pair_count": 20,
                    "adjudicated_pair_count": 20,
                },
            },
            "3R": {
                "status": "PASS",
                "data_manifest_hash": "d" * 64,
                "checks": {"guideline_experience_compute_envelope": {"passed": True}},
            },
            "4R": {"status": "PASS", "data_manifest_hash": "d" * 64},
        }
        target_manifest = {
            "dataset": "Xiao-Youth/NC_Physics",
            "source_split": "train",
            "selected_count": 10,
            "official_test_accessed": False,
        }
        with tempfile.TemporaryDirectory() as directory:
            freeze = self._freeze(
                directory, [generation.generation_id for generation in generations]
            )
            report = build_gate5_formal_report(
                review,
                generations,
                calls_by_target,
                orders,
                upstream_reports=upstream,
                target_manifest=target_manifest,
                freeze_manifest=freeze,
            )
            stale_upstream = {name: dict(value) for name, value in upstream.items()}
            stale_upstream["2R"]["data_manifest_hash"] = "f" * 64
            stale = build_gate5_formal_report(
                review,
                generations,
                calls_by_target,
                orders,
                upstream_reports=stale_upstream,
                target_manifest=target_manifest,
                freeze_manifest=freeze,
            )
        self.assertEqual(report["status"], "PASS", report["checks"])
        self.assertTrue(report["checks"]["formal_metadata_chain"]["passed"])
        self.assertTrue(report["checks"]["generation_call_traceability"]["passed"])
        self.assertEqual(report["config_hash"], "c" * 64)
        self.assertEqual(report["freeze_status"], "formal_pilot_frozen")
        self.assertFalse(
            report["checks"]["no_condition_ranking_or_final_statistics"][
                "condition_ranking_computed"
            ]
        )
        self.assertFalse(stale["checks"]["all_upstream_formal_gates_passed"]["passed"])
        self.assertEqual(stale["status"], "FAIL")

        first_target = targets[0]
        mismatched_calls = dict(calls_by_target)
        mismatched_calls[first_target] = [
            replace(calls_by_target[first_target][0], response_hash="f" * 64),
            *calls_by_target[first_target][1:],
        ]
        mismatched = build_gate5_formal_report(
            review,
            generations,
            mismatched_calls,
            orders,
            upstream_reports=upstream,
            target_manifest=target_manifest,
            freeze_manifest=freeze,
        )
        self.assertFalse(
            mismatched["checks"]["generation_call_traceability"]["passed"]
        )
        self.assertEqual(mismatched["status"], "FAIL")

    def test_official_test_access_prevents_freeze(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            freeze = self._freeze(directory, [f"gen_{index}" for index in range(50)])
            freeze["official_test_accessed"] = True
            self.assertFalse(validate_freeze_manifest(freeze))

    def test_stage5r_cli_blocks_before_live_work_without_upstream_gates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report, audit_path, freeze_path = run_formal_pilot(
                ROOT / "configs" / "dataset_formal_pilot.yaml",
                ROOT / "configs" / "compiler_formal.yaml",
                ROOT / "configs" / "budget_formal.yaml",
                ROOT / "configs" / "writer_formal.yaml",
                ROOT / "configs" / "providers.yaml",
                project_root=directory,
                execute_live=True,
            )
            self.assertEqual(report["status"], "BLOCKED")
            self.assertEqual(
                report["blockers"][0]["reason"], "upstream_formal_audits_missing"
            )
            self.assertTrue(audit_path.exists())
            self.assertIsNone(freeze_path)


if __name__ == "__main__":
    unittest.main()
