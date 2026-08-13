from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Mapping

from src.adapters.chat.deepseek import DeepSeekChatAdapter
from src.adapters.config import ProviderProfile, load_provider_profiles
from src.adapters.embedding.base import EmbeddingRequest
from src.adapters.embedding.qwen import EmbeddingCache, QwenEmbeddingAdapter
from src.adapters.environment import load_provider_environment
from src.adapters.http import HttpResponse, ProviderTransportError
from src.adapters.mock import MockChatAdapter, MockEmbeddingAdapter
from src.adapters.model import ModelRequest
from src.adapters.records import ProviderCallArtifact, ProviderCallRecorder
from src.audits.gate0_formal import build_gate0_formal_report
from src.cli.check_providers import main as check_providers_main


ROOT = Path(__file__).resolve().parents[1]


class FakeTransport:
    def __init__(self, responses: list[HttpResponse | Exception]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, Mapping[str, str], Mapping[str, Any], float]] = []

    def __call__(
        self,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout: float,
    ) -> HttpResponse:
        self.calls.append((url, headers, payload, timeout))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def chat_response(content: str = '{"ok":true}') -> HttpResponse:
    return HttpResponse(
        payload={
            "id": "chatcmpl-test",
            "model": "deepseek-v4-flash",
            "system_fingerprint": "fp_test",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15},
        },
        headers={"x-request-id": "request-header-test"},
    )


def formal_chat_request() -> ModelRequest:
    return ModelRequest(
        "formal system",
        "formal input",
        32,
        run_id="formal-run",
        role="experience_verifier",
        run_mode="formal",
        config_hash="c" * 64,
        data_manifest_hash="d" * 64,
    )


class Stage0FormalProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        profiles = load_provider_profiles(ROOT / "configs" / "providers.yaml")
        self.compiler_profile = profiles.require("deepseek_compiler_v1")
        self.writer_profile = profiles.require("deepseek_writer_v1")
        self.embedding_profile = profiles.require("qwen_embedding_v1")

    def test_frozen_provider_profiles(self) -> None:
        self.assertEqual(self.compiler_profile.model, "deepseek-v4-flash")
        self.assertTrue(self.compiler_profile.thinking_enabled)
        self.assertEqual(self.compiler_profile.reasoning_effort, "high")
        self.assertIsNone(self.compiler_profile.temperature)
        self.assertEqual(self.writer_profile.temperature, 0.0)
        self.assertFalse(self.writer_profile.thinking_enabled)
        self.assertEqual(self.compiler_profile.gateway, "opencode_go")
        self.assertEqual(self.compiler_profile.user_agent, "rhetcomp/0.1.0")
        self.assertEqual(self.embedding_profile.model, "qwen3.7-text-embedding")
        self.assertEqual(self.embedding_profile.max_batch_size, 20)
        self.assertEqual(self.embedding_profile.dimensions, 1024)

    def test_thinking_request_omits_sampling_parameters(self) -> None:
        transport = FakeTransport([chat_response()])
        adapter = DeepSeekChatAdapter(
            self.compiler_profile,
            "secret",
            "https://deepseek.invalid/v1",
            transport=transport,
            sleep=lambda _: None,
        )
        adapter.generate(
            ModelRequest(
                system_prompt="Return JSON.",
                user_prompt="Test.",
                max_output_tokens=32,
                response_format="json_object",
                run_id="test",
                role="extractor",
            )
        )
        payload = transport.calls[0][2]
        self.assertEqual(payload["thinking"], {"type": "enabled"})
        self.assertEqual(payload["reasoning_effort"], "high")
        self.assertNotIn("temperature", payload)
        self.assertNotIn("top_p", payload)
        self.assertNotIn("seed", payload)
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertFalse(payload["stream"])
        headers = transport.calls[0][1]
        self.assertEqual(headers["User-Agent"], "rhetcomp/0.1.0")

    def test_thinking_request_rejects_sampling_parameters(self) -> None:
        adapter = DeepSeekChatAdapter(
            self.compiler_profile,
            "secret",
            "https://deepseek.invalid/v1",
            transport=FakeTransport([chat_response()]),
        )
        with self.assertRaisesRegex(ValueError, "must omit"):
            adapter.generate(
                ModelRequest(
                    system_prompt="s",
                    user_prompt="u",
                    max_output_tokens=10,
                    temperature=0.0,
                )
            )

    def test_writer_request_is_non_thinking_and_deterministic(self) -> None:
        transport = FakeTransport([chat_response("writer")])
        adapter = DeepSeekChatAdapter(
            self.writer_profile,
            "secret",
            "https://deepseek.invalid/v1",
            transport=transport,
        )
        adapter.generate(ModelRequest("s", "u", 10))
        payload = transport.calls[0][2]
        self.assertEqual(payload["thinking"], {"type": "disabled"})
        self.assertEqual(payload["temperature"], 0.0)
        self.assertNotIn("reasoning_effort", payload)
        self.assertNotIn("top_p", payload)
        self.assertNotIn("seed", payload)

    def test_deepseek_response_metadata_and_call_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            adapter = DeepSeekChatAdapter(
                self.compiler_profile,
                "super-secret-key",
                "https://deepseek.invalid/v1",
                recorder=ProviderCallRecorder(directory),
                transport=FakeTransport([chat_response()]),
            )
            response = adapter.generate(
                ModelRequest("system secret-free", "input text", 32, run_id="run1", role="verifier")
            )
            self.assertEqual(response.input_tokens, 12)
            self.assertEqual(response.output_tokens, 3)
            self.assertEqual(response.metadata["returned_model"], "deepseek-v4-flash")
            self.assertEqual(response.metadata["provider_request_id"], "request-header-test")
            artifacts = list((Path(directory) / "calls").glob("*.json"))
            self.assertEqual(len(artifacts), 1)
            raw = artifacts[0].read_text(encoding="utf-8")
            self.assertNotIn("super-secret-key", raw)
            self.assertNotIn("input text", raw)
            artifact = ProviderCallArtifact.from_dict(json.loads(raw))
            self.assertEqual(artifact.status, "success")
            self.assertEqual(artifact.role, "verifier")
            self.assertEqual(artifact.system_fingerprint, "fp_test")
            self.assertEqual(artifact.gateway, "opencode_go")

    def test_formal_deepseek_requires_exact_returned_model_and_request_id(self) -> None:
        baseline = chat_response()
        cases = (
            (
                HttpResponse(
                    payload={key: value for key, value in baseline.payload.items() if key != "model"},
                    headers=baseline.headers,
                ),
                "explicit returned model",
            ),
            (
                HttpResponse(
                    payload={**baseline.payload, "model": "unexpected-model"},
                    headers=baseline.headers,
                ),
                "returned model mismatch",
            ),
            (
                HttpResponse(
                    payload={key: value for key, value in baseline.payload.items() if key != "id"},
                    headers={},
                ),
                "provider request ID",
            ),
        )
        for response, message in cases:
            with self.subTest(message=message):
                adapter = DeepSeekChatAdapter(
                    self.compiler_profile,
                    "secret",
                    "https://deepseek.invalid/v1",
                    transport=FakeTransport([response]),
                )
                with self.assertRaisesRegex(ValueError, message):
                    adapter.generate(formal_chat_request())

    def test_retry_reuses_identical_request_and_records_count(self) -> None:
        transport = FakeTransport(
            [
                ProviderTransportError("rate limited", 429),
                ProviderTransportError("unavailable", 503),
                chat_response(),
            ]
        )
        adapter = DeepSeekChatAdapter(
            self.compiler_profile,
            "secret",
            "https://deepseek.invalid/v1",
            transport=transport,
            sleep=lambda _: None,
        )
        response = adapter.generate(ModelRequest("s", "u", 10))
        self.assertEqual(response.metadata["retry_count"], 2)
        self.assertEqual(transport.calls[0][2], transport.calls[1][2])
        self.assertEqual(transport.calls[1][2], transport.calls[2][2])

    def test_nonretryable_transport_error_is_not_retried_and_is_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            transport = FakeTransport([ProviderTransportError("bad request", 400)])
            adapter = DeepSeekChatAdapter(
                self.compiler_profile,
                "secret",
                "https://deepseek.invalid/v1",
                recorder=ProviderCallRecorder(directory),
                transport=transport,
                sleep=lambda _: None,
            )
            with self.assertRaises(ProviderTransportError):
                adapter.generate(ModelRequest("s", "u", 10))
            self.assertEqual(len(transport.calls), 1)
            payload = json.loads(next((Path(directory) / "calls").glob("*.json")).read_text())
            self.assertEqual(payload["status"], "failed")
            self.assertEqual(payload["retry_count"], 0)

    def test_environment_loading_requires_all_named_values(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "RHETCOMP_DEEPSEEK_BASE_URL"):
            DeepSeekChatAdapter.from_env(
                self.compiler_profile,
                environ={"RHETCOMP_DEEPSEEK_API_KEY": "secret"},
            )

    def test_qwen_embedding_dimension_cache_and_artifact(self) -> None:
        text = "strategy plus applicable when"
        vector = [float(index) for index in range(1024)]
        transport = FakeTransport(
            [
                HttpResponse(
                    payload={
                        "id": "emb-test",
                        "model": "qwen3.7-text-embedding",
                        "data": [{"index": 0, "embedding": vector}],
                        "usage": {"prompt_tokens": 7, "total_tokens": 7},
                    },
                    headers={"x-request-id": "emb-header"},
                )
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = QwenEmbeddingAdapter(
                self.embedding_profile,
                "qwen-secret",
                "https://qwen.invalid/compatible-mode/v1",
                cache=EmbeddingCache(root / "cache"),
                recorder=ProviderCallRecorder(root / "run"),
                transport=transport,
            )
            request = EmbeddingRequest(
                inputs=(text, text),
                dimensions=1024,
                run_id="run-embed",
                role="embedding",
                run_mode="formal",
                config_hash="c" * 64,
                data_manifest_hash="d" * 64,
            )
            first = adapter.embed(request)
            second = adapter.embed(request)
            self.assertEqual(len(transport.calls), 1)
            self.assertEqual(len(first.vectors), 2)
            self.assertEqual(len(first.vectors[0]), 1024)
            self.assertEqual(first.input_tokens, 7)
            self.assertEqual(second.input_tokens, 0)
            self.assertEqual(second.metadata["cache_misses"], 0)
            artifacts = [
                ProviderCallArtifact.from_dict(json.loads(path.read_text(encoding="utf-8")))
                for path in (root / "run" / "calls").glob("*.json")
            ]
            self.assertEqual(len(artifacts), 2)
            network = next(item for item in artifacts if item.execution_kind == "network")
            cached = next(item for item in artifacts if item.execution_kind == "cache")
            for path in (root / "run" / "calls").glob("*.json"):
                raw = path.read_text(encoding="utf-8")
                self.assertNotIn(text, raw)
                self.assertNotIn("qwen-secret", raw)
            self.assertEqual(network.provider, "qwen")
            self.assertEqual(network.output_tokens, 0)
            self.assertEqual(cached.cache_origin_call_ids, (network.call_id,))
            self.assertEqual(cached.cache_origin_provider_request_ids, ("emb-header",))
            self.assertTrue(second.metadata["cache_provenance_complete"])

    def test_qwen_rejects_wrong_dimension_before_network(self) -> None:
        transport = FakeTransport([])
        with tempfile.TemporaryDirectory() as directory:
            adapter = QwenEmbeddingAdapter(
                self.embedding_profile,
                "secret",
                "https://qwen.invalid/v1",
                cache=EmbeddingCache(directory),
                transport=transport,
            )
            with self.assertRaisesRegex(ValueError, "must match profile"):
                adapter.embed(EmbeddingRequest(inputs=("x",), dimensions=256))
            self.assertEqual(transport.calls, [])

    def test_formal_qwen_requires_exact_returned_model(self) -> None:
        response = HttpResponse(
            payload={
                "id": "emb-model-mismatch",
                "model": "unexpected-embedding-model",
                "data": [{"index": 0, "embedding": [0.0] * 1024}],
                "usage": {"prompt_tokens": 1},
            },
            headers={"x-request-id": "emb-model-mismatch"},
        )
        with tempfile.TemporaryDirectory() as directory:
            adapter = QwenEmbeddingAdapter(
                self.embedding_profile,
                "secret",
                "https://qwen.invalid/v1",
                cache=EmbeddingCache(Path(directory) / "cache"),
                transport=FakeTransport([response]),
            )
            with self.assertRaisesRegex(ValueError, "returned model mismatch"):
                adapter.embed(
                    EmbeddingRequest(
                        inputs=("strategy",),
                        dimensions=1024,
                        run_id="formal-run",
                        role="experience_candidate_retrieval",
                        run_mode="formal",
                        config_hash="c" * 64,
                        data_manifest_hash="d" * 64,
                    )
                )

    def test_qwen_batches_more_than_provider_limit(self) -> None:
        vector = [0.0] * 1024

        def response(size: int, request_id: str) -> HttpResponse:
            return HttpResponse(
                payload={
                    "id": request_id,
                    "model": "qwen3.7-text-embedding",
                    "data": [
                        {"index": index, "embedding": vector} for index in range(size)
                    ],
                    "usage": {"prompt_tokens": size, "total_tokens": size},
                },
                headers={"x-request-id": request_id},
            )

        transport = FakeTransport([response(20, "batch-1"), response(1, "batch-2")])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = QwenEmbeddingAdapter(
                self.embedding_profile,
                "secret",
                "https://qwen.invalid/compatible-mode/v1",
                cache=EmbeddingCache(root / "cache"),
                recorder=ProviderCallRecorder(root / "run"),
                transport=transport,
            )
            result = adapter.embed(
                EmbeddingRequest(
                    inputs=tuple(f"text-{index}" for index in range(21)),
                    dimensions=1024,
                    run_id="batch-test",
                    role="embedding",
                    run_mode="formal",
                    config_hash="c" * 64,
                    data_manifest_hash="d" * 64,
                )
            )
            self.assertEqual([len(call[2]["input"]) for call in transport.calls], [20, 1])
            self.assertEqual(len(result.vectors), 21)
            self.assertEqual(result.input_tokens, 21)
            self.assertEqual(result.metadata["network_batch_count"], 2)
            self.assertEqual(result.metadata["provider_request_ids"], ("batch-1", "batch-2"))
            self.assertEqual(len(list((root / "run" / "calls").glob("*.json"))), 2)

    def test_local_env_is_loaded_without_overriding_process_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text(
                "RHETCOMP_DEEPSEEK_API_KEY=file-key\n"
                "RHETCOMP_DEEPSEEK_BASE_URL='https://file.invalid/v1'\n",
                encoding="utf-8",
            )
            values = load_provider_environment(
                directory,
                environ={"RHETCOMP_DEEPSEEK_API_KEY": "process-key"},
            )
            self.assertEqual(values["RHETCOMP_DEEPSEEK_API_KEY"], "process-key")
            self.assertEqual(
                values["RHETCOMP_DEEPSEEK_BASE_URL"], "https://file.invalid/v1"
            )

    def test_embedding_cache_key_covers_model_dimension_and_text_hash(self) -> None:
        one = EmbeddingCache.key("m1", 1024, "text")
        self.assertNotEqual(one, EmbeddingCache.key("m2", 1024, "text"))
        self.assertNotEqual(one, EmbeddingCache.key("m1", 256, "text"))
        self.assertNotEqual(one, EmbeddingCache.key("m1", 1024, "other"))

    def test_mock_adapters_share_upper_contracts(self) -> None:
        chat = MockChatAdapter(response_text='{"ok":true}')
        chat_response_value = chat.generate(ModelRequest("s", "u", 10))
        self.assertEqual(chat_response_value.text, '{"ok":true}')
        embedding = MockEmbeddingAdapter()
        embedding_response = embedding.embed(EmbeddingRequest(inputs=("x",), dimensions=8))
        self.assertEqual(len(embedding_response.vectors[0]), 8)
        self.assertFalse(embedding_response.metadata["formal"])

    def test_provider_call_artifact_round_trip(self) -> None:
        report = build_gate0_formal_report(ROOT)
        self.assertEqual(report["status"], "PASS")
        round_trip = next(
            check for check in report["checks"] if check["name"] == "call_artifact_round_trip"
        )
        artifact = ProviderCallArtifact.from_dict(round_trip["detail"])
        self.assertEqual(artifact.to_dict(), round_trip["detail"])

    def test_provider_smoke_cli_is_network_free_without_live_flag(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = check_providers_main([])
        self.assertEqual(exit_code, 0)
        self.assertIn("PROVIDER_CONFIG=PASS", output.getvalue())
        self.assertIn("LIVE_REQUESTS=DISABLED", output.getvalue())

    def test_profile_rejects_sampling_for_thinking_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "must omit"):
            ProviderProfile(
                profile_id="bad",
                provider="deepseek",
                model="deepseek-v4-flash",
                protocol="openai_chat_completions",
                thinking_enabled=True,
                reasoning_effort="high",
                temperature=0.0,
            )


if __name__ == "__main__":
    unittest.main()
