from __future__ import annotations

import unittest
from pathlib import Path

from src.artifacts import ArtifactStore, artifact_hash, artifact_id
from src.config import load_config
from src.domain.models import RepresentationArtifact, SourcePaper
from src.domain.schemas import validate_schema
from src.ingest.source import normalize_source_record


ROOT = Path(__file__).resolve().parents[1]


class Stage0ContractTests(unittest.TestCase):
    def test_all_configs_are_versioned(self) -> None:
        for path in sorted((ROOT / "configs").glob("*.yaml")):
            with self.subTest(path=path.name):
                config = load_config(path)
                self.assertTrue(config["config_version"])

    def test_source_schema_round_trip(self) -> None:
        source = normalize_source_record(
            {
                "source_id": "acl2024_test",
                "title": "Test paper",
                "authors": ["A. Author"],
                "venue": "ACL 2024",
                "track": "main",
                "introduction": "First claim. Second claim!\n\nAnother paragraph?",
            }
        )
        payload = source.to_dict()
        validate_schema("SourcePaper", payload)
        self.assertEqual(SourcePaper.from_dict(payload), source)

    def test_representation_wrapper_round_trip(self) -> None:
        artifact = RepresentationArtifact(
            representation_id="rep_test",
            type="raw",
            source_corpus_hash="a" * 64,
            compiler_model="none-stage0",
            compiler_prompt_version="none-stage0",
            compiler_input_tokens=0,
            compiler_output_tokens=0,
            compiler_calls=0,
            content="test",
            content_tokens=1,
            content_hash="b" * 64,
        )
        payload = artifact.to_dict()
        validate_schema("RepresentationArtifact", payload)
        self.assertEqual(RepresentationArtifact.from_dict(payload), artifact)

    def test_artifact_hash_and_id_are_stable(self) -> None:
        left = {"b": 2, "a": 1}
        right = {"a": 1, "b": 2}
        self.assertEqual(artifact_hash(left), artifact_hash(right))
        self.assertEqual(artifact_id("fixture", left), artifact_id("fixture", right))

    def test_artifact_store_round_trip(self) -> None:
        from tempfile import TemporaryDirectory
        import json

        with TemporaryDirectory() as directory:
            path = ArtifactStore(directory).put("tests", "sample", {"value": "ok"})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"value": "ok"})


if __name__ == "__main__":
    unittest.main()
