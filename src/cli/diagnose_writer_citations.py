from __future__ import annotations

import argparse
import re
from pathlib import Path
from uuid import uuid4

from src.adapters.chat.deepseek import DeepSeekChatAdapter
from src.adapters.config import load_provider_profiles
from src.adapters.environment import load_provider_environment
from src.adapters.records import ProviderCallRecorder
from src.budget.formal_tokenizer import load_formal_tokenizer
from src.common.jsonio import read_json
from src.config import load_config
from src.domain.models import RepresentationArtifact, TargetEvidencePack
from src.formal_metadata import build_formal_metadata
from src.writer.config import load_writer_settings
from src.writer.writer import Writer


ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run one paid Writer citation diagnostic on a real dev target"
    )
    parser.add_argument("--execute-live", action="store_true")
    parser.add_argument("--target-id", required=True)
    parser.add_argument(
        "--condition", choices=("raw", "summary", "guideline", "experience"), default="raw"
    )
    args = parser.parse_args(argv)
    if not args.execute_live:
        print("LIVE_REQUESTS=DISABLED")
        return 2

    dataset = load_config(ROOT / "configs" / "dataset_formal_pilot.yaml")
    budget = load_config(ROOT / "configs" / "budget_formal.yaml")
    writer_config = load_config(ROOT / "configs" / "writer_formal.yaml")
    provider_config = load_config(ROOT / "configs" / "providers.yaml")
    paths = dataset["paths"]
    gate3 = read_json(ROOT / paths["audits"] / "gate3_formal.json")
    representation = RepresentationArtifact.from_dict(
        read_json(ROOT / gate3["representation_artifact_root"] / f"{args.condition}.json")
    )
    pack = TargetEvidencePack.from_dict(
        read_json(ROOT / paths["evidence_packs"] / f"{args.target_id}.json")
    )
    acl_manifest = read_json(ROOT / paths["manifests"] / "acl_pilot.json")
    target_manifest = read_json(ROOT / paths["manifests"] / "nc_physics_pilot.json")
    profiles = load_provider_profiles(ROOT / "configs" / "providers.yaml")
    metadata = build_formal_metadata(
        str(dataset["run_id"]),
        configs={
            "dataset": dataset,
            "writer": writer_config,
            "budget": budget,
            "providers": provider_config,
        },
        manifests={
            "acl": acl_manifest["manifest_hash"],
            "targets": target_manifest["manifest_hash"],
        },
    )
    run_id = f"writer-citation-diagnostic-{uuid4().hex[:12]}"
    recorder = ProviderCallRecorder(
        ROOT / "artifacts" / "formal_pilot" / "diagnostics" / run_id
    )
    profile = profiles.require("deepseek_writer_v1")
    adapter = DeepSeekChatAdapter.from_env(
        profile,
        recorder=recorder,
        environ=load_provider_environment(ROOT),
    )
    writer = Writer(
        load_writer_settings(ROOT / "configs" / "writer_formal.yaml"),
        load_formal_tokenizer(ROOT / "configs" / "budget_formal.yaml", ROOT),
        adapter,
    )
    generation = writer.generate(
        pack,
        args.condition,
        representation,
        formal_mode=True,
        run_id=run_id,
        config_hash=metadata.config_hash,
        data_manifest_hash=metadata.data_manifest_hash,
        provider_profile_hash=profile.profile_hash,
    )
    bracket_count = len(re.findall(r"\[(.*?)\]", generation.text))
    print(f"WRITER_CITATION_DIAGNOSTIC={'PASS' if generation.citation_valid else 'FAIL'}")
    print(f"RUN_ID={run_id}")
    print(f"TARGET_ID={args.target_id}")
    print(f"CONDITION={args.condition}")
    print(f"BRACKET_COUNT={bracket_count}")
    print(f"CITATION_INDEX_COUNT={len(generation.citation_indices)}")
    return 0 if generation.citation_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
