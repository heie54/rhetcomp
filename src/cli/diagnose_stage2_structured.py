from __future__ import annotations

import argparse
from pathlib import Path
from uuid import uuid4

from src.adapters.chat.deepseek import DeepSeekChatAdapter
from src.adapters.config import load_provider_profiles
from src.adapters.environment import load_provider_environment
from src.adapters.records import ProviderCallRecorder
from src.common.jsonio import read_json, sha256_json
from src.compilers.config import load_compiler_settings
from src.compilers.experience.extract import extract_candidates
from src.compilers.experience.span_validate import validate_candidates
from src.compilers.experience.verify import verify_candidate
from src.config import load_config
from src.domain.models import SourcePaper


ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run one paid ACL extractor/verifier structured-output diagnostic"
    )
    parser.add_argument("--execute-live", action="store_true")
    parser.add_argument("--source-id", default=None)
    args = parser.parse_args(argv)
    if not args.execute_live:
        print("LIVE_REQUESTS=DISABLED")
        return 2

    dataset = load_config(ROOT / "configs" / "dataset_formal_pilot.yaml")
    paths = dataset["paths"]
    manifest = read_json(ROOT / paths["manifests"] / "acl_pilot.json")
    entries = manifest["entries"]
    entry = next(
        (item for item in entries if item["source_id"] == args.source_id),
        entries[0] if args.source_id is None else None,
    )
    if entry is None:
        raise ValueError(f"Unknown ACL source id: {args.source_id}")
    source = SourcePaper.from_dict(
        read_json(ROOT / paths["source_normalized"] / f"{entry['source_id']}.json")
    )

    settings = load_compiler_settings(ROOT / "configs" / "compiler_formal.yaml")
    profiles = load_provider_profiles(ROOT / "configs" / "providers.yaml")
    run_id = f"stage2r-structured-diagnostic-{uuid4().hex[:12]}"
    recorder = ProviderCallRecorder(
        ROOT / "artifacts" / "formal_pilot" / "diagnostics" / run_id
    )
    adapter = DeepSeekChatAdapter.from_env(
        profiles.require("deepseek_compiler_v1"),
        recorder=recorder,
        environ=load_provider_environment(ROOT),
    )
    config_hash = sha256_json(
        {
            "dataset": dataset,
            "compiler": load_config(ROOT / "configs" / "compiler_formal.yaml"),
            "providers": load_config(ROOT / "configs" / "providers.yaml"),
        }
    )
    data_manifest_hash = sha256_json({"acl": manifest["manifest_hash"]})
    extraction = extract_candidates(
        [source],
        settings,
        adapter,
        formal_mode=True,
        run_id=run_id,
        config_hash=config_hash,
        data_manifest_hash=data_manifest_hash,
    )
    validated, _ = validate_candidates(extraction.candidates, {source.source_id: source})
    exact = [item for item in validated if item.grounding_status == "span_verified"]
    verifier_status = "NOT_RUN"
    if exact:
        verifier_status = verify_candidate(
            exact[0].candidate,
            settings,
            adapter,
            formal_mode=True,
            run_id=run_id,
            config_hash=config_hash,
            data_manifest_hash=data_manifest_hash,
        ).grounding_status

    passed = bool(extraction.candidates) and bool(exact) and verifier_status in {
        "support_verified",
        "rejected",
    }
    print(f"STRUCTURED_DIAGNOSTIC={'PASS' if passed else 'FAIL'}")
    print(f"RUN_ID={run_id}")
    print(f"SOURCE_ID={source.source_id}")
    print(f"CANDIDATES={len(extraction.candidates)}")
    print(f"EXACT_CANDIDATES={len(exact)}")
    print(f"EXTRACT_FORMAT_REPAIRS={extraction.format_repair_count}")
    print(f"VERIFIER_STATUS={verifier_status}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
