from __future__ import annotations

import argparse
from pathlib import Path

from src.budget.formal_tokenizer import load_formal_tokenizer
from src.common.jsonio import write_json
from src.config import load_config


ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Provision the pinned DeepSeek formal tokenizer")
    parser.add_argument("--config", default="configs/budget_formal.yaml")
    parser.add_argument(
        "--download",
        action="store_true",
        help="explicitly download tokenizer-only assets from the official Hugging Face model repo",
    )
    args = parser.parse_args(argv)
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    tokenizer_config = config["tokenizer"]
    local_path = Path(tokenizer_config["local_path"])
    asset_root = local_path if local_path.is_absolute() else ROOT / local_path
    if not asset_root.exists() and not args.download:
        print("FORMAL_TOKENIZER=BLOCKED")
        print(f"MISSING_ASSETS={asset_root}")
        print(
            "RESUME_COMMAND=D:\\AnacondaData\\envs_dirs\\py3.13\\python.exe "
            "-m src.cli.prepare_formal_tokenizer --download"
        )
        return 2
    if args.download:
        from huggingface_hub import HfApi, snapshot_download

        asset_root.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=str(tokenizer_config["model_repo"]),
            revision=str(tokenizer_config["revision"]),
            local_dir=asset_root,
            allow_patterns=[
                "config.json",
                "tokenizer*",
                "*tokenizer*",
                "vocab*",
                "merges*",
                "special_tokens*",
                "added_tokens*",
                "tokenization*",
            ],
        )
        resolved_revision = HfApi().model_info(
            str(tokenizer_config["model_repo"]), revision=str(tokenizer_config["revision"])
        ).sha
    else:
        resolved_revision = str(tokenizer_config["revision"])
    tokenizer = load_formal_tokenizer(config_path, ROOT)
    manifest_path = ROOT / "data" / "manifests" / "deepseek_formal_tokenizer.json"
    write_json(
        manifest_path,
        {
            "manifest_version": "deepseek-formal-tokenizer-1",
            "run_mode": "formal",
            "model_repo": tokenizer.model_repo,
            "configured_revision": tokenizer.revision,
            "resolved_revision": resolved_revision,
            "tokenizer_hash": tokenizer.asset_hash,
            "tokenizer_version": tokenizer.version,
        },
    )
    print("FORMAL_TOKENIZER=PASS")
    print(f"TOKENIZER_HASH={tokenizer.asset_hash}")
    print(f"MANIFEST={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
