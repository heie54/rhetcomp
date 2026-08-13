from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any, Sequence

from src.config import load_config


TOKENIZER_ASSET_PATTERNS = (
    "tokenizer",
    "vocab",
    "merges",
    "special_tokens",
    "added_tokens",
    "tokenization",
)


def tokenizer_asset_hash(root: Path) -> str:
    files = [
        path
        for path in root.rglob("*")
        if path.is_file() and any(pattern in path.name.lower() for pattern in TOKENIZER_ASSET_PATTERNS)
    ]
    if not files:
        raise ValueError(f"No tokenizer assets found under {root}")
    digest = sha256()
    for path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


class DeepSeekFormalTokenizer:
    def __init__(
        self,
        backend: Any,
        *,
        model_repo: str,
        revision: str,
        asset_hash: str,
    ) -> None:
        if not model_repo or not revision or len(asset_hash) != 64:
            raise ValueError("Formal tokenizer requires model repo, revision, and SHA-256 asset hash")
        self._backend = backend
        self.model_repo = model_repo
        self.revision = revision
        self.asset_hash = asset_hash

    @classmethod
    def from_local_assets(
        cls, asset_root: str | Path, *, model_repo: str, revision: str
    ) -> "DeepSeekFormalTokenizer":
        root = Path(asset_root).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Formal tokenizer assets not found: {root}")
        from transformers import AutoTokenizer

        backend = AutoTokenizer.from_pretrained(
            str(root), local_files_only=True, trust_remote_code=False
        )
        return cls(
            backend,
            model_repo=model_repo,
            revision=revision,
            asset_hash=tokenizer_asset_hash(root),
        )

    @property
    def version(self) -> str:
        return f"deepseek_formal:{self.model_repo}@{self.revision}:{self.asset_hash}"

    def encode(self, text: str) -> Sequence[int]:
        return self._backend.encode(text, add_special_tokens=False)

    def decode(self, tokens: Sequence[int]) -> str:
        return self._backend.decode(list(tokens), skip_special_tokens=False)


def load_formal_tokenizer(
    config_path: str | Path, project_root: str | Path | None = None
) -> DeepSeekFormalTokenizer:
    path = Path(config_path).resolve()
    root = Path(project_root).resolve() if project_root else path.parent.parent
    config = load_config(path)
    if config.get("run_mode") != "formal":
        raise ValueError("Formal tokenizer requires a formal budget config")
    tokenizer = config.get("tokenizer")
    if not isinstance(tokenizer, dict):
        raise ValueError("Formal budget config requires tokenizer settings")
    local_path = Path(str(tokenizer["local_path"]))
    asset_root = local_path if local_path.is_absolute() else root / local_path
    return DeepSeekFormalTokenizer.from_local_assets(
        asset_root,
        model_repo=str(tokenizer["model_repo"]),
        revision=str(tokenizer["revision"]),
    )
