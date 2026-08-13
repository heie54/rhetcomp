from __future__ import annotations

from typing import Any

from src.artifacts import artifact_id
from src.common.jsonio import sha256_text
from src.domain.models import RepresentationArtifact


def build_representation(
    representation_type: str,
    content: str,
    content_tokens: int,
    source_corpus_hash: str,
    compiler_model: str,
    compiler_prompt_version: str,
    compiler_input_tokens: int,
    compiler_output_tokens: int,
    compiler_calls: int,
    *,
    run_id: str = "unscoped",
    run_mode: str = "mechanics",
    config_hash: str | None = None,
    data_manifest_hash: str | None = None,
    provider_profile_hash: str | None = None,
) -> RepresentationArtifact:
    """Build the common representation wrapper (spec §9)."""
    content_hash = sha256_text(content)
    representation_id = artifact_id(
        f"rep_{representation_type}",
        {
            "type": representation_type,
            "source_corpus_hash": source_corpus_hash,
            "compiler_prompt_version": compiler_prompt_version,
            "content_hash": content_hash,
        },
    )
    return RepresentationArtifact(
        representation_id=representation_id,
        type=representation_type,
        source_corpus_hash=source_corpus_hash,
        compiler_model=compiler_model,
        compiler_prompt_version=compiler_prompt_version,
        compiler_input_tokens=compiler_input_tokens,
        compiler_output_tokens=compiler_output_tokens,
        compiler_calls=compiler_calls,
        content=content,
        content_tokens=content_tokens,
        content_hash=content_hash,
        run_id=run_id,
        run_mode=run_mode,
        config_hash=config_hash,
        data_manifest_hash=data_manifest_hash,
        provider_profile_hash=provider_profile_hash,
    )


def source_reading_tokens(sources: list[Any], tokenizer: Any) -> int:
    """Compiler-side source-reading compute used to compare Guideline vs Experience (spec §2.7)."""
    return sum(
        len(tokenizer.encode(source.introduction.normalized_text)) for source in sources
    )
