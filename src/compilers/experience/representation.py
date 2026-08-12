from __future__ import annotations

from typing import Iterable

from src.adapters.model import ModelAdapter
from src.budget.tokenizer import Tokenizer
from src.compilers.base import build_representation, source_reading_tokens
from src.compilers.config import CompilerSettings
from src.compilers.experience.pipeline import ExperienceLibraryResult, compile_experience_library
from src.domain.models import RepresentationArtifact, SourcePaper


def experience_prompt_version(settings: CompilerSettings) -> str:
    return "|".join(
        (
            settings.extraction_prompt_version,
            settings.verifier_prompt_version,
            settings.adjudication_prompt_version,
        )
    )


def compile_experience_representation(
    sources: Iterable[SourcePaper],
    settings: CompilerSettings,
    tokenizer: Tokenizer,
    writing_condition_tokens: int,
    adapter: ModelAdapter | None = None,
) -> tuple[RepresentationArtifact, ExperienceLibraryResult]:
    """Ours — Provenance-Grounded Writing Experience (spec §5, §8-9)."""
    result = compile_experience_library(
        list(sources), settings, tokenizer, writing_condition_tokens, adapter
    )
    compiler_model = settings.model or "deterministic-mock-1"
    if result.compiler_calls == 0:
        # Deterministic mechanics mode: report the equivalent source-reading compute so
        # Guideline and Experience costs remain directly comparable (spec §2.7).
        compiler_input_tokens = source_reading_tokens(list(sources), tokenizer)
        compiler_output_tokens = result.library.content_tokens
        compiler_calls = 0
    else:
        compiler_input_tokens = result.compiler_input_tokens
        compiler_output_tokens = result.compiler_output_tokens
        compiler_calls = result.compiler_calls
    artifact = build_representation(
        representation_type="experience",
        content=result.library.content,
        content_tokens=result.library.content_tokens,
        source_corpus_hash=result.source_corpus_hash,
        compiler_model=compiler_model,
        compiler_prompt_version=experience_prompt_version(settings),
        compiler_input_tokens=compiler_input_tokens,
        compiler_output_tokens=compiler_output_tokens,
        compiler_calls=compiler_calls,
    )
    return artifact, result
