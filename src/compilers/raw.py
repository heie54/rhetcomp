from __future__ import annotations

from typing import Iterable

from src.budget.selection import atomic_budget
from src.budget.tokenizer import Tokenizer
from src.compilers.base import build_representation, source_reading_tokens
from src.compilers.config import CompilerSettings
from src.compilers.corpus import source_corpus_hash
from src.domain.models import RepresentationArtifact, SourcePaper

RAW_SELECTION_VERSION = "stage3-raw-1"


def compile_raw(
    sources: Iterable[SourcePaper],
    settings: CompilerSettings,
    tokenizer: Tokenizer,
    writing_condition_tokens: int,
) -> RepresentationArtifact:
    """B1 — Raw Source Exemplars: token-budgeted raw ACL/NLP exemplar Introduction text.

    Deterministic selection: whole normalized Introductions, source_id ascending, kept
    atomically within the writing-condition budget. No model call.
    """
    ordered = sorted(sources, key=lambda source: source.source_id)
    blocks = [
        f"[{source.source_id}] {source.introduction.normalized_text}"
        for source in ordered
    ]
    budget = atomic_budget(blocks, tokenizer, writing_condition_tokens)
    return build_representation(
        representation_type="raw",
        content=budget.content,
        content_tokens=budget.post_truncation_tokens,
        source_corpus_hash=source_corpus_hash(sources),
        compiler_model=settings.model or "deterministic-mock-1",
        compiler_prompt_version=RAW_SELECTION_VERSION,
        compiler_input_tokens=source_reading_tokens(ordered, tokenizer),
        compiler_output_tokens=budget.post_truncation_tokens,
        compiler_calls=0,
    )
