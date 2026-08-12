from __future__ import annotations

from typing import Iterable

from src.adapters.counting import CountingAdapter
from src.adapters.model import ModelAdapter, ModelRequest
from src.budget.selection import atomic_budget
from src.budget.tokenizer import BudgetController, Tokenizer
from src.compilers.base import build_representation, source_reading_tokens
from src.compilers.config import CompilerSettings
from src.compilers.corpus import source_corpus_hash
from src.domain.models import RepresentationArtifact, SourcePaper


_SUMMARY_SYSTEM = (
    "You are a scientific-text compressor. Produce a concise source-corpus summary. "
    "The objective is source information compression, not writing-strategy induction."
)


def _corpus_text(sources: Iterable[SourcePaper]) -> str:
    ordered = sorted(sources, key=lambda source: source.source_id)
    return "\n\n".join(
        f"[{source.source_id}]\n{source.introduction.normalized_text}"
        for source in ordered
    )


def _summary_user(sources: Iterable[SourcePaper], budget_tokens: int) -> str:
    return (
        "[TASK: matched abstractive summary]\n"
        "Source corpus:\n{corpus}\n\n"
        "Produce a compressed summary of this source corpus that preserves its informative "
        "content in at most {budget} tokens. Do not induce writing strategies."
    ).format(corpus=_corpus_text(sources), budget=budget_tokens)


def _deterministic_summary(sources: Iterable[SourcePaper], tokenizer: Tokenizer, limit: int) -> str:
    """Extractive lead summary: first sentence of each source's Introduction, source_id order."""
    ordered = sorted(sources, key=lambda source: source.source_id)
    blocks = [
        f"[{source.source_id}] {source.introduction.paragraphs[0].sentences[0].text}"
        for source in ordered
    ]
    return atomic_budget(blocks, tokenizer, limit).content


def compile_summary(
    sources: Iterable[SourcePaper],
    settings: CompilerSettings,
    tokenizer: Tokenizer,
    writing_condition_tokens: int,
    adapter: ModelAdapter | None = None,
) -> RepresentationArtifact:
    """B2 — Matched Abstractive Summary (spec §5): source information compression."""
    ordered = list(sources)
    corpus_hash = source_corpus_hash(sources)
    controller = BudgetController(tokenizer)

    if adapter is None:
        content = _deterministic_summary(ordered, tokenizer, writing_condition_tokens)
        content_tokens = len(tokenizer.encode(content))
        return build_representation(
            representation_type="summary",
            content=content,
            content_tokens=content_tokens,
            source_corpus_hash=corpus_hash,
            compiler_model=settings.model or "deterministic-mock-1",
            compiler_prompt_version=settings.summary_prompt_version,
            compiler_input_tokens=source_reading_tokens(ordered, tokenizer),
            compiler_output_tokens=content_tokens,
            compiler_calls=0,
        )

    counting = CountingAdapter(adapter)
    response = counting.generate(
        ModelRequest(
            system_prompt=_SUMMARY_SYSTEM,
            user_prompt=_summary_user(ordered, writing_condition_tokens),
            max_output_tokens=writing_condition_tokens,
            temperature=0.0,
            top_p=1.0,
            seed=0,
        )
    )
    budget = controller.apply(response.text, writing_condition_tokens)
    return build_representation(
        representation_type="summary",
        content=budget.content,
        content_tokens=budget.post_truncation_tokens,
        source_corpus_hash=corpus_hash,
        compiler_model=adapter.model_name,
        compiler_prompt_version=settings.summary_prompt_version,
        compiler_input_tokens=counting.input_tokens,
        compiler_output_tokens=counting.output_tokens,
        compiler_calls=counting.calls,
    )
