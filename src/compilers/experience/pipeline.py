from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from src.adapters.counting import CountingAdapter
from src.adapters.model import ModelAdapter
from src.budget.tokenizer import Tokenizer
from src.compilers.config import CompilerSettings
from src.compilers.corpus import source_corpus_hash
from src.compilers.experience.adjudicate import adjudicate_pair
from src.compilers.experience.canonicalize import CanonicalExperience, canonicalize
from src.compilers.experience.extract import extract_candidates
from src.compilers.experience.library import LibraryBudgetResult, build_library
from src.compilers.experience.pair_retrieval import retrieve_pairs
from src.compilers.experience.span_validate import validate_candidates
from src.compilers.experience.verify import verify_candidate
from src.domain.models import Experience, ExperienceDerivedMeta, SourcePaper


@dataclass(frozen=True)
class ExperienceLibraryResult:
    source_corpus_hash: str
    adapter_mode: str
    candidate_count: int
    span_verified_count: int
    span_rejected_count: int
    support_verified_count: int
    verifier_rejected_count: int
    retrieved_pair_count: int
    adjudicated_pair_count: int
    merge_edge_count: int
    canonical_count: int
    stable_core_count: int
    supported_rare_count: int
    compiler_calls: int
    compiler_input_tokens: int
    compiler_output_tokens: int
    experiences: tuple[Experience, ...]
    derived_meta: tuple[ExperienceDerivedMeta, ...]
    canonical: tuple["CanonicalExperience", ...]
    library: LibraryBudgetResult
    trace: tuple[dict[str, Any], ...]
    merged_without_adjudication: bool


def compile_experience_library(
    sources: Sequence[SourcePaper],
    settings: CompilerSettings,
    tokenizer: Tokenizer,
    writing_condition_tokens: int,
    adapter: ModelAdapter | None = None,
) -> ExperienceLibraryResult:
    """Run the frozen main Experience pipeline (spec §8) over the source corpus."""
    corpus_hash = source_corpus_hash(sources)
    if adapter is not None:
        adapter = CountingAdapter(adapter)
    extraction = extract_candidates(sources, settings, adapter)
    source_by_id = {source.source_id: source for source in sources}

    validated, span_trace = validate_candidates(extraction.candidates, source_by_id)
    trace = list(extraction.trace) + span_trace

    verified: list = []
    for result in validated:
        if result.grounding_status == "rejected":
            continue
        checked = verify_candidate(result.candidate, settings, adapter)
        trace.append(
            {
                "source_id": checked.candidate.source_id,
                "stage": "verify",
                "level": "error" if checked.grounding_status == "rejected" else "info",
                "grounding_status": checked.grounding_status,
                "rejection_reason": checked.rejection_reason,
                "verifier_result": checked.verifier_result,
            }
        )
        verified.append(checked)

    pool = [item for item in verified if item.grounding_status == "support_verified"]
    pairs = retrieve_pairs(
        pool,
        dimensions=settings.retrieval_dimensions,
        top_k=settings.retrieval_top_k,
        min_cosine=settings.retrieval_min_cosine,
    )
    adjudicated = []
    for pair in pairs:
        verdict = adjudicate_pair(
            pool[pair.index_a],
            pool[pair.index_b],
            pair.index_a,
            pair.index_b,
            settings,
            adapter,
        )
        adjudicated.append(verdict)
        trace.append(
            {
                "stage": "adjudicate",
                "level": "info",
                "index_a": pair.index_a,
                "index_b": pair.index_b,
                "relation": verdict.relation,
                "compatible": verdict.compatible_for_canonicalization,
                "applicability_conflict": verdict.applicability_conflict,
                "merges": verdict.merges,
                "notes": verdict.notes,
            }
        )

    canonical = canonicalize(pool, adjudicated, settings)
    library = build_library(
        canonical.canonical,
        tokenizer,
        writing_condition_tokens,
        settings.library_tier_order,
    )

    experiences = tuple(item.experience for item in canonical.canonical)
    meta = tuple(item.meta for item in canonical.canonical)
    return ExperienceLibraryResult(
        source_corpus_hash=corpus_hash,
        adapter_mode=extraction.adapter_mode,
        candidate_count=len(extraction.candidates),
        span_verified_count=sum(
            1 for item in validated if item.grounding_status == "span_verified"
        ),
        span_rejected_count=sum(
            1 for item in validated if item.grounding_status == "rejected"
        ),
        support_verified_count=len(pool),
        verifier_rejected_count=len(verified) - len(pool),
        retrieved_pair_count=len(pairs),
        adjudicated_pair_count=len(adjudicated),
        merge_edge_count=len(canonical.merge_record),
        canonical_count=len(canonical.canonical),
        stable_core_count=sum(1 for item in canonical.canonical if item.meta.tier == "stable_core"),
        supported_rare_count=sum(
            1 for item in canonical.canonical if item.meta.tier == "supported_rare"
        ),
        compiler_calls=getattr(adapter, "calls", 0),
        compiler_input_tokens=getattr(adapter, "input_tokens", 0),
        compiler_output_tokens=getattr(adapter, "output_tokens", 0),
        experiences=experiences,
        derived_meta=meta,
        canonical=tuple(canonical.canonical),
        library=library,
        trace=tuple(trace),
        merged_without_adjudication=False,
    )
