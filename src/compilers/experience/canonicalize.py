from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from src.common.jsonio import sha256_json
from src.compilers.config import CompilerSettings
from src.compilers.experience.adjudicate import AdjudicationResult
from src.compilers.experience.verify import VerifiedCandidate
from src.domain.models import (
    EvidenceLocation,
    Experience,
    ExperienceDerivedMeta,
    ExperienceEvidence,
)


@dataclass(frozen=True)
class CanonicalExperience:
    experience: Experience
    meta: ExperienceDerivedMeta
    member_indices: tuple[int, ...]


@dataclass(frozen=True)
class CanonicalizationResult:
    canonical: tuple[CanonicalExperience, ...]
    merge_record: tuple[dict[str, Any], ...]


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left: int, right: int) -> None:
        self.parent[self.find(left)] = self.find(right)


def _evidence_key(candidate: VerifiedCandidate) -> tuple[str, int, int, int]:
    location = candidate.candidate.location
    return (candidate.candidate.source_id, location.paragraph, location.sentence_start, location.sentence_end)


def _select_representative(
    group: list[int],
    verified: Sequence[VerifiedCandidate],
    subsumed: set[int],
) -> int:
    candidates = [index for index in group if index not in subsumed] or group
    return min(
        candidates,
        key=lambda index: (
            -len(verified[index].candidate.span),
            _evidence_key(verified[index]),
        ),
    )


def _experience_id(observed_pattern: str, strategy: str, applicable_when: str, evidence: list[ExperienceEvidence]) -> str:
    provenance = [
        {
            "source_id": item.source_id,
            "paragraph": item.location.paragraph,
            "sentence_start": item.location.sentence_start,
            "sentence_end": item.location.sentence_end,
        }
        for item in evidence
    ]
    digest = sha256_json(
        {
            "observed_pattern": observed_pattern,
            "strategy": strategy,
            "applicable_when": applicable_when,
            "evidence": provenance,
        }
    )
    return f"exp_{digest[:20]}"


def canonicalize(
    verified: Sequence[VerifiedCandidate],
    merge_pairs: Sequence[AdjudicationResult],
    settings: CompilerSettings,
) -> CanonicalizationResult:
    """Canonicalization + provenance union (spec §8.6) and tier assignment (§8.7).

    The merge graph is built exclusively from adjudicated, merge-compatible pairs, so
    canonicalization can never merge without semantic adjudication.
    """
    merge_pairs = [pair for pair in merge_pairs if pair.merges]
    union_find = _UnionFind(len(verified))
    for pair in merge_pairs:
        union_find.union(pair.index_a, pair.index_b)

    groups: dict[int, list[int]] = {}
    for index in range(len(verified)):
        groups.setdefault(union_find.find(index), []).append(index)

    subsumed: set[int] = set()
    for pair in merge_pairs:
        if pair.relation == "a_subsumes_b":
            subsumed.add(pair.index_b)
        elif pair.relation == "b_subsumes_a":
            subsumed.add(pair.index_a)

    canonical_list: list[CanonicalExperience] = []
    merge_record: list[dict[str, Any]] = []
    for cluster_index, group in enumerate(
        sorted(groups.values(), key=lambda group: sorted(group)), start=1
    ):
        group = sorted(group)
        representative = _select_representative(group, verified, subsumed)
        rep = verified[representative]

        evidence_by_key: dict[tuple[str, int, int, int], ExperienceEvidence] = {}
        for index in group:
            candidate = verified[index].candidate
            key = _evidence_key(verified[index])
            evidence_by_key[key] = ExperienceEvidence(
                source_id=candidate.source_id,
                location=EvidenceLocation(
                    section="Introduction",
                    paragraph=candidate.location.paragraph,
                    sentence_start=candidate.location.sentence_start,
                    sentence_end=candidate.location.sentence_end,
                ),
                span=candidate.span,
                support_relation="instantiates_observed_pattern",
            )
        evidence = [evidence_by_key[key] for key in sorted(evidence_by_key)]
        distinct_sources = len({item.source_id for item in evidence})
        tier = "stable_core" if distinct_sources >= 2 else "supported_rare"

        experience = Experience(
            experience_id=_experience_id(
                rep.candidate.observed_pattern,
                rep.candidate.strategy,
                rep.candidate.applicable_when,
                evidence,
            ),
            observed_pattern=rep.candidate.observed_pattern,
            strategy=rep.candidate.strategy,
            applicable_when=rep.candidate.applicable_when,
            evidence=tuple(evidence),
            grounding_status="support_verified",
        )
        meta = ExperienceDerivedMeta(
            experience_id=experience.experience_id,
            distinct_source_count=distinct_sources,
            cluster_id=f"cluster_{cluster_index:03d}",
            tier=tier,
            verifier_result=rep.verifier_result,
            verifier_score=None,
            run_support=None,
        )
        canonical_list.append(
            CanonicalExperience(
                experience=experience,
                meta=meta,
                member_indices=tuple(group),
            )
        )
        if len(group) > 1:
            merge_record.append(
                {
                    "cluster_id": meta.cluster_id,
                    "experience_id": experience.experience_id,
                    "member_indices": group,
                    "representative_index": representative,
                    "tier": tier,
                    "distinct_source_count": distinct_sources,
                }
            )

    return CanonicalizationResult(
        canonical=tuple(canonical_list),
        merge_record=tuple(merge_record),
    )
