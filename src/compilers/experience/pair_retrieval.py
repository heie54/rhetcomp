from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Sequence

from src.compilers.experience.verify import VerifiedCandidate

_WORD = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True)
class RetrievedPair:
    index_a: int
    index_b: int
    similarity: float


def _tokens(text: str) -> list[str]:
    return [word.lower() for word in _WORD.findall(text) if word]


def _feature_vector(text: str, dimensions: int, seed: int) -> list[float]:
    """Deterministic hashed bag-of-ngrams embedding (no external dependency)."""
    vector = [0.0] * dimensions
    tokens = _tokens(text)
    for size in (1, 2):
        for index in range(len(tokens) - size + 1):
            gram = " ".join(tokens[index:index + size])
            digest = hashlib.sha256(f"{seed}:{gram}".encode("utf-8")).digest()
            feature_index = int.from_bytes(digest[:4], "big") % dimensions
            sign = 1.0 if (digest[4] & 1) else -1.0
            vector[feature_index] += sign
    return vector


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    norm_left = math.sqrt(sum(value * value for value in left))
    norm_right = math.sqrt(sum(value * value for value in right))
    if norm_left == 0.0 or norm_right == 0.0:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    return dot / (norm_left * norm_right)


def candidate_vectors(
    verified: Sequence[VerifiedCandidate],
    dimensions: int,
    seed: int = 0,
) -> list[list[float]]:
    return [
        _feature_vector(
            f"{item.candidate.observed_pattern} {item.candidate.strategy} "
            f"{item.candidate.applicable_when}",
            dimensions,
            seed,
        )
        for item in verified
    ]


def retrieve_pairs(
    verified: Sequence[VerifiedCandidate],
    dimensions: int,
    top_k: int,
    min_cosine: float,
    seed: int = 0,
) -> list[RetrievedPair]:
    """Embedding candidate-pair retrieval (spec §8.4). Retrieval only; never a merge decision."""
    vectors = candidate_vectors(verified, dimensions, seed)
    pairs: list[RetrievedPair] = []
    for index_a in range(len(vectors)):
        for index_b in range(index_a + 1, len(vectors)):
            similarity = _cosine(vectors[index_a], vectors[index_b])
            if similarity >= min_cosine:
                pairs.append(RetrievedPair(index_a, index_b, similarity))
    pairs.sort(key=lambda pair: (-pair.similarity, pair.index_a, pair.index_b))
    return pairs[:top_k]
