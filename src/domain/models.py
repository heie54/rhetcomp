from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, ClassVar, Literal


def _required_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be non-empty text")
    return value


@dataclass(frozen=True)
class Sentence:
    sentence_id: int
    text: str
    char_start: int
    char_end: int

    def __post_init__(self) -> None:
        if self.sentence_id < 1 or self.char_start < 0 or self.char_end <= self.char_start:
            raise ValueError("Invalid sentence coordinates")
        _required_text(self.text, "sentence.text")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Sentence":
        return cls(**value)


@dataclass(frozen=True)
class Paragraph:
    paragraph_id: int
    sentences: tuple[Sentence, ...]

    def __post_init__(self) -> None:
        if self.paragraph_id < 1 or not self.sentences:
            raise ValueError("Paragraph must have an id and at least one sentence")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Paragraph":
        return cls(
            paragraph_id=value["paragraph_id"],
            sentences=tuple(Sentence.from_dict(item) for item in value["sentences"]),
        )


@dataclass(frozen=True)
class Introduction:
    normalized_text: str
    paragraphs: tuple[Paragraph, ...]

    def __post_init__(self) -> None:
        _required_text(self.normalized_text, "introduction.normalized_text")
        if not self.paragraphs:
            raise ValueError("Introduction must contain at least one paragraph")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Introduction":
        return cls(
            normalized_text=value["normalized_text"],
            paragraphs=tuple(Paragraph.from_dict(item) for item in value["paragraphs"]),
        )


@dataclass(frozen=True)
class SourcePaper:
    source_id: str
    title: str
    authors: tuple[str, ...]
    venue: str
    track: str
    introduction: Introduction
    document_hash: str

    def __post_init__(self) -> None:
        for name in ("source_id", "title", "venue", "introduction.document_hash"):
            value = self.document_hash if name == "introduction.document_hash" else getattr(self, name)
            _required_text(value, name)
        if not self.authors or any(not isinstance(author, str) or not author.strip() for author in self.authors):
            raise ValueError("authors must contain non-empty names")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SourcePaper":
        return cls(
            source_id=value["source_id"],
            title=value["title"],
            authors=tuple(value["authors"]),
            venue=value["venue"],
            track=value.get("track", ""),
            introduction=Introduction.from_dict(value["introduction"]),
            document_hash=value["document_hash"],
        )


@dataclass(frozen=True)
class TargetVisible:
    target_id: str
    title: str
    abstract: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TargetEvidence:
    target_id: str
    non_intro_sections: dict[str, str]
    reference_metadata: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TargetGold:
    target_id: str
    introduction: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TargetEvidencePack:
    target_id: str
    budget_tokens: int
    content: str
    source_fields: tuple[str, ...]
    input_hash: str
    tokenizer_version: str
    pre_truncation_tokens: int
    post_truncation_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TargetEvidencePack":
        return cls(
            target_id=value["target_id"],
            budget_tokens=value["budget_tokens"],
            content=value["content"],
            source_fields=tuple(value["source_fields"]),
            input_hash=value["input_hash"],
            tokenizer_version=value["tokenizer_version"],
            pre_truncation_tokens=value["pre_truncation_tokens"],
            post_truncation_tokens=value["post_truncation_tokens"],
        )


@dataclass(frozen=True)
class EvidenceLocation:
    section: Literal["Introduction"]
    paragraph: int
    sentence_start: int
    sentence_end: int

    def __post_init__(self) -> None:
        if self.section != "Introduction":
            raise ValueError("Evidence location section must be Introduction")
        if (
            self.paragraph < 1
            or self.sentence_start < 1
            or self.sentence_end < self.sentence_start
        ):
            raise ValueError("Invalid evidence location coordinates")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EvidenceLocation":
        return cls(**value)


@dataclass(frozen=True)
class ExperienceEvidence:
    source_id: str
    location: EvidenceLocation
    span: str
    support_relation: Literal["instantiates_observed_pattern"]

    def __post_init__(self) -> None:
        _required_text(self.source_id, "experience_evidence.source_id")
        _required_text(self.span, "experience_evidence.span")
        if self.support_relation != "instantiates_observed_pattern":
            raise ValueError("Invalid support_relation")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ExperienceEvidence":
        return cls(
            source_id=value["source_id"],
            location=EvidenceLocation.from_dict(value["location"]),
            span=value["span"],
            support_relation=value["support_relation"],
        )


@dataclass(frozen=True)
class Experience:
    ALLOWED_STATUSES: ClassVar[set[str]] = {
        "unverified", "span_verified", "support_verified", "rejected"
    }
    experience_id: str
    observed_pattern: str
    strategy: str
    applicable_when: str
    evidence: tuple[ExperienceEvidence, ...]
    grounding_status: str

    def __post_init__(self) -> None:
        if self.grounding_status not in self.ALLOWED_STATUSES:
            raise ValueError("Invalid grounding_status")
        for name in ("experience_id", "observed_pattern", "strategy", "applicable_when"):
            _required_text(getattr(self, name), f"experience.{name}")
        if not self.evidence:
            raise ValueError("Experience must carry at least one evidence span")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Experience":
        return cls(
            experience_id=value["experience_id"],
            observed_pattern=value["observed_pattern"],
            strategy=value["strategy"],
            applicable_when=value["applicable_when"],
            evidence=tuple(ExperienceEvidence.from_dict(item) for item in value["evidence"]),
            grounding_status=value["grounding_status"],
        )


@dataclass(frozen=True)
class ExperienceDerivedMeta:
    """Diagnostic metadata kept outside the semantic Experience schema (spec §7)."""

    experience_id: str
    distinct_source_count: int
    cluster_id: str
    tier: Literal["stable_core", "supported_rare"]
    verifier_result: dict[str, Any] | None
    verifier_score: float | None
    run_support: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RepresentationArtifact:
    ALLOWED_TYPES: ClassVar[set[str]] = {"raw", "summary", "guideline", "experience"}
    representation_id: str
    type: str
    source_corpus_hash: str
    compiler_model: str
    compiler_prompt_version: str
    compiler_input_tokens: int
    compiler_output_tokens: int
    compiler_calls: int
    content: str
    content_tokens: int
    content_hash: str
    run_id: str = "unscoped"
    run_mode: str = "mechanics"
    config_hash: str | None = None
    data_manifest_hash: str | None = None
    provider_profile_hash: str | None = None

    def __post_init__(self) -> None:
        if self.type not in self.ALLOWED_TYPES:
            raise ValueError("Invalid representation type")
        if self.run_mode not in {"mechanics", "formal"}:
            raise ValueError("Invalid representation run mode")
        if self.run_mode == "formal" and any(
            not value
            for value in (
                self.run_id,
                self.config_hash,
                self.data_manifest_hash,
                self.provider_profile_hash,
            )
        ):
            raise ValueError("Formal representation requires complete artifact metadata")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RepresentationArtifact":
        return cls(**value)
