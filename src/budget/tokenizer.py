from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, Sequence


class Tokenizer(Protocol):
    @property
    def version(self) -> str: ...

    def encode(self, text: str) -> Sequence[str]: ...

    def decode(self, tokens: Sequence[str]) -> str: ...


class DeterministicRegexTokenizer:
    """A dependency-free, versioned pilot tokenizer replaceable by a model tokenizer."""

    _pattern = re.compile(r"\w+|[^\w\s]", re.UNICODE)

    def __init__(self, version: str = "1") -> None:
        if version != "1":
            raise ValueError(f"Unsupported deterministic_regex tokenizer version: {version}")
        self._version = f"deterministic_regex:{version}"

    @property
    def version(self) -> str:
        return self._version

    def encode(self, text: str) -> Sequence[str]:
        return self._pattern.findall(text)

    def decode(self, tokens: Sequence[str]) -> str:
        return " ".join(tokens)


@dataclass(frozen=True)
class BudgetResult:
    content: str
    pre_truncation_tokens: int
    post_truncation_tokens: int
    truncated: bool
    tokenizer_version: str


class BudgetController:
    def __init__(self, tokenizer: Tokenizer) -> None:
        self.tokenizer = tokenizer

    def apply(self, content: str, limit: int) -> BudgetResult:
        if limit < 0:
            raise ValueError("Token limit must be non-negative")
        tokens = list(self.tokenizer.encode(content))
        if len(tokens) <= limit:
            output = content
            output_tokens = len(tokens)
            truncated = False
        else:
            output = self.tokenizer.decode(tokens[:limit])
            output_tokens = limit
            truncated = True
        return BudgetResult(
            content=output,
            pre_truncation_tokens=len(tokens),
            post_truncation_tokens=output_tokens,
            truncated=truncated,
            tokenizer_version=self.tokenizer.version,
        )
