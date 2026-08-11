from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from src.common.jsonio import canonical_json


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
            try:
                structured = json.loads(content)
            except json.JSONDecodeError:
                structured = None
            if isinstance(structured, dict):
                preserved = tuple(
                    field for field in ("target_id", "title", "abstract") if field in structured
                )
                if "title" in structured and "abstract" in structured:
                    return self.apply_structured(
                        structured,
                        limit,
                        preserved_fields=preserved,
                        field_order=("non_intro_body", "reference_metadata"),
                    )
                raise ValueError("Over-budget JSON requires an explicit structured budget policy")
            output = self.tokenizer.decode(tokens[:limit])
            output_tokens = len(self.tokenizer.encode(output))
            truncated = True
        return BudgetResult(
            content=output,
            pre_truncation_tokens=len(tokens),
            post_truncation_tokens=output_tokens,
            truncated=truncated,
            tokenizer_version=self.tokenizer.version,
        )

    def apply_structured(
        self,
        payload: dict[str, Any],
        limit: int,
        *,
        preserved_fields: Sequence[str],
        field_order: Sequence[str],
    ) -> BudgetResult:
        """Budget canonical JSON without ever truncating its serialized syntax."""
        if limit < 0:
            raise ValueError("Token limit must be non-negative")
        missing = [field for field in preserved_fields if field not in payload]
        if missing:
            raise ValueError(f"Structured budget missing preserved fields: {', '.join(missing)}")
        unmanaged = set(payload) - set(preserved_fields) - set(field_order)
        if unmanaged:
            raise ValueError(f"Structured budget has unmanaged fields: {', '.join(sorted(unmanaged))}")

        original = canonical_json(payload)
        pre_tokens = len(self.tokenizer.encode(original))
        if pre_tokens <= limit:
            return BudgetResult(
                content=original,
                pre_truncation_tokens=pre_tokens,
                post_truncation_tokens=pre_tokens,
                truncated=False,
                tokenizer_version=self.tokenizer.version,
            )

        reduced: dict[str, Any] = {
            field: deepcopy(payload[field]) for field in preserved_fields
        }
        for field in field_order:
            value = payload.get(field)
            if isinstance(value, dict):
                reduced[field] = {}
            elif isinstance(value, (list, tuple)):
                reduced[field] = []
            elif isinstance(value, str):
                reduced[field] = ""
            else:
                raise ValueError(f"Unsupported structured budget field: {field}")

        if self._count_structured(reduced) > limit:
            raise ValueError("Token limit is too small to preserve required structured fields")

        for field in field_order:
            value = payload.get(field)
            if isinstance(value, dict):
                self._fill_mapping(reduced, field, value, limit)
            elif isinstance(value, (list, tuple)):
                self._fill_sequence(reduced, field, value, limit)
            elif isinstance(value, str):
                reduced[field] = self._longest_text_prefix(reduced, field, value, limit)

        output = canonical_json(reduced)
        post_tokens = len(self.tokenizer.encode(output))
        if post_tokens > limit:
            raise AssertionError("Structured budget strategy exceeded its token limit")
        return BudgetResult(
            content=output,
            pre_truncation_tokens=pre_tokens,
            post_truncation_tokens=post_tokens,
            truncated=True,
            tokenizer_version=self.tokenizer.version,
        )

    def _count_structured(self, payload: dict[str, Any]) -> int:
        return len(self.tokenizer.encode(canonical_json(payload)))

    def _fill_mapping(
        self,
        reduced: dict[str, Any],
        field: str,
        values: dict[Any, Any],
        limit: int,
    ) -> None:
        output = reduced[field]
        for key in sorted(values, key=lambda item: str(item)):
            value = values[key]
            output[key] = deepcopy(value)
            if self._count_structured(reduced) <= limit:
                continue
            del output[key]
            if isinstance(value, str):
                prefix = self._longest_mapping_text_prefix(reduced, field, key, value, limit)
                if prefix:
                    output[key] = prefix
            break

    def _fill_sequence(
        self,
        reduced: dict[str, Any],
        field: str,
        values: Sequence[Any],
        limit: int,
    ) -> None:
        output = reduced[field]
        for value in values:
            output.append(deepcopy(value))
            if self._count_structured(reduced) > limit:
                output.pop()
                break

    def _longest_mapping_text_prefix(
        self,
        reduced: dict[str, Any],
        field: str,
        key: Any,
        value: str,
        limit: int,
    ) -> str:
        output = reduced[field]
        low, high = 0, len(value)
        while low < high:
            middle = (low + high + 1) // 2
            output[key] = value[:middle]
            if self._count_structured(reduced) <= limit:
                low = middle
            else:
                high = middle - 1
        output.pop(key, None)
        return value[:low]

    def _longest_text_prefix(
        self,
        reduced: dict[str, Any],
        field: str,
        value: str,
        limit: int,
    ) -> str:
        low, high = 0, len(value)
        while low < high:
            middle = (low + high + 1) // 2
            reduced[field] = value[:middle]
            if self._count_structured(reduced) <= limit:
                low = middle
            else:
                high = middle - 1
        return value[:low]
