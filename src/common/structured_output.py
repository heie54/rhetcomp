from __future__ import annotations

import json
import re
from typing import Any


_JSON_FENCE = re.compile(r"^```(?:json)?\s*(?P<body>.*?)\s*```$", re.IGNORECASE | re.DOTALL)


def load_json_object(raw: str) -> dict[str, Any]:
    """Load one JSON object, allowing only a surrounding Markdown JSON fence."""
    text = raw.lstrip("\ufeff").strip()
    fenced = _JSON_FENCE.fullmatch(text)
    if fenced:
        text = fenced.group("body").strip()
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("Structured output must be one JSON object")
    return payload
