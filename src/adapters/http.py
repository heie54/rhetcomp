from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class HttpResponse:
    payload: Mapping[str, Any]
    headers: Mapping[str, str]


class ProviderTransportError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code

    @property
    def retryable(self) -> bool:
        return self.status_code is None or self.status_code == 429 or self.status_code >= 500


HttpTransport = Callable[[str, Mapping[str, str], Mapping[str, Any], float], HttpResponse]


def post_json(
    url: str,
    headers: Mapping[str, str],
    payload: Mapping[str, Any],
    timeout_seconds: float,
) -> HttpResponse:
    request = Request(
        url,
        data=json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
        headers=dict(headers),
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8")
            response_headers = {key.lower(): value for key, value in response.headers.items()}
    except HTTPError as exc:
        detail = exc.read(2048).decode("utf-8", errors="replace")
        raise ProviderTransportError(
            f"Provider HTTP {exc.code}: {detail}", status_code=exc.code
        ) from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise ProviderTransportError(f"Provider transport failed: {exc}") from exc
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProviderTransportError("Provider returned non-JSON response", status_code=502) from exc
    if not isinstance(decoded, dict):
        raise ProviderTransportError("Provider response must be a JSON object", status_code=502)
    return HttpResponse(payload=decoded, headers=response_headers)
