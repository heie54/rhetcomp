from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

from huggingface_hub import hf_hub_download


_HF_DATASET_RESOLVE_URL = re.compile(
    r"^https://huggingface\.co/datasets/(?P<repo>.+?)/resolve/"
    r"(?P<revision>[^/]+)/(?P<filename>[^?]+)(?:\?.*)?$"
)


def fetch_bytes(url: str, *, timeout_seconds: float = 180.0) -> bytes:
    request = Request(
        url,
        headers={"Accept": "application/json, application/xml, application/pdf, text/plain, */*", "User-Agent": "RhetComp/0.1 formal-data-bootstrap"},
    )
    with urlopen(request, timeout=timeout_seconds) as response:
        return response.read()


def fetch_to_path(
    url: str,
    destination: Path,
    *,
    timeout_seconds: float = 180.0,
) -> None:
    request = Request(
        url,
        headers={
            "Accept": "application/json, application/x-ndjson, text/plain, */*",
            "User-Agent": "RhetComp/0.1 formal-data-bootstrap",
        },
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with urlopen(request, timeout=timeout_seconds) as response, temporary.open("wb") as handle:
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def fetch_huggingface_dataset_to_path(url: str, destination: Path) -> None:
    """Download one dataset file at an immutable revision with the official Hub client."""

    match = _HF_DATASET_RESOLVE_URL.match(url)
    if match is None:
        raise ValueError("Hugging Face dataset URL must include repo, revision, and filename")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="hf-stage-", dir=destination.parent
    ) as staging_directory:
        downloaded = Path(
            hf_hub_download(
                repo_id=match.group("repo"),
                filename=match.group("filename"),
                repo_type="dataset",
                revision=match.group("revision"),
                local_dir=staging_directory,
                force_download=True,
            )
        )
        downloaded.replace(destination)
