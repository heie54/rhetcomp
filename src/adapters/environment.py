from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping


def load_provider_environment(
    project_root: str | Path,
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Load local provider credentials without mutating the process environment.

    Values already present in the process environment override the ignored local
    ``.env`` file. The parser intentionally supports only simple KEY=VALUE lines.
    """

    values: dict[str, str] = {}
    path = Path(project_root) / ".env"
    if path.exists():
        for line_number, raw_line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                raise ValueError(f"Invalid .env line {line_number}: missing equals sign")
            name, value = line.split("=", 1)
            name = name.strip()
            value = value.strip()
            if not name:
                raise ValueError(f"Invalid .env line {line_number}: empty name")
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {'\"', "'"}:
                value = value[1:-1]
            values[name] = value
    inherited = os.environ if environ is None else environ
    for name, value in inherited.items():
        if value is not None:
            values[name] = str(value)
    return values
