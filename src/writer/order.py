from __future__ import annotations

import random
from hashlib import sha256

from src.writer.config import WRITER_CONDITIONS


def condition_order(target_id: str, seed: str) -> tuple[str, ...]:
    if not target_id or not seed:
        raise ValueError("Condition ordering requires target id and local experiment seed")
    digest = sha256(f"{seed}\0{target_id}".encode("utf-8")).digest()
    generator = random.Random(int.from_bytes(digest[:8], "big"))
    conditions = list(WRITER_CONDITIONS)
    generator.shuffle(conditions)
    if tuple(conditions) == WRITER_CONDITIONS:
        conditions = conditions[1:] + conditions[:1]
    return tuple(conditions)
