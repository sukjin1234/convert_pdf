from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class StageTimer:
    """Small monotonic stage timer used by synchronous pipeline code."""

    _started: float = field(default_factory=time.perf_counter)
    _last: float = field(init=False)
    _stages: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._last = self._started

    def mark(self, name: str) -> float:
        now = time.perf_counter()
        elapsed_ms = (now - self._last) * 1000
        self._stages[name] = self._stages.get(name, 0.0) + elapsed_ms
        self._last = now
        return elapsed_ms

    def snapshot(self, *, final_stage: str | None = None) -> dict[str, float]:
        if final_stage:
            self.mark(final_stage)
        result = {name: round(value, 3) for name, value in self._stages.items()}
        result["total"] = round((time.perf_counter() - self._started) * 1000, 3)
        return result
