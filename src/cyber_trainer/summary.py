from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, List

from .rules_deadlift import Hint


@dataclass
class SessionSummary:
    started_ts_ms: int
    ended_ts_ms: int
    hints: List[Hint]

    def counts(self) -> Dict[str, int]:
        c: Dict[str, int] = {}
        for h in self.hints:
            c[h.issue_code] = c.get(h.issue_code, 0) + 1
        return c


def save_summary(summary: SessionSummary, out_dir: str = "sessions") -> str:
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"session_{stamp}.json")

    payload = {
        "started_ts_ms": summary.started_ts_ms,
        "ended_ts_ms": summary.ended_ts_ms,
        "counts": summary.counts(),
        "timeline": [asdict(h) for h in summary.hints],
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return path
