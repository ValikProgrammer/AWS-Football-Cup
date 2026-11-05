"""Structured per-tick logging: one JSON object per line on stdout.

AgentCore forwards the runtime's stdout to CloudWatch Logs, so each line is
queryable with Logs Insights (e.g. `filter kind="tick" | stats count() by source`).

Never raises: logging must not break a tick.
"""

import json
import sys
import time

# Set to False to drop the raw payload from the log line (smaller lines).
LOG_PAYLOAD = True


import os as _os
import uuid as _uuid

# Process identity: tells whether one Python process serves the whole match
# (module globals such as memory survive) or each call starts fresh.
_PROC = {"proc_id": _uuid.uuid4().hex[:8], "os_pid": _os.getpid(), "started": round(time.time(), 1)}
_CALLS = 0


def now_ms() -> float:
    return time.perf_counter() * 1000.0


def log_tick(
    label: str,
    team_id,
    player_id,
    source: str,
    latency_ms: float,
    commands,
    prompt_data=None,
    error: str | None = None,
    extra: dict | None = None,
) -> None:
    """Write one tick record. source is one of: policy, llm, fallback, last_resort."""
    global _CALLS
    _CALLS += 1
    try:
        gs = (prompt_data or {}).get("gameState", {}) if isinstance(prompt_data, dict) else {}
        rec = {
            "kind": "tick",
            "ts": round(time.time(), 3),
            "label": label,
            "team_id": team_id,
            "pid": player_id,
            "tick": gs.get("tick"),
            "game_time": gs.get("gameTime"),
            "play_mode": gs.get("playMode"),
            "source": source,
            "latency_ms": round(latency_ms, 1),
            "proc": _PROC["proc_id"],
            "os_pid": _PROC["os_pid"],
            "proc_calls": _CALLS,
            "cmd": commands[0] if isinstance(commands, list) and commands else commands,
        }
        if error:
            rec["error"] = error[:300]
        if extra:
            rec.update(extra)
        if LOG_PAYLOAD:
            rec["payload"] = prompt_data
        sys.stdout.write(json.dumps(rec, separators=(",", ":"), default=str) + "\n")
        sys.stdout.flush()
    except Exception:
        pass
