"""Team-wide shared strategy in SSM Parameter Store.

Parameter /afc/team/strategy holds JSON {"strategy","reason","seq","ts","by"}.
Every agent reads it (1 s cache, ~300 ms hard timeout). The GK manager writes it.

Never raises. On any error the last known value stays in use (default BALANCED),
and SSM is not asked again for ERROR_BACKOFF_S seconds.
Env SHARED_STATE=0: no SSM at all, only the in-process value.
"""

import json
import os
import threading
import time

import strategies

PARAM_NAME = "/afc/team/strategy"
CACHE_S = 1.0
TIMEOUT_S = 0.3
ERROR_BACKOFF_S = 5.0

_lock = threading.Lock()
_client = None
_state = {
    "value": {"strategy": strategies.DEFAULT, "reason": "default", "seq": 0, "ts": 0, "by": "default"},
    "source": "default",     # default | shared | local
    "fetched": -1e9,         # monotonic time of the last read attempt
    "retry_after": 0.0,
    "errors": 0,
}


def enabled() -> bool:
    return os.environ.get("SHARED_STATE", "1") != "0"


def set_client(client) -> None:
    """Inject a client (tests) or None to rebuild lazily."""
    global _client
    _client = client


def reset() -> None:
    """Forget all state (tests, new match)."""
    _state.update(
        value={"strategy": strategies.DEFAULT, "reason": "default", "seq": 0, "ts": 0, "by": "default"},
        source="default", fetched=-1e9, retry_after=0.0, errors=0)


def _get_client():
    global _client
    if _client is None:
        import boto3
        from botocore.config import Config
        region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
        cfg = Config(connect_timeout=TIMEOUT_S, read_timeout=TIMEOUT_S,
                     retries={"total_max_attempts": 1, "mode": "standard"})
        _client = boto3.client("ssm", region_name=region, config=cfg)
    return _client


def _parse(raw) -> dict | None:
    try:
        v = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError):
        return None
    if not isinstance(v, dict):
        return None
    name = strategies.normalize(v.get("strategy"))
    if name is None:
        return None
    seq = v.get("seq", 0)
    return {
        "strategy": name,
        "reason": str(v.get("reason", ""))[:200],
        "seq": seq if isinstance(seq, (int, float)) else 0,
        "ts": v.get("ts", 0),
        "by": str(v.get("by", "")),
    }


def _refresh(now: float) -> None:
    _state["fetched"] = now
    if now < _state["retry_after"]:
        return
    try:
        resp = _get_client().get_parameter(Name=PARAM_NAME)
        v = _parse(resp["Parameter"]["Value"])
        if v is not None:
            _state["value"] = v
            _state["source"] = "shared"
    except Exception as e:  # noqa: BLE001  (missing param, timeout, IAM, throttling ...)
        _state["errors"] += 1
        _state["last_error"] = repr(e)[:200]
        _state["retry_after"] = now + ERROR_BACKOFF_S


def read() -> dict:
    """Current strategy record (copy). Cached for CACHE_S seconds."""
    try:
        if enabled():
            now = time.monotonic()
            if now - _state["fetched"] >= CACHE_S and _lock.acquire(blocking=False):
                try:
                    _refresh(now)
                finally:
                    _lock.release()
        return dict(_state["value"])
    except Exception:  # noqa: BLE001
        return {"strategy": strategies.DEFAULT, "reason": "error", "seq": 0, "ts": 0, "by": "error"}


def read_strategy() -> str:
    """Name of the current team strategy. Never raises."""
    return read().get("strategy", strategies.DEFAULT)


def source() -> str:
    """'shared' (read from SSM), 'local' (written in this process) or 'default'."""
    return _state.get("source", "default")


def current() -> dict:
    """Last known record without any SSM call."""
    return dict(_state["value"])


def write_strategy(name, reason: str = "", by: str = "manager") -> bool:
    """Store a new strategy (seq + 1). Local value always updates. True if SSM accepted it."""
    try:
        name = strategies.normalize(name) or strategies.DEFAULT
        seq = int(_state["value"].get("seq", 0) or 0) + 1
        rec = {"strategy": name, "reason": str(reason)[:200], "seq": seq,
               "ts": round(time.time(), 1), "by": by}
        _state["value"] = rec
        _state["source"] = "local"
        if not enabled():
            return False
        _get_client().put_parameter(Name=PARAM_NAME, Value=json.dumps(rec), Type="String", Overwrite=True)
        _state["source"] = "shared"
        _state["fetched"] = time.monotonic()   # our write is the freshest value
        return True
    except Exception as e:  # noqa: BLE001
        _state["errors"] += 1
        _state["last_error"] = repr(e)[:200]
        return False
