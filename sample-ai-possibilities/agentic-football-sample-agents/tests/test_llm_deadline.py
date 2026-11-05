"""Per-tick LLM path (enable_llm=true): the tick never waits past LLM_DEADLINE_S."""
import asyncio
import json
import time

import agent_base
from fallback import FallbackConfig
from test_helpers import GAME_STATE


class _App:
    class logger:
        info = warn = error = staticmethod(lambda m: None)

    def entrypoint(self, fn):
        return fn


def _agent(delay):
    class A:
        model = None
        system_prompt = "x"

        def __init__(self, *a, **k):
            pass

        def __call__(self, s):
            time.sleep(delay)
            return '[{"commandType":"MOVE_TO","parameters":{"target_x":1,"target_y":2,"sprint":false}}]'
    return A


def _run(monkeypatch, delay):
    monkeypatch.setenv("ENABLE_LLM", "1")
    monkeypatch.setattr(agent_base, "LLM_DEADLINE_S", 0.3)
    A = _agent(delay)
    monkeypatch.setattr(agent_base, "Agent", A)
    h = agent_base.create_invoke_handler(_App(), A(), 2, "MID",
                                         lambda gs, t, p: [{"commandType": "SET_STANCE"}], FallbackConfig())
    payload = {"prompt": json.dumps({"gameState": GAME_STATE, "teamId": 0, "myPlayers": [2]})}

    async def go():
        return [x async for x in h(payload, None)]
    t0 = time.perf_counter()
    out = asyncio.run(go())
    return json.loads(out[0]), time.perf_counter() - t0


def test_slow_llm_falls_back_to_policy_in_time(monkeypatch, capsys):
    cmds, dt = _run(monkeypatch, delay=3.0)
    assert dt < 0.6, dt
    recs = [json.loads(l) for l in capsys.readouterr().out.splitlines() if l.startswith("{")]
    assert recs[-1]["source"] == "policy" and recs[-1]["error"] == "llm_timeout"
    assert cmds[0]["playerId"] == 2 and cmds[0]["teamId"] == 0


def test_fast_llm_is_used(monkeypatch, capsys):
    cmds, dt = _run(monkeypatch, delay=0.0)
    recs = [json.loads(l) for l in capsys.readouterr().out.splitlines() if l.startswith("{")]
    assert recs[-1]["source"] == "llm"
    assert cmds[0]["commandType"] == "MOVE_TO"
