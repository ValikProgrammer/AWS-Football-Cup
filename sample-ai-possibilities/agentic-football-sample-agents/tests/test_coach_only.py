"""Coach-only mode (team_config: enable_manager=false, shared_state=false)."""
import agent_base
import orders
from test_helpers import GAME_STATE


def _run(monkeypatch, chat, label="GK", pid=0):
    calls = {"ssm": 0, "manager": 0}
    monkeypatch.setattr(agent_base._shared, "read_strategy",
                        lambda: calls.__setitem__("ssm", calls["ssm"] + 1) or "ALL_OUT")
    monkeypatch.setattr(agent_base._manager, "maybe_trigger",
                        lambda *a, **k: calls.__setitem__("manager", calls["manager"] + 1))
    monkeypatch.setattr(agent_base, "team_flag", lambda key, default: False)
    tr = orders.OrderTracker()
    monkeypatch.setattr(agent_base._orders, "TRACKER", tr)
    gs = dict(GAME_STATE, teamChat=chat, gameTime=30.0)
    cmds, extra = agent_base.strategy_commands(gs, 0, pid, label)
    return cmds, extra, calls


def test_no_manager_no_ssm_balanced_default(monkeypatch):
    cmds, extra, calls = _run(monkeypatch, [])
    assert calls == {"ssm": 0, "manager": 0}
    assert extra["strategy"] == "BALANCED" and extra["strategy_source"] == "default"
    assert cmds and cmds[0]["playerId"] == 0


def test_coach_code_still_applies(monkeypatch):
    _, extra, calls = _run(monkeypatch, [{"senderTeamCode": "home", "message": "S2", "timestamp": 0}], "MID", 2)
    assert extra["strategy"] == "HIGH_PRESS" and extra["strategy_source"] == "coach"
    assert calls["ssm"] == 0


def test_shipped_config_is_coach_only():
    assert agent_base.team_flag("enable_manager", True) is False
    assert agent_base.team_flag("shared_state", True) is False
    assert agent_base.llm_enabled() is False
