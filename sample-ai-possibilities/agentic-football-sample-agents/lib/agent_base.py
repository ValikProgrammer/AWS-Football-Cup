"""Base agent factory for AI soccer position agents."""

import concurrent.futures
import json
from typing import Callable
from strands import Agent
from strands.models import BedrockModel

from parsing import parse_commands
from state import summarize_state
from fallback import FallbackConfig, build_last_resort
from ticklog import log_tick, now_ms

try:  # ball axis fix (match ball = {x, y=height, z=lateral}); optional like the policy
    from norm import canonical_ball as _canonical_ball
except Exception:  # pragma: no cover
    _canonical_ball = None

# Per-tick LLM path only (enable_llm=true): total tick budget in seconds.
# The game allows ~2 s per decision; past this the policy command is sent.
LLM_DEADLINE_S = float(__import__('os').environ.get('LLM_DEADLINE_S', '1.8'))
_LLM_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix='afc-llm')

import os

try:  # the policy is optional: a missing file must not break the import
    import policy as _policy
    import orders as _orders
    _POLICY_PARAMS = _policy.load_params()
except Exception:  # pragma: no cover
    _policy = None
    _orders = None
    _POLICY_PARAMS = None

try:  # team strategy layer: optional, the plain policy works without it
    import strategies as _strategies
    import shared as _shared
    import match_memory as _memory
    import manager as _manager
except Exception:  # pragma: no cover
    _strategies = _shared = _memory = _manager = None

_CTX_LOGGED = False


def llm_enabled() -> bool:
    """team_config.json enable_llm; env ENABLE_LLM=1 forces it on."""
    if os.environ.get("ENABLE_LLM") == "1":
        return True
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "team_config.json")) as f:
            return bool(json.load(f).get("enable_llm", False))
    except Exception:
        return False


def policy_commands(game_state: dict, team_id: int, pid: int, label: str):
    """(commands, extra) from the deterministic policy. Raises on any failure."""
    if _policy is None or _POLICY_PARAMS is None:
        raise RuntimeError("policy unavailable")
    order = _orders.TRACKER.update(game_state, team_id, int(_POLICY_PARAMS.get("order_ticks", 20)))
    cands = _policy.decide(game_state, team_id, pid, label, _POLICY_PARAMS, order)
    cmd = dict(cands[0].command)
    cmd["teamId"] = team_id
    cmd["playerId"] = pid
    extra = {
        "order": order,
        "reason": cands[0].reason,
        "top": [{"t": c.command.get("commandType"), "s": round(c.score, 3), "r": c.reason} for c in cands[:3]],
    }
    return [cmd], extra


def session_info(context) -> dict:
    """session_id from the AgentCore context (defensive); context type logged once."""
    global _CTX_LOGGED
    out = {}
    try:
        sid = getattr(context, "session_id", None)
        if sid is None and isinstance(context, dict):
            sid = context.get("session_id") or context.get("sessionId")
        out["session_id"] = None if sid is None else str(sid)
        if not _CTX_LOGGED:
            _CTX_LOGGED = True
            out["ctx_type"] = type(context).__name__
    except Exception:
        pass
    return out


def strategy_commands(game_state: dict, team_id: int, pid: int, label: str):
    """(commands, extra) from the policy under the team strategy. Raises on any failure.

    strategy = active coach tactic order, else the shared (SSM) strategy.
    The GK also starts the manager (background thread, never waits).
    """
    if _policy is None or _POLICY_PARAMS is None or _strategies is None:
        raise RuntimeError("strategy layer unavailable")
    _memory.update(game_state, team_id)
    order = _orders.TRACKER.update(game_state, team_id, int(_POLICY_PARAMS.get("order_ticks", 20)))
    # Sticky coach strategy (S1..S6 / tactic words) beats the manager; an active
    # action order (C1..C6 / shoot, ...) is kept alongside it.
    coach = _orders.strategy_for(order) or _orders.strategy_for(getattr(_orders.TRACKER, "strategy_word", None))
    if coach:
        strategy, src = coach, "coach"
        if _orders.strategy_for(order):
            order = None
    else:
        strategy = _shared.read_strategy()
        src = "shared" if _shared.source() != "default" else "default"
    params = _strategies.apply(_POLICY_PARAMS, strategy)
    cands = _policy.decide(game_state, team_id, pid, label, params, order)
    cmd = dict(cands[0].command)
    cmd["teamId"] = team_id
    cmd["playerId"] = pid
    extra = {
        "order": order,
        "strategy": strategy,
        "strategy_source": src,
        "reason": cands[0].reason,
        "top": [{"t": c.command.get("commandType"), "s": round(c.score, 3), "r": c.reason} for c in cands[:3]],
    }
    if label == "GK":
        try:
            trig = _manager.maybe_trigger(game_state, team_id, label)
            if trig:
                extra["manager_trigger"] = trig
            if _manager._state.get("last_result"):
                extra["manager_last"] = _manager._state["last_result"]
        except Exception as e:
            extra["manager_error"] = repr(e)[:120]
    return [cmd], extra


def _fresh_agent(agent: Agent) -> Agent:
    """New Agent per tick on the same model: no history growth, no stdout token echo."""
    try:
        return Agent(model=agent.model, system_prompt=agent.system_prompt, callback_handler=None)
    except Exception:
        agent.messages.clear()
        return agent


def create_agent(system_prompt: str, model_id: str = "us.amazon.nova-micro-v1:0") -> Agent:
    """Create a Strands Agent with the given system prompt."""
    model = BedrockModel(model_id=model_id)
    return Agent(model=model, system_prompt=system_prompt)


def parse_prompt(payload) -> dict:
    """Parse the invocation payload's prompt, accepting either str or dict.

    The main path has always accepted both, but the error handler used to
    re-parse with a bare json.loads(), which raises TypeError on a dict. That
    exception was caught by the outer bare except, so the rule-based fallback
    was skipped entirely and control fell through to the last resort.
    """
    prompt = payload.get("prompt", "{}")
    if isinstance(prompt, dict):
        return prompt
    try:
        parsed = json.loads(prompt)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def team_id_from(prompt_data: dict, log, label: str) -> int:
    """teamId from the payload; HOME only as a loudly logged last resort.

    teamId is authoritative input, not a guessable default. Silently
    defaulting to 0 makes an away agent emit commands tagged HOME, which the
    server attributes to the opposing team — and leaves nothing in the logs
    to explain why the away side played as though it were the home side.
    """
    team_id = prompt_data.get("teamId")
    if team_id is None:
        log.error(
            f"{label}: payload contained no teamId; falling back to HOME (0). "
            "Commands this tick may be attributed to the wrong team."
        )
        return 0
    return team_id


def create_invoke_handler(
    app,
    agent: Agent,
    my_player_id: int,
    position_label: str,
    fallback_fn: Callable[[dict, int, int], list[dict]],
    fallback_cfg: FallbackConfig,
):
    """Create and register the @app.entrypoint invoke handler.

    Three layers of error handling, from best to worst:
      1. LLM response → parse into commands
      2. fallback_fn(game_state, team_id, my_player_id) → rule-based commands
      3. last-resort command from fallback_cfg → single safe command
    """
    log = app.logger
    last_resort = build_last_resort(fallback_cfg, my_player_id)

    @app.entrypoint
    async def invoke(payload, context):
        # Parsed once, outside the try, so the error handler below can reuse
        # it instead of re-parsing and throwing a second exception.
        t0 = now_ms()
        prompt_data = parse_prompt(payload)
        team_id = team_id_from(prompt_data, log, position_label)
        # Canonical ball axes before anything reads the state: state.py,
        # fallback.py, match_memory and the policy all expect y = lateral.
        # The tick log keeps the raw payload (logged as received).
        raw_prompt = prompt_data
        if _canonical_ball is not None and isinstance(prompt_data.get("gameState"), dict):
            try:
                prompt_data = dict(prompt_data, gameState=_canonical_ball(prompt_data["gameState"]))
            except Exception as ce:
                log.warn(f"{position_label} canonical_ball failed: {ce!r}")
        try:
            game_state = prompt_data.get("gameState", {})

            # Honor myPlayers from payload if present, otherwise use configured player ID
            my_players = prompt_data.get("myPlayers", [my_player_id])
            effective_pid = my_players[0] if my_players else my_player_id

            state_summary = summarize_state(
                game_state, team_id, effective_pid, position_label
            )
            if not llm_enabled():
                try:
                    commands, extra = strategy_commands(game_state, team_id, effective_pid, position_label)
                except Exception as se:
                    log.warn(f"{position_label} strategy path failed, plain policy: {se!r}")
                    commands, extra = policy_commands(game_state, team_id, effective_pid, position_label)
                    extra["strategy_error"] = repr(se)[:200]
                try:
                    extra.update(session_info(context))
                except Exception:
                    pass
                log_tick(position_label, team_id, effective_pid, "policy", now_ms() - t0, commands, raw_prompt,
                         extra=extra)
                yield json.dumps(commands)
                return

            log.info(f"{position_label} agent invoked for team {team_id}, controlling player {effective_pid}")

            fut = _LLM_POOL.submit(_fresh_agent(agent), state_summary)
            budget = max(0.05, LLM_DEADLINE_S - (now_ms() - t0) / 1000.0)
            try:
                response = fut.result(timeout=budget)
            except concurrent.futures.TimeoutError:
                log.warn(f"{position_label} LLM over {LLM_DEADLINE_S}s deadline, sending policy command")
                commands, extra = policy_commands(game_state, team_id, effective_pid, position_label)
                log_tick(position_label, team_id, effective_pid, "policy", now_ms() - t0, commands, raw_prompt,
                         error="llm_timeout", extra=extra)
                yield json.dumps(commands)
                return
            response_text = str(response)

            def on_recovered(raw: str) -> None:
                # The model wrote Python-flavoured JSON (usually `True`/`False`/`None`).
                # We recovered it rather than dropping the command and falling back —
                # logged so you can see how often your model does this.
                log.warn(f"{position_label} recovered malformed JSON from the model: {raw[:200]}")

            commands = parse_commands(response_text, team_id, effective_pid, on_recovered)

            if commands:
                log.info(f"LLM returned {len(commands)} commands: "
                         f"{[c.get('commandType') for c in commands]}")
                log_tick(position_label, team_id, effective_pid, "llm", now_ms() - t0, commands, raw_prompt)
                yield json.dumps(commands)
            else:
                log.warn(f"LLM parse failed, using fallback. Response: {response_text[:200]}")
                commands = fallback_fn(game_state, team_id, effective_pid)
                log.info(f"Fallback returned {len(commands)} commands")
                log_tick(position_label, team_id, effective_pid, "fallback", now_ms() - t0, commands, raw_prompt,
                         error="parse_failed: " + response_text[:200])
                yield json.dumps(commands)

        except Exception as e:
            log.error(f"{position_label} agent error: {e}")
            try:
                # Reuse the payload parsed at entry. Re-parsing here is what
                # broke this path: json.loads() on an already-decoded dict
                # raises TypeError, the bare except below swallowed it, and
                # the rule-based fallback never ran.
                my_players = prompt_data.get("myPlayers", [my_player_id])
                effective_pid = my_players[0] if my_players else my_player_id
                commands = fallback_fn(
                    prompt_data.get("gameState", {}),
                    team_id,
                    effective_pid,
                )
                log_tick(position_label, team_id, effective_pid, "fallback", now_ms() - t0, commands, raw_prompt,
                         error=repr(e))
                yield json.dumps(commands)
            except Exception as fallback_error:
                log.error(f"{position_label} fallback also failed: {fallback_error}")
                cmd = dict(last_resort)
                # team_id came from the payload at entry, so the last resort
                # is no longer hardcoded to HOME.
                cmd["teamId"] = team_id
                log_tick(position_label, team_id, my_player_id, "last_resort", now_ms() - t0, [cmd], raw_prompt,
                         error=repr(fallback_error))
                yield json.dumps([cmd])

    return invoke
