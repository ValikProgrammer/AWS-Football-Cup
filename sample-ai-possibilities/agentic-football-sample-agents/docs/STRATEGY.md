# Team strategy

The deterministic policy (`lib/policy.py`) runs with one team-wide strategy.
A strategy is a preset of `lib/params.json` overrides (`lib/strategies.py`).

## Strategies

| Name | What changes |
|------|--------------|
| BALANCED | Nothing (plain `params.json`). Default. |
| POSSESSION | More pass bonus, more risk penalty, shorter shoot range. |
| COUNTER | Deep line, strong forward-pass weight, earlier THROUGH/AERIAL passes, no high chase. |
| HIGH_PRESS | Two pressers, high line, press intensity 1.0, sprint to lower stamina. |
| LOW_BLOCK | Low line, one presser that presses only in our half, clear the ball under pressure. |
| ALL_OUT | High line, long shoot range, low shot corridor, max shot bonus. |

An unknown name means BALANCED.

## Per tick (every agent)

1. `match_memory.update()` adds the tick to the in-process match log.
2. `orders.TRACKER` reads the coach chat. A tactic word gives a coach strategy for `order_ticks` ticks.
3. Effective strategy = coach strategy if active, else `shared.read_strategy()`.
4. `strategies.apply(params, strategy)` and `policy.decide()`.
5. The GK calls `manager.maybe_trigger()`. It returns at once.
6. The tick log gets `strategy`, `strategy_source` (`coach` / `shared` / `default`) and `session_id`.

If the strategy path fails, the agent uses the plain policy, then the stock fallback.

## Coach words

Old one-word orders stay: `shoot`, `attack`, `defend`, `press`, `mark`, `pass`.
New tactic words (typos are accepted, for example `cuonter`, `posession`, `hihgpress`, `alout`):

- `balanced` -> BALANCED
- `possession` -> POSSESSION
- `counter` -> COUNTER
- `highpress`, `high press` -> HIGH_PRESS
- `lowblock`, `low block`, `park` -> LOW_BLOCK
- `allout`, `all out` -> ALL_OUT

## Shared state (`lib/shared.py`)

- SSM parameter `/afc/team/strategy`, JSON `{"strategy","reason","seq","ts","by"}`.
- Read cache 1 s. Timeout 300 ms, one attempt. After an error: keep the last value, no SSM call for 5 s.
- `SHARED_STATE=0`: no SSM. Each process uses only its own value.
- The runtime role needs `ssm:GetParameter` (all agents) and `ssm:PutParameter` (GK).

## Manager (`lib/manager.py`, GK only)

- Trigger: kick-off, each goal, every 30 s of game time. One background thread at a time.
- LLM: `us.amazon.nova-lite-v1:0`, temperature 0.2, max 150 tokens, new Agent per call.
- Input: `match_memory.report()` (max 12 lines), strategy list, score and time.
- Output: `{"strategy": "...", "reason": "..."}`. Invalid or later than 10 s: keep the current strategy.
- LLM error: rules. Losing after 150 s -> ALL_OUT. Leading after 230 s -> LOW_BLOCK. 2+ goals conceded from counters -> LOW_BLOCK.
- Hysteresis: max one switch per 20 s of game time, except after a goal.
- `ENABLE_MANAGER=0` disables it. It does not depend on `enable_llm`.

## Manual override

```sh
aws ssm put-parameter --name /afc/team/strategy --type String --overwrite \
  --value '{"strategy":"LOW_BLOCK","reason":"manual","seq":999,"ts":0}'
```

The agents use it within about 1 s. The manager can replace it at its next trigger.
Set `ENABLE_MANAGER=0` on the GK runtime to keep a manual value.
