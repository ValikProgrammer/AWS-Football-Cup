# Agent rules for this repository

## Language
- Write in English everywhere: code, comments, docstrings, commit messages, docs, logs, and replies to the user.

## Layout
- Team code: `sample-ai-possibilities/agentic-football-sample-agents/`.
- Deployed team: `ai-team-strands-extremely-aggressive/` (runtimes `agg_{gk,def,mid,fwd1,fwd2}_agent`).
- Deploy with `deploy-all.sh` in that folder. It copies only `src/main.py`, `lib/`, `requirements.txt`. Put new modules in `lib/`.
- Do not rename runtimes (`name:` in `.bedrock_agentcore.yaml.template`): that creates new ARNs.

## Working rules
- Never leave the team undeployable. One change per deploy. All tests green before deploy.
- Record each experiment in `docs/EXPERIMENTS.md`: change, opponent, score, conclusion.
