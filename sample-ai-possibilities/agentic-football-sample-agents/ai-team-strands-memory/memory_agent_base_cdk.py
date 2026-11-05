"""Memory-aware agent factory for AI soccer position agents (CDK/npm-CLI flow).

Sibling of memory_agent_base.py with the same public API, for deployments
made with the npm AgentCore CLI (`python deploy_all.py`). There the memory
resource is declared in agentcore/agentcore.json ("team_memory") and created
by the CDK app, which injects its ID into every runtime as
MEMORY_TEAM_MEMORY_ID — no MEMORY_ID env var is passed at deploy time.

deploy_all.py stages this file into each agent directory AS
memory_agent_base.py, so src/main.py imports it unchanged. The original
memory_agent_base.py remains the module staged by the legacy deploy-all.sh
flow.
"""

import os
from strands import Agent
from strands.models import BedrockModel
from bedrock_agentcore.memory.integrations.strands.session_manager import AgentCoreMemorySessionManager
from bedrock_agentcore.memory.integrations.strands.config import AgentCoreMemoryConfig


def create_memory_agent(
    system_prompt: str,
    player_id: int,
    position_label: str,
    model_id: str = "us.amazon.nova-micro-v1:0",
) -> Agent:
    """Create a Strands Agent backed by AgentCore Memory (STM).

    Required env vars:
      MEMORY_TEAM_MEMORY_ID — injected by the CDK stack for the
                              "team_memory" resource in agentcore.json
      TEAM_ID               — used as actor_id and session_id prefix
                              (optional; each team deploys into its own
                              account, so the default cannot collide)
    """
    memory_id = os.environ.get("MEMORY_TEAM_MEMORY_ID")
    team_id = os.environ.get("TEAM_ID", "default-team")

    if not memory_id:
        raise RuntimeError(
            "MEMORY_TEAM_MEMORY_ID environment variable is required. "
            "It is injected automatically when the agent is deployed via "
            "`python deploy_all.py` with the team_memory resource declared "
            "in agentcore/agentcore.json."
        )

    session_manager = AgentCoreMemorySessionManager(
        agentcore_memory_config=AgentCoreMemoryConfig(
            memory_id=memory_id,
            session_id=f"match-{team_id}-{position_label}",
            actor_id=f"{team_id}-{position_label}",
        ),
        # The CDK flow injects no region env var of its own; the AgentCore
        # runtime provides AWS_REGION. None lets boto3 resolve as a last resort.
        region_name=os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION"),
    )

    model = BedrockModel(model_id=model_id)
    return Agent(
        model=model,
        system_prompt=system_prompt,
        session_manager=session_manager,
    )
