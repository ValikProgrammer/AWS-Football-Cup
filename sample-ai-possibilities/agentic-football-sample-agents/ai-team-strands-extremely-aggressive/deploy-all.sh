#!/bin/bash
set -e

# ============================================================================
# Deploy all 5 AI Team (Extremely Aggressive) agents to Bedrock AgentCore
# ============================================================================
#
# Usage:
#   AWS_PROFILE=your-profile ./deploy-all.sh          # deploy all
#   AWS_PROFILE=your-profile ./deploy-all.sh ai-gk    # deploy one agent
#   DEPLOY_PARALLEL=0 ./deploy-all.sh                 # old one-by-one mode
#
# How it works:
#   1. Creates a _build/<agent>/ staging directory for each agent
#   2. Copies the agent's src/ + shared lib/ + requirements.txt into it
#   3. Generates .bedrock_agentcore.yaml from the agent's template
#   4. Deploys all staging directories in parallel (one log per agent in _deploy_logs/)
#   5. Cleans up _build/ when done
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/_build"

AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"
export AWS_DEFAULT_REGION

ALL_AGENTS=("ai-gk" "ai-def" "ai-mid" "ai-fwd1" "ai-fwd2")

if [ -n "$1" ]; then
  AGENTS=("$1")
else
  AGENTS=("${ALL_AGENTS[@]}")
fi

echo "=========================================="
echo "  AI Team (Extremely Aggressive) — Deploy"
echo "=========================================="
echo ""

echo "Checking prerequisites..."

if ! command -v agentcore &> /dev/null; then
  echo "ERROR: 'agentcore' CLI not found."
  echo "Install: pip install bedrock-agentcore-starter-toolkit"
  exit 1
fi
echo "  agentcore CLI: OK"

if ! command -v rsync &> /dev/null; then
  echo "ERROR: 'rsync' not found."
  exit 1
fi
echo "  rsync: OK"

if ! command -v aws &> /dev/null; then
  echo "ERROR: 'aws' CLI not found."
  exit 1
fi
echo "  aws CLI: OK"

AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text 2>/dev/null) || {
  echo "ERROR: No valid AWS credentials."
  exit 1
}
export AWS_ACCOUNT_ID
echo "  AWS Account: $AWS_ACCOUNT_ID"
echo "  AWS Region:  $AWS_DEFAULT_REGION"
echo ""

LOG_DIR="$SCRIPT_DIR/_deploy_logs"
mkdir -p "$LOG_DIR"
PARALLEL="${DEPLOY_PARALLEL:-1}"
PIDS=()

cleanup() {
  # Stop running deploys on Ctrl+C / error, then remove staging dirs.
  for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
  echo ""
  echo "Cleaning up build directory..."
  rm -rf "$BUILD_DIR"
}
trap cleanup EXIT

DEPLOYED=()
FAILED=()
STAGED=()

# --- 1. Stage every agent (fast, sequential) --------------------------------
for agent in "${AGENTS[@]}"; do
  AGENT_SRC="$SCRIPT_DIR/$agent"
  STAGE="$BUILD_DIR/$agent"

  if [ ! -d "$AGENT_SRC" ]; then
    echo "  ERROR: Agent directory not found: $AGENT_SRC"
    FAILED+=("$agent")
    continue
  fi

  rm -rf "$STAGE"
  mkdir -p "$STAGE/src"

  cp "$AGENT_SRC/src/main.py" "$STAGE/src/main.py"
  rsync -a --exclude='__pycache__' "$SCRIPT_DIR/../lib/" "$STAGE/lib/"
  cp "$AGENT_SRC/requirements.txt" "$STAGE/requirements.txt"

  sed \
    -e "s|\${AWS_ACCOUNT_ID}|$AWS_ACCOUNT_ID|g" \
    -e "s|\${AWS_DEFAULT_REGION}|$AWS_DEFAULT_REGION|g" \
    "$AGENT_SRC/.bedrock_agentcore.yaml.template" > "$STAGE/.bedrock_agentcore.yaml"

  STAGED+=("$agent")
  echo "  Staged: $agent"
done
echo ""

deploy_one() {
  (cd "$BUILD_DIR/$1" && agentcore deploy --auto-update-on-conflict) > "$LOG_DIR/$1.log" 2>&1
}

# --- 2. Deploy (parallel by default) ----------------------------------------
START=$(date +%s)
if [ "$PARALLEL" = "1" ] && [ ${#STAGED[@]} -gt 1 ]; then
  echo "Deploying ${#STAGED[@]} agents in parallel (logs: $LOG_DIR/<agent>.log)..."
  for agent in "${STAGED[@]}"; do
    deploy_one "$agent" &
    PIDS+=("$!")
  done
  for i in "${!STAGED[@]}"; do
    agent="${STAGED[$i]}"
    if wait "${PIDS[$i]}"; then
      echo "  ✅ $agent: DEPLOYED ($(( $(date +%s) - START ))s)"
      DEPLOYED+=("$agent")
    else
      echo "  ❌ $agent: FAILED — last lines of $LOG_DIR/$agent.log:"
      tail -n 15 "$LOG_DIR/$agent.log" | sed 's/^/      /'
      FAILED+=("$agent")
    fi
  done
  PIDS=()
else
  for agent in "${STAGED[@]}"; do
    echo "  Deploying: $agent (log: $LOG_DIR/$agent.log)"
    if deploy_one "$agent"; then
      echo "  ✅ $agent: DEPLOYED ($(( $(date +%s) - START ))s)"
      DEPLOYED+=("$agent")
    else
      echo "  ❌ $agent: FAILED — last lines of $LOG_DIR/$agent.log:"
      tail -n 15 "$LOG_DIR/$agent.log" | sed 's/^/      /'
      FAILED+=("$agent")
    fi
  done
fi
echo ""

echo "=========================================="
echo "  Deployment Summary"
echo "=========================================="
echo ""
echo "  Deployed: ${DEPLOYED[*]:-none}"
echo "  Failed:   ${FAILED[*]:-none}"
echo "  Account:  $AWS_ACCOUNT_ID"
echo "  Region:   $AWS_DEFAULT_REGION"
echo "  Time:     $(( $(date +%s) - START ))s"
echo ""

if [ ${#FAILED[@]} -gt 0 ]; then
  echo "Some agents failed to deploy. Check the output above."
  exit 1
fi

echo "All agents deployed successfully."
