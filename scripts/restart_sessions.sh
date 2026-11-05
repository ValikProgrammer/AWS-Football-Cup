#!/usr/bin/env bash
# Stop stale AgentCore sessions so the next call loads the newest deployed version.
# The game reuses one session id per scrimmage slot (e.g. scrimmage-drill-<uuid>-pos1);
# a running session keeps the code it started with, even after a deploy.
# Only sessions whose id starts with $PREFIX (default "scrimmage") are stopped,
# so live tournament matches are not interrupted. PREFIX="" stops all found sessions.
set -uo pipefail
REGION="${AWS_DEFAULT_REGION:-us-east-1}"
PREFIX="${PREFIX-scrimmage}"
NAMES="${*:-agg_gk_agent agg_def_agent agg_mid_agent agg_fwd1_agent agg_fwd2_agent}"

for name in $NAMES; do
  arn=$(aws bedrock-agentcore-control list-agent-runtimes --region "$REGION" \
        --query "agentRuntimes[?agentRuntimeName=='$name'].agentRuntimeArn | [0]" --output text)
  [ "$arn" = "None" ] && { echo "$name: runtime not found"; continue; }
  id="${arn##*/}"
  sessions=$(aws logs describe-log-streams --region "$REGION" \
      --log-group-name "/aws/bedrock-agentcore/runtimes/${id}-DEFAULT" \
      --order-by LastEventTime --descending --max-items 10 \
      --query 'logStreams[].logStreamName' --output text 2>/dev/null \
    | tr '\t' '\n' | sed -n 's/.*\[runtime-logs-\(.*\)\].*/\1/p' | sort -u)
  [ -z "$sessions" ] && { echo "$name: no sessions in logs"; continue; }
  for s in $sessions; do
    case "$s" in "$PREFIX"*) ;; *) echo "$name: skip $s"; continue ;; esac
    code=$(aws bedrock-agentcore stop-runtime-session --region "$REGION" --agent-runtime-arn "$arn" \
           --runtime-session-id "$s" --qualifier DEFAULT --query statusCode --output text 2>&1 | tail -1)
    case "$code" in *ResourceNotFoundException*) code="already stopped" ;; esac
    echo "$name: stop $s -> $code"
  done
done
