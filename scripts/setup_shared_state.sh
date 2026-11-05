#!/usr/bin/env bash
# One-time setup of the shared team state in SSM Parameter Store.
#  1. Creates parameter /afc/team/strategy (initial value BALANCED).
#  2. Adds inline policy "AfcSharedState" to the IAM role of each of the 5 runtimes:
#     only ssm:GetParameter + ssm:PutParameter on parameter/afc/*.
# Undo: aws iam delete-role-policy --role-name <role> --policy-name AfcSharedState
#       aws ssm delete-parameter --name /afc/team/strategy
set -euo pipefail
REGION="${AWS_DEFAULT_REGION:-us-east-1}"
ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
RUNTIMES=(agg_gk_agent-xNoJIy3Qfj agg_def_agent-LeMQg44C2C agg_mid_agent-Vl8yz18uO5 agg_fwd1_agent-tYVE7o9LaU agg_fwd2_agent-2Zc8eYCPUO)

aws ssm put-parameter --region "$REGION" --name /afc/team/strategy --type String \
  --value '{"strategy":"BALANCED","reason":"init","seq":0,"ts":0}' --overwrite >/dev/null
echo "SSM parameter /afc/team/strategy: OK"

POLICY=$(cat <<EOF
{"Version":"2012-10-17","Statement":[{"Sid":"AfcSharedState","Effect":"Allow",
"Action":["ssm:GetParameter","ssm:PutParameter"],
"Resource":"arn:aws:ssm:${REGION}:${ACCOUNT}:parameter/afc/*"}]}
EOF
)

for id in "${RUNTIMES[@]}"; do
  role="$(aws bedrock-agentcore-control get-agent-runtime --region "$REGION" --agent-runtime-id "$id" \
          --query roleArn --output text | sed 's#.*/##')"
  aws iam put-role-policy --role-name "$role" --policy-name AfcSharedState --policy-document "$POLICY"
  echo "$id -> $role: AfcSharedState OK"
done
