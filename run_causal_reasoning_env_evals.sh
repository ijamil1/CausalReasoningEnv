#!/bin/bash


models=(
  "openai/gpt-4.1-mini"
  "openai/gpt-5.1-codex"
  "qwen/qwen3-30b-a3b-thinking-2507"
  "qwen/qwen3-30b-a3b-instruct-2507"
  "Qwen/Qwen3-30B-A3B-Instruct-2507:o3iqo2s22s20ec9lezol5v6g"
  "Qwen/Qwen3-30B-A3B-Thinking-2507:gko6mk1n7klelc2mipr7ve61"
  )

EVALS_DIR="./environments/CausalReasoningEnv/outputs/evals/"
set -a; source .env; set +a
for model in "${models[@]}"; do
  short_name="${model#*/}"
  if ls "$EVALS_DIR" 2>/dev/null | grep -q "$short_name"; then
    echo "Skipping $model (already has eval results for $short_name)"
  else
    echo "Running eval with model: $model"
    prime eval run irfanjamil/CausalReasoningEnv@0.5.7 -n 100 -r 3 -m "$model"
  fi
  echo ""
done
