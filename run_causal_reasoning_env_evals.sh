#!/bin/bash


models=(
    "Qwen/Qwen3-VL-32B-Instruct"
    "Qwen/Qwen3-VL-8B-Instruct"
)

EVALS_DIR="./environments/CausalReasoningEnv_1/outputs/evals"
source .env
for model in "${models[@]}"; do
  short_name="${model#*/}"
  if ls "$EVALS_DIR" 2>/dev/null | grep -q "$short_name"; then
    echo "Skipping $model (already has eval results for $short_name)"
  else
    echo "Running eval with model: $model"
    prime eval run irfanjamil/CausalReasoningEnv_1 -n 100 -r 3 -m "$model"
  fi
  echo ""
done
