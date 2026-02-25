#!/bin/bash

models=(
    "qwen/qwen3-vl-235b-a22b-instruct"
    "qwen/qwen3-vl-235b-a22b-thinking"
    "qwen/qwen3-vl-30b-a3b-instruct"
    "qwen/qwen3-vl-30b-a3b-thinking"
    "qwen/qwen3-vl-8b-instruct"   
)

EVALS_DIR="./environments/CausalReasoningEnv_1/outputs/evals"

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
