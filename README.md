# CausalReasoningEnv_1

Single-turn causal inference environment where a model must identify the minimal adjustment set for a given DAG. Built with Prime Intellect's [verifiers](https://github.com/PrimeIntellect-ai/verifiers) framework.

## Setup

```bash
prime env install CausalReasoningEnv_1
```

## Usage

```bash
# Install locally
prime env install CausalReasoningEnv_1

# Run evaluation
prime eval run CausalReasoningEnv_1

# Push to Prime Hub
prime env push -p ./environments/CausalReasoningEnv_1
```

## Environment

| Environment | Description |
| ----------- | ----------- |
| [CausalReasoningEnv_1](environments/CausalReasoningEnv_1/) | Single-turn environment where the model receives a randomly generated DAG (as a node/edge list and rendered image) and must identify the minimal adjustment set that blocks all backdoor paths from a treatment node X to an outcome node Y. Problems are stratified by difficulty (standard, collider, ancestor). |
