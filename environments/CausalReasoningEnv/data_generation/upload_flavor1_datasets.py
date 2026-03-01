"""One-off script to upload flavor1 datasets to HuggingFace Hub."""

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "datasets",
# ]
# ///

from datasets import load_from_disk, DatasetDict

train = load_from_disk("environments/CausalReasoningEnv/datasets/flavor1/train")
eval_ = load_from_disk("environments/CausalReasoningEnv/datasets/flavor1/eval")

print(f"Train: {len(train)} rows, columns: {train.column_names}")
print(f"Eval:  {len(eval_)} rows, columns: {eval_.column_names}")

DatasetDict({"train": train, "test": eval_}).push_to_hub(
    "irfanjamil/causal-reasoning-flavor1",
    private=False,
)

print("\nDone. Visit: https://huggingface.co/datasets/irfanjamil/causal-reasoning-flavor1")
