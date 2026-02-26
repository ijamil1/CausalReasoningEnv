"""analyze_dataset.py — Diagnostic script for CausalReasoningEnv_1 datasets.

Loads the saved train and eval splits and prints distributions of:
  - num_nodes
  - size of the minimal adjustment set
  - num_parents_X
  - problem_type (standard / ancestor / collider)
  - ratio = |min_set| / |parents_X|

Usage:
    uv run --with datasets python analyze_dataset.py
"""

import json
from collections import Counter
from pathlib import Path

from datasets import load_from_disk

_DATASET_DIR = Path(__file__).parent / "environments" / "CausalReasoningEnv_1" / "datasets"
_TRAIN_PATH = _DATASET_DIR / "train"
_EVAL_PATH = _DATASET_DIR / "eval"


def analyze(split_name: str, dataset) -> None:
    print(f"\n{'='*60}")
    print(f"  {split_name}  ({len(dataset)} examples)")
    print(f"{'='*60}")

    num_nodes_counter: Counter = Counter()
    set_size_counter: Counter = Counter()
    parents_counter: Counter = Counter()
    type_counter: Counter = Counter()
    ratio_buckets: list[float] = []
    ratio_lt1_count = 0

    for row in dataset:
        info = json.loads(row["info"])
        n_nodes = info["num_nodes"]
        set_size = len(info["minimal_adjustment_set"])
        n_parents = info.get("num_parents_X") or 0
        ptype = info.get("problem_type") or "unknown"

        num_nodes_counter[n_nodes] += 1
        set_size_counter[set_size] += 1
        parents_counter[n_parents] += 1
        type_counter[ptype] += 1

        if n_parents > 0:
            ratio = set_size / n_parents
            ratio_buckets.append(ratio)
            if ratio < 1.0:
                ratio_lt1_count += 1

    total = len(dataset)

    print(f"\nnum_nodes distribution:")
    for k in sorted(num_nodes_counter):
        bar = "#" * num_nodes_counter[k]
        print(f"  {k:3d}: {num_nodes_counter[k]:4d}  {bar}")

    print(f"\nmin_set size distribution:")
    for k in sorted(set_size_counter):
        bar = "#" * set_size_counter[k]
        print(f"  {k:3d}: {set_size_counter[k]:4d}  {bar}")

    print(f"\nnum_parents_X distribution:")
    for k in sorted(parents_counter):
        bar = "#" * parents_counter[k]
        print(f"  {k:3d}: {parents_counter[k]:4d}  {bar}")

    print(f"\nproblem_type distribution:")
    for ptype in sorted(type_counter):
        count = type_counter[ptype]
        pct = 100 * count / total
        print(f"  {ptype:<10s}: {count:4d}  ({pct:.1f}%)")

    if ratio_buckets:
        n_lt1 = ratio_lt1_count
        pct_lt1 = 100 * n_lt1 / total
        avg_ratio = sum(ratio_buckets) / len(ratio_buckets)
        min_ratio = min(ratio_buckets)
        max_ratio = max(ratio_buckets)
        print(f"\nratio = |min_set| / |parents_X|:")
        print(f"  ratio < 1.0 : {n_lt1:4d}  ({pct_lt1:.1f}%)")
        print(f"  ratio = 1.0 : {total - n_lt1:4d}  ({100 - pct_lt1:.1f}%)")
        print(f"  min={min_ratio:.3f}  avg={avg_ratio:.3f}  max={max_ratio:.3f}")

        type_lt1: Counter = Counter()
        for row in dataset:
            info = json.loads(row["info"])
            n_parents = info.get("num_parents_X") or 0
            set_size = len(info["minimal_adjustment_set"])
            ptype = info.get("problem_type") or "unknown"
            if n_parents > 0 and set_size < n_parents:
                type_lt1[ptype] += 1
        if type_lt1:
            print(f"\n  Among ratio<1 examples:")
            for ptype, count in sorted(type_lt1.items()):
                pct = 100 * count / n_lt1
                print(f"    {ptype:<10s}: {count:4d}  ({pct:.1f}% of ratio<1)")


if __name__ == "__main__":
    print("CausalReasoningEnv_1 — Dataset Diagnostics")

    if not _TRAIN_PATH.exists() or not _EVAL_PATH.exists():
        print(f"ERROR: datasets not found at {_DATASET_DIR}")
        print("Run: uv run python environments/CausalReasoningEnv_1/CausalReasoningEnv_1.py")
        raise SystemExit(1)

    train_ds = load_from_disk(str(_TRAIN_PATH))
    eval_ds = load_from_disk(str(_EVAL_PATH))

    analyze("TRAIN", train_ds)
    analyze("EVAL", eval_ds)
    print()
