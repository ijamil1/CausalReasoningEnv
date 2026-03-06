"""Regenerate and optionally upload the CausalATE dataset.

Usage:
  python data_generation/generate_datasets.py --n 350 --save-local
  python data_generation/generate_datasets.py --n 350 --push-hub
  python data_generation/generate_datasets.py --n 200 --save-local  # quick check
"""
import argparse
import random
import sys
import os
from collections import Counter, defaultdict

# Allow running from the environments/CausalReasoningEnv/ directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_generation.gen import generate_problems, build_dataset



def stratified_split(
    problems: list[dict],
    train_size: int = 250,
    eval_size: int = 100,
    max_frac_diff: float = 0.05,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Split problems into train/eval with stratification by problem type.

    Allocates each type proportionally (train_size / total), then verifies
    that no type's fraction differs by more than max_frac_diff between splits.
    Raises ValueError if the constraint cannot be satisfied.
    """
    rng = random.Random(seed)
    train_frac = train_size / (train_size + eval_size)

    by_type = defaultdict(list)
    for p in problems:
        by_type[p["problem_type"]].append(p)

    train, eval_ = [], []
    for ptype, group in by_type.items():
        rng.shuffle(group)
        n_train = round(len(group) * train_frac)
        train.extend(group[:n_train])
        eval_.extend(group[n_train:])

    # Verify fraction constraint
    n_train, n_eval = len(train), len(eval_)
    for ptype in by_type:
        tf = sum(1 for p in train if p["problem_type"] == ptype) / n_train
        ef = sum(1 for p in eval_ if p["problem_type"] == ptype) / n_eval
        if abs(tf - ef) > max_frac_diff:
            raise ValueError(
                f"Stratification constraint violated for '{ptype}': "
                f"train={tf:.3f}, eval={ef:.3f}, diff={abs(tf-ef):.3f} > {max_frac_diff}"
            )

    rng.shuffle(train)
    rng.shuffle(eval_)
    return train, eval_


def print_distribution(label: str, problems: list[dict]) -> None:
    n = len(problems)
    counts = Counter(p["problem_type"] for p in problems)
    print(f"\n{label} ({n} problems):")
    for ptype, count in sorted(counts.items()):
        print(f"  {ptype}: {count} ({100 * count / n:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Regenerate the CausalATE benchmark dataset.")
    parser.add_argument("--n", type=int, default=350, help="Total problems to generate before dedup")
    parser.add_argument("--train-size", type=int, default=250, help="Target train split size")
    parser.add_argument("--eval-size", type=int, default=100, help="Target eval split size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save-local", action="store_true", help="Save dataset to disk")
    parser.add_argument("--push-hub", action="store_true", help="Push dataset to HuggingFace Hub")
    parser.add_argument("--output-dir", type=str, default="datasets/causal_ate/", help="Local output directory")
    args = parser.parse_args()

    print(f"Generating {args.n} problems (seed={args.seed})...")
    problems = generate_problems(n=args.n, seed=args.seed)
    print(f"Generated {len(problems)} problems.")

    print_distribution("Full pool", problems)

    train_problems, eval_problems = stratified_split(
        problems,
        train_size=args.train_size,
        eval_size=args.eval_size,
        seed=args.seed,
    )
    print_distribution("Train split", train_problems)
    print_distribution("Eval split", eval_problems)

    train_ds = build_dataset(train_problems)
    eval_ds = build_dataset(eval_problems)
    print(f"\nTrain dataset: {len(train_ds)} examples")
    print(f"Eval dataset:  {len(eval_ds)} examples")

    if args.save_local:
        train_ds.save_to_disk(os.path.join(args.output_dir, "train"))
        eval_ds.save_to_disk(os.path.join(args.output_dir, "eval"))
        print(f"Saved to {args.output_dir}{{train,eval}}")

    if args.push_hub:
        from datasets import DatasetDict
        DatasetDict({"train": train_ds, "eval": eval_ds}).push_to_hub("irfanjamil/causal-reasoning-ate")
        print("Pushed to Hub: irfanjamil/causal-reasoning-ate")

    if not args.save_local and not args.push_hub:
        print("\nNo output action specified. Use --save-local or --push-hub.")


if __name__ == "__main__":
    main()
