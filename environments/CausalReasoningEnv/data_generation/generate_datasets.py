"""Regenerate and optionally upload the CausalATE dataset.

Usage:
  python data_generation/generate_datasets.py --n 1000 --save-local
  python data_generation/generate_datasets.py --n 1000 --push-hub
  python data_generation/generate_datasets.py --n 200 --save-local  # quick check
"""
import argparse
import sys
import os

# Allow running from the environments/CausalReasoningEnv/ directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_generation.gen import generate_problems, build_dataset


def main():
    parser = argparse.ArgumentParser(description="Regenerate the CausalATE benchmark dataset.")
    parser.add_argument("--n", type=int, default=1000, help="Number of problems to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save-local", action="store_true", help="Save dataset to disk")
    parser.add_argument("--push-hub", action="store_true", help="Push dataset to HuggingFace Hub")
    parser.add_argument("--output-dir", type=str, default="datasets/causal_ate/", help="Local output directory")
    args = parser.parse_args()

    print(f"Generating {args.n} problems (seed={args.seed})...")
    problems = generate_problems(n=args.n, seed=args.seed)

    # Print distribution
    from collections import Counter
    counts = Counter(p["problem_type"] for p in problems)
    print("Problem type distribution:")
    for ptype, count in sorted(counts.items()):
        print(f"  {ptype}: {count} ({100 * count / len(problems):.1f}%)")

    ds = build_dataset(problems)
    print(f"Built dataset: {len(ds)} examples")

    if args.save_local:
        ds.save_to_disk(args.output_dir)
        print(f"Saved {len(ds)} examples to {args.output_dir}")

    if args.push_hub:
        ds.push_to_hub("irfanjamil/causal-reasoning-ate")
        print("Pushed to Hub: irfanjamil/causal-reasoning-ate")

    if not args.save_local and not args.push_hub:
        print("No output action specified. Use --save-local or --push-hub.")


if __name__ == "__main__":
    main()
