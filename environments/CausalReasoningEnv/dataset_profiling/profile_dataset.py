"""Profile the train and eval splits of irfanjamil/causal-reasoning-ate.

Outputs per split:
  - Counts by problem_type
  - Counts by identification_method
  - Counts by number of observed nodes
  - Counts by (problem_type, num_nodes) combo
  - ATE/LATE distribution per identification method (percentiles + histogram)
"""

import json
from collections import Counter, defaultdict

from datasets import load_dataset


# ── helpers ──────────────────────────────────────────────────────────────────

def parse_info(row: dict) -> dict:
    info = row["info"]
    return json.loads(info) if isinstance(info, str) else info


def ate_bin_label(val: float, width: float = 0.5) -> str:
    lo = round((val // width) * width, 2)
    hi = round(lo + width, 2)
    return f"[{lo:+.2f}, {hi:+.2f})"


def percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    idx = (len(sorted_vals) - 1) * p / 100
    lo, hi = int(idx), min(int(idx) + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (idx - lo)


def print_counter(counter: Counter, title: str, total: int) -> None:
    print(f"\n  {title}")
    print(f"  {'─' * 50}")
    for key, cnt in sorted(counter.items(), key=lambda x: -x[1]):
        pct = 100 * cnt / total if total else 0
        print(f"    {str(key):<40s}  {cnt:5d}  ({pct:5.1f}%)")


def print_ate_distribution(vals: list[float], method: str) -> None:
    if not vals:
        print(f"    (no data)")
        return
    sv = sorted(vals)
    n = len(sv)
    print(f"    n={n}  min={sv[0]:+.4f}  p10={percentile(sv,10):+.4f}  "
          f"p25={percentile(sv,25):+.4f}  median={percentile(sv,50):+.4f}  "
          f"p75={percentile(sv,75):+.4f}  p90={percentile(sv,90):+.4f}  max={sv[-1]:+.4f}")

    # Histogram bins of width 0.5 across [-4, 4]
    bin_width = 0.5
    bins: dict[str, int] = {}
    for v in vals:
        label = ate_bin_label(v, bin_width)
        bins[label] = bins.get(label, 0) + 1

    bar_max = max(bins.values()) if bins else 1
    bar_width = 30
    print(f"    {'Bin':<22s}  {'Count':>6s}  {'Bar'}")
    print(f"    {'─'*22}  {'─'*6}  {'─'*bar_width}")
    for label in sorted(bins.keys()):
        cnt = bins[label]
        bar = "█" * round(bar_width * cnt / bar_max)
        print(f"    {label:<22s}  {cnt:6d}  {bar}")


# ── main profiling ────────────────────────────────────────────────────────────

def profile_split(ds, split_name: str) -> None:
    print(f"\n{'═' * 60}")
    print(f"  SPLIT: {split_name.upper()}  ({len(ds)} rows)")
    print(f"{'═' * 60}")

    total = len(ds)

    ptype_counter: Counter = Counter()
    method_counter: Counter = Counter()
    num_nodes_counter: Counter = Counter()
    ptype_nodes_counter: Counter = Counter()
    ate_by_method: dict[str, list[float]] = defaultdict(list)

    for row in ds:
        info = parse_info(row)

        ptype = info.get("problem_type", "unknown")
        ptype_counter[ptype] += 1

        methods = info.get("identification_methods", [])
        method = methods[0] if methods else "unknown"
        method_counter[method] += 1

        num_nodes = len(info.get("nodes", []))
        num_nodes_counter[num_nodes] += 1

        ptype_nodes_counter[(ptype, num_nodes)] += 1

        val = info.get("true_LATE") if ptype == "iv" else info.get("true_ATE")
        if val is not None:
            ate_by_method[method].append(float(val))

    print_counter(ptype_counter, "By problem_type", total)
    print_counter(method_counter, "By identification_method", total)
    print_counter(num_nodes_counter, "By number of nodes", total)
    print_counter(ptype_nodes_counter, "By (problem_type, num_nodes)", total)

    print(f"\n  ATE / LATE distribution per identification method")
    print(f"  {'─' * 50}")
    for method in sorted(ate_by_method.keys()):
        print(f"\n  method={method}")
        print_ate_distribution(ate_by_method[method], method)


def main() -> None:
    print("Loading dataset irfanjamil/causal-reasoning-ate ...")
    train_ds = load_dataset("irfanjamil/causal-reasoning-ate", split="train")
    eval_ds  = load_dataset("irfanjamil/causal-reasoning-ate", split="eval")

    profile_split(train_ds, "train")
    profile_split(eval_ds,  "eval")
    print()


if __name__ == "__main__":
    main()
