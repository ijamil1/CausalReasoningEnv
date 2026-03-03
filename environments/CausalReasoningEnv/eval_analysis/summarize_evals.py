"""Summarize eval results across all models in outputs/evals/.

For each model, computes per-metric averages overall, grouped by
identifiability_status, and grouped by problem_type.

Usage:
    python scripts/summarize_evals.py
    python scripts/summarize_evals.py --output path/to/summary.txt
"""

import argparse
import json
import pathlib
from collections import defaultdict

EVALS_DIR = pathlib.Path(__file__).parent.parent / "outputs" / "evals"

METRICS = [
    "format_compliance",
    "status_check",
    "answer_quality",
    "answer_correctness",
]


def avg(vals):
    return sum(vals) / len(vals) if vals else float("nan")


def fmt(v):
    return f"{v:.4f}" if v == v else "N/A"  # nan check


def load_model_data(path: pathlib.Path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def discover_models(evals_dir: pathlib.Path) -> dict[str, pathlib.Path]:
    """Return {model_name: results.jsonl path} for all models found."""
    models = {}
    for jsonl in sorted(evals_dir.glob("**/results.jsonl")):
        # Directory name is CausalReasoningEnv--<provider>--<model>
        model_dir = jsonl.parts[-3]  # e.g. CausalReasoningEnv--openai--gpt-4.1-mini
        parts = model_dir.split("--", 2)
        model_name = parts[2] if len(parts) == 3 else model_dir
        models[model_name] = jsonl
    return models


def compute_stats(rows: list[dict], metrics: list[str]) -> dict:
    available = [m for m in metrics if m in rows[0]]

    overall = {m: avg([r[m] for r in rows]) for m in available}

    by_id: dict[str, list] = defaultdict(list)
    by_pt: dict[str, list] = defaultdict(list)
    for r in rows:
        by_id[r["info"]["identifiability_status"]].append(r)
        by_pt[r["info"]["problem_type"]].append(r)

    by_id_avgs = {
        k: {m: avg([r[m] for r in v]) for m in available}
        for k, v in by_id.items()
    }
    by_pt_avgs = {
        k: {m: avg([r[m] for r in v]) for m in available}
        for k, v in by_pt.items()
    }

    return {
        "n": len(rows),
        "available_metrics": available,
        "overall": overall,
        "by_identifiability_status": by_id_avgs,
        "by_identifiability_status_counts": {k: len(v) for k, v in by_id.items()},
        "by_problem_type": by_pt_avgs,
        "by_problem_type_counts": {k: len(v) for k, v in by_pt.items()},
    }


def build_report(all_stats: dict[str, dict]) -> str:
    lines = []
    lines.append("=" * 80)
    lines.append("CAUSAL REASONING ENV — EVAL METRICS SUMMARY")
    lines.append("=" * 80)

    for model, data in all_stats.items():
        metrics = data["available_metrics"]
        lines.append(f"\nMODEL: {model}  (n={data['n']})")
        lines.append("-" * 60)

        # Overall
        lines.append("\n  Overall:")
        lines.append(f"    {'Metric':<25} {'Value':>8}")
        lines.append(f"    {'-' * 33}")
        for m in metrics:
            lines.append(f"    {m:<25} {fmt(data['overall'][m]):>8}")

        # By identifiability_status
        lines.append("\n  By identifiability_status:")
        id_statuses = sorted(data["by_identifiability_status"].keys())
        header = f"    {'Metric':<25}" + "".join(f"{s:>22}" for s in id_statuses)
        lines.append(header)
        lines.append(f"    {'-' * (25 + 22 * len(id_statuses))}")
        for m in metrics:
            row = f"    {m:<25}" + "".join(
                f"{fmt(data['by_identifiability_status'][s][m]):>22}" for s in id_statuses
            )
            lines.append(row)
        count_row = f"    {'n':<25}" + "".join(
            f"{data['by_identifiability_status_counts'].get(s, 0):>22}" for s in id_statuses
        )
        lines.append(count_row)

        # By problem_type
        lines.append("\n  By problem_type:")
        pt_statuses = sorted(data["by_problem_type"].keys())
        col_w = 18
        header2 = f"    {'Problem Type':<35} " + "  ".join(f"{m:>{col_w}}" for m in metrics) + "    n"
        lines.append(header2)
        lines.append(f"    {'-' * (35 + (col_w + 2) * len(metrics) + 6)}")
        for pt in pt_statuses:
            vals = "  ".join(
                f"{fmt(data['by_problem_type'][pt][m]):>{col_w}}" for m in metrics
            )
            n = data["by_problem_type_counts"].get(pt, 0)
            lines.append(f"    {pt:<35} {vals}  {n:>4}")

    lines.append("\n" + "=" * 80)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=EVALS_DIR / "metrics_summary.txt",
        help="Path to write the summary (default: outputs/evals/metrics_summary.txt)",
    )
    args = parser.parse_args()

    models = discover_models(EVALS_DIR)
    if not models:
        print(f"No results.jsonl files found under {EVALS_DIR}")
        return

    all_stats = {}
    for model_name, path in models.items():
        rows = load_model_data(path)
        all_stats[model_name] = compute_stats(rows, METRICS)

    report = build_report(all_stats)
    print(report)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report + "\n")
    print(f"\nWritten to: {args.output}")


if __name__ == "__main__":
    main()
