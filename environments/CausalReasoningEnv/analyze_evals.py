"""analyze_evals.py — Conditional metrics analysis for CausalATEEnv evaluation results.

Usage:
    python environments/CausalReasoningEnv/analyze_evals.py [--evals-dir PATH]

For each results.jsonl found under the evals directory, prints:
  - Raw (unconditional) mean for each rubric component
  - Conditional means that isolate true per-axis capability
  - Breakdown by problem_type

Conditional metric definitions:
  set_valid   | format_compliance > 0       : validity when a set tag was declared
  minimality  | set_valid == 1              : minimality when set is valid
  proc_correct| set_valid == 1              : tool-call correctness when set is valid
  ate_accuracy| process_correctness == 1    : ATE correctness when process is correct
               AND format_compliance == 1     (also implies answer tag was present)

Also saves a timestamped CSV with one row per (model_run, problem_type).
"""

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

PROBLEM_TYPES = ["backdoor_standard", "backdoor_empty", "frontdoor", "not_identifiable"]

DEFAULT_EVALS_DIR = Path(__file__).parent / "outputs" / "evals"

CSV_COLS = [
    "model_run", "problem_type", "n", "reward",
    "format_compliance", "set_valid", "minimality", "process_correctness", "ate_accuracy",
    "set_valid_cond", "set_valid_cond_n",
    "minimality_cond", "minimality_cond_n",
    "proc_correct_cond", "proc_correct_cond_n",
    "ate_acc_cond", "ate_acc_cond_n",
]


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────


def load_rows(jsonl_path: Path) -> list[dict]:
    rows = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            info = row.get("info", {})
            if isinstance(info, str):
                try:
                    info = json.loads(info)
                except json.JSONDecodeError:
                    info = {}
            row["_problem_type"] = info.get("problem_type", "unknown")
            rows.append(row)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Stats computation
# ─────────────────────────────────────────────────────────────────────────────


def _f(row: dict, key: str) -> float:
    val = row.get(key)
    try:
        return float(val) if val is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _mean(vals: list[float]) -> float | None:
    return sum(vals) / len(vals) if vals else None


def _cond_mean(vals: list[float], mask: list[bool]) -> tuple[float | None, int]:
    subset = [v for v, m in zip(vals, mask) if m]
    return (_mean(subset), len(subset))


def compute_stats(rows: list[dict]) -> dict:
    """Compute raw and conditional metrics for a list of rows."""
    if not rows:
        return {}

    fc = [_f(r, "format_compliance") for r in rows]
    sv = [_f(r, "set_valid") for r in rows]
    mi = [_f(r, "minimality") for r in rows]
    pc = [_f(r, "process_correctness") for r in rows]
    aa = [_f(r, "ate_accuracy") for r in rows]
    rw = [_f(r, "reward") for r in rows]

    fmt_ok      = [f > 0        for f in fc]
    sv_ok       = [s == 1.0     for s in sv]
    pc_and_fc   = [p == 1.0 and f == 1.0 for p, f in zip(pc, fc)]

    sv_cond,  sv_n  = _cond_mean(sv, fmt_ok)
    mi_cond,  mi_n  = _cond_mean(mi, sv_ok)
    pc_cond,  pc_n  = _cond_mean(pc, sv_ok)
    aa_cond,  aa_n  = _cond_mean(aa, pc_and_fc)

    return {
        "n":                    len(rows),
        "reward":               _mean(rw),
        "format_compliance":    _mean(fc),
        "set_valid":            _mean(sv),
        "minimality":           _mean(mi),
        "process_correctness":  _mean(pc),
        "ate_accuracy":         _mean(aa),
        "set_valid_cond":       sv_cond,
        "set_valid_cond_n":     sv_n,
        "minimality_cond":      mi_cond,
        "minimality_cond_n":    mi_n,
        "proc_correct_cond":    pc_cond,
        "proc_correct_cond_n":  pc_n,
        "ate_acc_cond":         aa_cond,
        "ate_acc_cond_n":       aa_n,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────────────────────


def _fv(val) -> str:
    if val is None:
        return "—"
    if isinstance(val, int):
        return str(val)
    return f"{val:.4f}"


def _fc(val, n: int) -> str:
    if val is None:
        return f"— (N={n})"
    return f"{val:.4f} (N={n})"


# ─────────────────────────────────────────────────────────────────────────────
# Terminal output
# ─────────────────────────────────────────────────────────────────────────────

PT_LABELS = {
    "backdoor_standard": "bkdr_std",
    "backdoor_empty":    "bkdr_emp",
    "frontdoor":         "frontdoor",
    "not_identifiable":  "not_ident",
}


def _print_overall(stats: dict) -> None:
    rows = [
        ("metric",               "raw",                                   "conditional (N)"),
        ("—" * 22,               "—" * 8,                                 "—" * 18),
        ("format_compliance",    _fv(stats["format_compliance"]),          "—"),
        ("set_valid",            _fv(stats["set_valid"]),                  _fc(stats["set_valid_cond"],    stats["set_valid_cond_n"])),
        ("minimality",           _fv(stats["minimality"]),                 _fc(stats["minimality_cond"],   stats["minimality_cond_n"])),
        ("process_correctness",  _fv(stats["process_correctness"]),        _fc(stats["proc_correct_cond"], stats["proc_correct_cond_n"])),
        ("ate_accuracy",         _fv(stats["ate_accuracy"]),               _fc(stats["ate_acc_cond"],      stats["ate_acc_cond_n"])),
        ("—" * 22,               "—" * 8,                                 "—" * 18),
        ("reward",               _fv(stats["reward"]),                     "—"),
    ]
    col_w = [max(len(r[i]) for r in rows) for i in range(3)]
    for r in rows:
        print(f"  {r[0]:<{col_w[0]}}  {r[1]:<{col_w[1]}}  {r[2]}")


def _print_by_type(by_type: dict[str, dict]) -> None:
    col_w = 15
    header = f"  {'metric':<26}" + "".join(f"  {PT_LABELS[pt]:<{col_w}}" for pt in PROBLEM_TYPES)
    print(header)
    print("  " + "—" * (len(header) - 2))

    def row(label: str, key: str, cond_key: str | None = None, n_key: str | None = None) -> None:
        line = f"  {label:<26}"
        for pt in PROBLEM_TYPES:
            s = by_type.get(pt, {})
            if not s:
                cell = "—"
            elif cond_key:
                cell = _fc(s.get(cond_key), s.get(n_key, 0))
            else:
                cell = _fv(s.get(key))
            line += f"  {cell:<{col_w}}"
        print(line)

    row("n (rollouts)",          "n")
    row("reward (raw)",          "reward")
    print()
    row("format_compliance",     "format_compliance")
    row("set_valid (raw)",       "set_valid")
    row("set_valid (cond|fc>0)", None, "set_valid_cond",    "set_valid_cond_n")
    print()
    row("minimality (raw)",      "minimality")
    row("minimality (cond|sv=1)","minimality",  "minimality_cond",   "minimality_cond_n")
    print()
    row("proc_corr (raw)",       "process_correctness")
    row("proc_corr (cond|sv=1)", None, "proc_correct_cond", "proc_correct_cond_n")
    print()
    row("ate_acc (raw)",         "ate_accuracy")
    row("ate_acc (cond|pc&fc=1)",None, "ate_acc_cond",      "ate_acc_cond_n")


def print_run(label: str, all_rows: list[dict]) -> tuple[dict, dict[str, dict]]:
    overall = compute_stats(all_rows)
    by_type = {pt: compute_stats([r for r in all_rows if r["_problem_type"] == pt])
               for pt in PROBLEM_TYPES}

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")

    print(f"\n[OVERALL — {overall['n']} rollouts]\n")
    _print_overall(overall)

    print(f"\n[BY PROBLEM TYPE]\n")
    _print_by_type(by_type)

    return overall, by_type


# ─────────────────────────────────────────────────────────────────────────────
# CSV output
# ─────────────────────────────────────────────────────────────────────────────


def _csv_val(val) -> str:
    if val is None:
        return ""
    if isinstance(val, int):
        return str(val)
    if isinstance(val, float):
        return f"{val:.6f}"
    return str(val)


def build_csv_rows(label: str, overall: dict, by_type: dict[str, dict]) -> list[dict]:
    def make_row(pt: str, stats: dict) -> dict:
        row = {"model_run": label, "problem_type": pt}
        for col in CSV_COLS[2:]:
            row[col] = _csv_val(stats.get(col))
        return row

    rows = [make_row("all", overall)]
    for pt in PROBLEM_TYPES:
        rows.append(make_row(pt, by_type.get(pt, {})))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze CausalATEEnv eval results with conditional metrics."
    )
    parser.add_argument(
        "--evals-dir", type=Path, default=DEFAULT_EVALS_DIR,
        help=f"Root directory to search for results.jsonl files (default: {DEFAULT_EVALS_DIR})",
    )
    args = parser.parse_args()

    evals_dir: Path = args.evals_dir
    if not evals_dir.exists():
        print(f"Error: evals directory not found: {evals_dir}", file=sys.stderr)
        sys.exit(1)

    jsonl_files = sorted(evals_dir.rglob("results.jsonl"))
    if not jsonl_files:
        print(f"Error: no results.jsonl files found under {evals_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(jsonl_files)} result file(s) under {evals_dir}.")

    all_csv_rows: list[dict] = []

    for jf in jsonl_files:
        # Label: path components between "evals/" and "results.jsonl"
        try:
            idx = list(jf.parts).index("evals")
            label = " / ".join(jf.parts[idx + 1 : -1])
        except ValueError:
            label = str(jf.parent)

        rows = load_rows(jf)
        if not rows:
            print(f"\nWarning: no rows loaded from {jf}", file=sys.stderr)
            continue

        overall, by_type = print_run(label, rows)
        all_csv_rows.extend(build_csv_rows(label, overall, by_type))

    # Save CSV
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = evals_dir / f"analysis_{ts}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLS)
        writer.writeheader()
        writer.writerows(all_csv_rows)

    print(f"\nCSV saved → {csv_path}\n")


if __name__ == "__main__":
    main()
