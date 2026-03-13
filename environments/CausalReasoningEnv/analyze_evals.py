"""analyze_evals.py — Conditional metrics analysis for CausalATEEnv evaluation results.

Usage:
    python environments/CausalReasoningEnv/analyze_evals.py [--evals-dir PATH]

For each results.jsonl found under the evals directory, prints:
  - format_compliance: raw mean + root-cause breakdown for failures
  - method_validity:   conditioned on no format-termination substring in any user message
                       AND no <answer> tag in the first assistant message
  - set_validity:      conditioned on method_validity == 1
  - minimality:        conditioned on set_validity == 1
  - process_correctness: conditioned on set_validity == 1
  - ate_accuracy:      conditioned on format_compliance == 1 AND set_validity == 1
  - breakdown by problem_type

Also saves a timestamped CSV with one row per (model_run, problem_type).
"""

import argparse
import csv
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

PROBLEM_TYPES = ["backdoor_standard", "backdoor_empty", "frontdoor", "iv"]

DEFAULT_EVALS_DIR = Path(__file__).parent / "outputs" / "evals"

# Mirror of env.py constant — substrings that appear in the Turn-1 termination user message.
_FORMAT_TERMINATION_SUBSTRINGS = [
    "declare() was not called",
    "could not parse declare() arguments",
    "unknown method",
    "no probability tool calls were made",
    "exceeded maximum of",
]

_FORMAT_TERM_LABELS = {
    "declare() was not called":             "declare_not_called",
    "could not parse declare() arguments":  "parse_error",
    "unknown method":                       "unknown_method",
    "no probability tool calls were made":  "no_prob_calls",
    "exceeded maximum of":                  "exceeded_max_calls",
}

# Root-cause keys in the order we want to display them.
_FC_CAUSE_KEYS = [
    "declare_not_called",
    "parse_error",
    "unknown_method",
    "no_prob_calls",
    "exceeded_max_calls",
    "answer_in_turn1",
    "missing_answer",
    "other",
]

CSV_COLS = [
    "model_run", "problem_type", "n", "reward",
    "format_compliance",
    "fc_fail_n",
    *[f"fc_cause_{k}" for k in _FC_CAUSE_KEYS],
    "method_validity", "method_validity_cond", "method_validity_cond_n",
    "set_validity", "set_valid_cond", "set_valid_cond_n",
    "minimality", "minimality_cond", "minimality_cond_n",
    "process_correctness", "proc_correct_cond", "proc_correct_cond_n",
    "ate_accuracy_binary", "ate_acc_cond", "ate_acc_cond_n", "ate_acc_cond2", "ate_acc_cond2_n",
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
# Helpers
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


def _get_completion(row: dict) -> list:
    c = row.get("completion", [])
    return c if isinstance(c, list) else []


def _fc_root_cause(row: dict) -> str:
    """Classify why format_compliance failed for this row.

    Only call when format_compliance < 1.
    Scans user messages in the completion for termination substrings, then
    checks for answer-in-turn-1, then falls back to missing_answer / other.
    """
    completion = _get_completion(row)

    for msg in completion:
        if msg.get("role") == "user":
            content = msg.get("content", "") or ""
            for sub, label in _FORMAT_TERM_LABELS.items():
                if sub in content:
                    return label

    first_assistant = next((m for m in completion if m.get("role") == "assistant"), None)
    if first_assistant and re.search(r"<answer>", first_assistant.get("content", "") or ""):
        return "answer_in_turn1"

    # set_validity == 1 means the model reached Turn 2; missing answer is the only
    # remaining reason format_compliance would be 0.
    if _f(row, "set_validity") >= 1.0:
        return "missing_answer"

    return "other"


def _no_format_violation(row: dict) -> bool:
    """True if the rollout did not trigger a format-termination AND the first
    assistant message contains no <answer> tag.

    This is the conditioning mask for method_validity: we only evaluate
    whether the model picked the right method when it at least completed
    a structurally valid Turn 1.
    """
    completion = _get_completion(row)

    for msg in completion:
        if msg.get("role") == "user":
            content = msg.get("content", "") or ""
            if any(sub in content for sub in _FORMAT_TERMINATION_SUBSTRINGS):
                return False

    first_assistant = next((m for m in completion if m.get("role") == "assistant"), None)
    if first_assistant and re.search(r"<answer>", first_assistant.get("content", "") or ""):
        return False

    return True


# ─────────────────────────────────────────────────────────────────────────────
# Stats computation
# ─────────────────────────────────────────────────────────────────────────────


def compute_stats(rows: list[dict]) -> dict:
    """Compute raw and conditional metrics for a list of rows."""
    if not rows:
        return {}

    fc = [_f(r, "format_compliance")   for r in rows]
    mv = [_f(r, "method_validity")     for r in rows]
    sv = [_f(r, "set_validity")        for r in rows]
    mi = [_f(r, "minimality")          for r in rows]
    pc = [_f(r, "process_correctness") for r in rows]
    aa = [_f(r, "ate_accuracy_binary") for r in rows]
    rw = [_f(r, "reward")              for r in rows]

    # ── format_compliance root-cause breakdown ────────────────────────────────
    fc_fail_rows = [r for r, f in zip(rows, fc) if f < 1.0]
    fc_causes: Counter = Counter(_fc_root_cause(r) for r in fc_fail_rows)

    # ── conditioning masks ────────────────────────────────────────────────────
    no_fmt_viol = [_no_format_violation(r) for r in rows]   # for method_validity
    mv_ok       = [m == 1.0 for m in mv]                    # for set_validity
    sv_ok       = [s == 1.0 for s in sv]                    # for minimality / process_correctness
    fc_and_sv      = [f == 1.0 and s == 1.0 for f, s in zip(fc, sv)]              # for ate_accuracy
    fc_sv_and_pc   = [f == 1.0 and s == 1.0 and p == 1.0 for f, s, p in zip(fc, sv, pc)]  # stricter

    mv_cond,  mv_n  = _cond_mean(mv, no_fmt_viol)
    sv_cond,  sv_n  = _cond_mean(sv, mv_ok)
    mi_cond,  mi_n  = _cond_mean(mi, sv_ok)
    pc_cond,  pc_n  = _cond_mean(pc, sv_ok)
    aa_cond,  aa_n  = _cond_mean(aa, fc_and_sv)
    aa_cond2, aa_n2 = _cond_mean(aa, fc_sv_and_pc)

    return {
        "n":                    len(rows),
        "reward":               _mean(rw),
        "format_compliance":    _mean(fc),
        "fc_fail_n":            len(fc_fail_rows),
        "fc_causes":            fc_causes,          # Counter — not stored in CSV directly
        **{f"fc_cause_{k}": fc_causes.get(k, 0) for k in _FC_CAUSE_KEYS},
        "method_validity":      _mean(mv),
        "method_validity_cond": mv_cond,
        "method_validity_cond_n": mv_n,
        "set_validity":         _mean(sv),
        "set_valid_cond":       sv_cond,
        "set_valid_cond_n":     sv_n,
        "minimality":           _mean(mi),
        "minimality_cond":      mi_cond,
        "minimality_cond_n":    mi_n,
        "process_correctness":  _mean(pc),
        "proc_correct_cond":    pc_cond,
        "proc_correct_cond_n":  pc_n,
        "ate_accuracy_binary":   _mean(aa),
        "ate_acc_cond":          aa_cond,
        "ate_acc_cond_n":        aa_n,
        "ate_acc_cond2":         aa_cond2,
        "ate_acc_cond2_n":       aa_n2,
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


def _fc_str(val, n: int) -> str:
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
    "iv":                "iv",
}


def _print_fc_breakdown(stats: dict) -> None:
    """Print root-cause breakdown for format_compliance failures."""
    fc_fail_n = stats.get("fc_fail_n", 0)
    if fc_fail_n == 0:
        print("    (no failures)")
        return
    print(f"    {fc_fail_n} rollout(s) with format_compliance < 1:")
    for k in _FC_CAUSE_KEYS:
        cnt = stats.get(f"fc_cause_{k}", 0)
        if cnt:
            pct = 100 * cnt / fc_fail_n
            print(f"      {k:<24s}  {cnt:4d}  ({pct:5.1f}%)")


def _print_overall(stats: dict) -> None:
    rows = [
        ("metric",              "raw",                              "conditional (N)"),
        ("—" * 22,              "—" * 8,                            "—" * 26),
        ("format_compliance",   _fv(stats["format_compliance"]),    "—"),
        ("method_validity",     _fv(stats["method_validity"]),      _fc_str(stats["method_validity_cond"], stats["method_validity_cond_n"]) + "  [no_fmt_viol]"),
        ("set_validity",        _fv(stats["set_validity"]),         _fc_str(stats["set_valid_cond"],       stats["set_valid_cond_n"])       + "  [mv=1]"),
        ("minimality",          _fv(stats["minimality"]),           _fc_str(stats["minimality_cond"],      stats["minimality_cond_n"])      + "  [sv=1]"),
        ("process_correctness", _fv(stats["process_correctness"]),  _fc_str(stats["proc_correct_cond"],    stats["proc_correct_cond_n"])    + "  [sv=1]"),
        ("ate_accuracy_binary",  _fv(stats["ate_accuracy_binary"]),  _fc_str(stats["ate_acc_cond"],  stats["ate_acc_cond_n"])  + "  [fc=1,sv=1]"),
        ("",                     "",                                  _fc_str(stats["ate_acc_cond2"], stats["ate_acc_cond2_n"]) + "  [fc=1,sv=1,pc=1]"),
        ("—" * 22,              "—" * 8,                            "—" * 26),
        ("reward",              _fv(stats["reward"]),               "—"),
    ]
    col_w = [max(len(r[i]) for r in rows) for i in range(3)]
    for r in rows:
        print(f"  {r[0]:<{col_w[0]}}  {r[1]:<{col_w[1]}}  {r[2]}")

    print(f"\n  format_compliance failures:")
    _print_fc_breakdown(stats)


def _print_by_type(by_type: dict[str, dict]) -> None:
    col_w = 18
    header = f"  {'metric':<30}" + "".join(f"  {PT_LABELS.get(pt, pt):<{col_w}}" for pt in PROBLEM_TYPES)
    print(header)
    print("  " + "—" * (len(header) - 2))

    def row(label: str, key: str, cond_key: str | None = None, n_key: str | None = None) -> None:
        line = f"  {label:<30}"
        for pt in PROBLEM_TYPES:
            s = by_type.get(pt, {})
            if not s:
                cell = "—"
            elif cond_key:
                cell = _fc_str(s.get(cond_key), s.get(n_key, 0))
            else:
                cell = _fv(s.get(key))
            line += f"  {cell:<{col_w}}"
        print(line)

    def row_fc_cause(cause_key: str) -> None:
        label = f"    fc:{cause_key}"
        line = f"  {label:<30}"
        for pt in PROBLEM_TYPES:
            s = by_type.get(pt, {})
            cnt = s.get(f"fc_cause_{cause_key}", 0)
            fail_n = s.get("fc_fail_n", 0)
            if fail_n == 0:
                cell = "— "
            else:
                cell = f"{cnt}/{fail_n}"
            line += f"  {cell:<{col_w}}"
        print(line)

    row("n (rollouts)",                  "n")
    row("reward (raw)",                  "reward")
    print()
    row("format_compliance (raw)",        "format_compliance")
    row("  fc_fail_n",                   "fc_fail_n")
    for k in _FC_CAUSE_KEYS:
        row_fc_cause(k)
    print()
    row("method_validity (raw)",          "method_validity")
    row("method_validity (cond|no_fmt)",  None, "method_validity_cond", "method_validity_cond_n")
    print()
    row("set_validity (raw)",            "set_validity")
    row("set_validity (cond|mv=1)",      None, "set_valid_cond",    "set_valid_cond_n")
    print()
    row("minimality (raw)",              "minimality")
    row("minimality (cond|sv=1)",        None, "minimality_cond",   "minimality_cond_n")
    print()
    row("proc_corr (raw)",               "process_correctness")
    row("proc_corr (cond|sv=1)",         None, "proc_correct_cond", "proc_correct_cond_n")
    print()
    row("ate_acc_binary (raw)",              "ate_accuracy_binary")
    row("ate_acc_binary (cond|fc,sv=1)",     None, "ate_acc_cond",  "ate_acc_cond_n")
    row("ate_acc_binary (cond|fc,sv,pc=1)",  None, "ate_acc_cond2", "ate_acc_cond2_n")


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
