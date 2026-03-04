"""Flavor 2 — ATE Estimation (Analytical + Nonparametric).

Problems provide a DAG and, depending on type, may include a linear SCM
(structural equations with numeric coefficients), observational data, or both.

  SCM-only problems (~20%): linear SCM, no data. X is a binary root node.
    Model computes exact ATE via directed path-tracing.

  Data-only problems (~80%): DAG + discrete observational data, no SCM.
    Variables are binary or ternary; latent nodes excluded from data.
    Model determines identifiability and estimates ATE nonparametrically.

Environment type: vf.StatefulToolEnv (multi-turn, max 10 turns).

Tools:
  check_d_separation       — verifies d-separation in backdoor graph
  find_adjustment_sets     — finds minimal adjustment sets (training scaffold)
  run_python               — executes Python; pandas/numpy pre-loaded; df available
  load_data                — returns data CSV in head/describe/full format
"""

import contextlib
import io
import json
import re
import traceback

import networkx as nx
import verifiers as vf
from networkx.algorithms.d_separation import find_minimal_d_separator, is_d_separator

from prompts import build_system_prompt


# ─────────────────────────────────────────────────────────────────────────────
# Prompt construction
# ─────────────────────────────────────────────────────────────────────────────

_F2_INTRO = """\
You will be given a causal problem. You will always receive a DAG. Depending
on the problem, you may also receive a fully specified linear SCM (structural
equations with numeric coefficients), observational data, or both.\
"""

_F2_TASK = """\
Compute ATE = E[Y|do(X=1)] − E[Y|do(X=0)].

If possible given the information provided, compute the exact ATE.
Otherwise, compute the empirical ATE estimate from
the data. Your strategy when computing ATE should be to prioritize methods that lead to the simplest ATE computation. \
If ATE is not identifiable from the given information, state so.\
"""

_F2_RESPONSE_FORMAT = """\
RESPONSE FORMAT
────────────────
Write all reasoning inside a <reasoning> block. After </reasoning>, write an
<ate_type> block, an <extra_nodes> block, then exactly one <answer> block.
Omit <ate_type> and <extra_nodes> only when declaring not_identifiable.

<ate_type> block — state whether the ATE is:
    exact
    empirical

  Example: <ate_type>exact</ate_type>

<extra_nodes> block — the set of node IDs, beyond X and Y, that appear
in your ATE calculation.

  Examples:
      <extra_nodes>{3, 5}</extra_nodes>
      <extra_nodes>{}</extra_nodes>

<answer> block — use exactly one of these forms:

  Numeric estimate:
      <answer>ATE=0.27</answer>

  Not identifiable:
      <answer>not_identifiable</answer>

Rules: do not write <answer>, <ate_type>, or <extra_nodes> inside <reasoning>;
report numeric values as decimals; the <answer> tag may only contain ATE=...
or not_identifiable.\
"""

SYSTEM_PROMPT = build_system_prompt(_F2_INTRO, _F2_RESPONSE_FORMAT, task=_F2_TASK)


# ─────────────────────────────────────────────────────────────────────────────
# Problem formatter (unified)
# ─────────────────────────────────────────────────────────────────────────────


def format_problem(p: dict) -> str:
    """Render a Flavor 2 problem. Always shows DAG; conditionally shows SCM and data."""
    nodes = sorted(p["nodes"])
    observed = sorted(p["observed_nodes"])
    latent = sorted(p["latent_nodes"])

    obs_str = ", ".join(str(nd) for nd in observed)
    lat_str = ", ".join(str(nd) for nd in latent) if latent else "none"
    edge_str = ", ".join(f"{u}→{v}" for u, v in p["edges"])

    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(p["edges"])
    adj_lines = []
    for nd in nodes:
        pa = sorted(G.predecessors(nd))
        ch = sorted(G.successors(nd))
        kind = "latent" if nd in latent else "observed"
        label = f" ({kind})" if latent else ""
        adj_lines.append(
            f"  Node {nd}{label}: parents=[{', '.join(map(str, pa))}], "
            f"children=[{', '.join(map(str, ch))}]"
        )

    dag_section = (
        f"DAG INFORMATION\n"
        f"───────────────\n"
        f"Nodes:    {', '.join(str(nd) for nd in nodes)}\n"
        f"Observed: {obs_str}\n"
        f"Latent:   {lat_str}\n"
        f"Edges:    {edge_str}\n\n"
        f"Adjacency:\n" + "\n".join(adj_lines)
    )

    # SCM section (only when structural equations are available)
    scm_section = ""
    if p.get("structural_equations_text"):
        scm_section = (
            f"\n\nSTRUCTURAL EQUATIONS\n"
            f"────────────────────\n"
            f"{p['structural_equations_text']}"
        )

    # Data section (only when observational data is available)
    data_section = ""
    if p.get("data_csv"):
        n_rows = p["data_csv"].count("\n") - 1
        data_section = (
            f"\n\nOBSERVATIONAL DATA\n"
            f"──────────────────\n"
            f"The dataset has {n_rows} rows. "
            f"Observed columns: {obs_str}. "
            f"Latent nodes are NOT in the data.\n"
            f"Use the load_data tool to access the data."
        )

    question_section = (
        f"\n\nTreatment (X): {p['X']}\n"
        f"Outcome   (Y): {p['Y']}\n\n"
        f"QUESTION\n"
        f"────────\n"
        f"Compute ATE = E[Y|do(X=1)] − E[Y|do(X=0)].\n"
        f"If the exact ATE can be computed from the given information, report it;\n"
        f"otherwise report the empirical ATE estimate from the data.\n"
        f"If ATE is not identifiable, respond <answer>not_identifiable</answer>."
    )

    return dag_section + scm_section + data_section + question_section


# ─────────────────────────────────────────────────────────────────────────────
# Answer and formula parsing
# ─────────────────────────────────────────────────────────────────────────────


def parse_answer_2(content: str) -> dict | None:
    """Parse the model's <answer> tag.

    Valid forms:
      <answer>ATE=0.27</answer>
        → {"status": "identifiable", "ATE": 0.27}
      <answer>not_identifiable</answer>
        → {"status": "not_identifiable", "ATE": None}

    Returns None if no valid form found.
    """
    stripped = re.sub(r"<reasoning>.*?</reasoning>", "", content, flags=re.DOTALL)
    stripped = re.sub(r"<ate_type>.*?</ate_type>", "", stripped, flags=re.DOTALL)
    stripped = re.sub(r"<extra_nodes>.*?</extra_nodes>", "", stripped, flags=re.DOTALL)
    matches = re.findall(r"<answer>\s*(.*?)\s*</answer>", stripped, flags=re.DOTALL)
    if len(matches) != 1:
        return None
    inner = matches[0].strip()

    if inner == "not_identifiable":
        return {"status": "not_identifiable", "ATE": None}

    ate_match = re.search(r"ATE\s*=\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)", inner)
    if not ate_match:
        return None

    try:
        ate_val = float(ate_match.group(1))
    except ValueError:
        return None

    return {"status": "identifiable", "ATE": ate_val}


def parse_ate_type(content: str) -> str | None:
    """Extract <ate_type> tag content ('exact' or 'empirical'). Returns None if absent."""
    stripped = re.sub(r"<reasoning>.*?</reasoning>", "", content, flags=re.DOTALL)
    matches = re.findall(r"<ate_type>\s*(.*?)\s*</ate_type>", stripped, flags=re.DOTALL)
    if len(matches) != 1:
        return None
    return matches[0].strip().lower()


def parse_extra_nodes(content: str) -> list[int] | None:
    """Extract node IDs from the <extra_nodes> tag.

    Returns:
        list of int node IDs (may be empty for {})
        None if tag absent or unparseable
    """
    stripped = re.sub(r"<reasoning>.*?</reasoning>", "", content, flags=re.DOTALL)
    matches = re.findall(r"<extra_nodes>\s*(.*?)\s*</extra_nodes>", stripped, flags=re.DOTALL)
    if len(matches) != 1:
        return None
    inner = matches[0].strip()
    # Strip surrounding braces if present
    inner = re.sub(r"^\{|\}$", "", inner).strip()
    if not inner:
        return []
    try:
        return [int(x.strip()) for x in inner.split(",") if x.strip()]
    except ValueError:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Tool implementations
# ─────────────────────────────────────────────────────────────────────────────


async def check_d_separation(
    edges: list[list[int]], X: int, Y: int, Z: list[int]
) -> str:
    """Check if Z d-separates X from Y in the backdoor graph (X's outgoing edges removed).

    Args:
        edges: List of [u, v] directed edges.
        X: Treatment node index.
        Y: Outcome node index.
        Z: Proposed conditioning set (list of node indices).
    Returns:
        "d-separated" or "not d-separated" with a brief explanation.
    """
    G = nx.DiGraph()
    G.add_edges_from([tuple(e) for e in edges])

    G_bd = G.copy()
    G_bd.remove_edges_from(list(G.out_edges(X)))

    Z_set = frozenset(Z)
    try:
        sep = is_d_separator(G_bd, {X}, {Y}, Z_set)
        if sep:
            return (
                f"d-separated: conditioning on Z={sorted(Z)} blocks all backdoor "
                f"paths from {X} to {Y} in the backdoor graph."
            )
        else:
            return (
                f"not d-separated: conditioning on Z={sorted(Z)} leaves at least one "
                f"unblocked path between {X} and {Y} in the backdoor graph."
            )
    except Exception as exc:
        return f"Error checking d-separation: {exc}"


async def find_adjustment_sets(edges: list[list[int]], X: int, Y: int) -> str:
    """Find all minimal valid adjustment sets for (X, Y) in the given DAG.

    NOTE: Training scaffold only — not available at eval time.

    Args:
        edges: List of [u, v] directed edges.
        X: Treatment node index.
        Y: Outcome node index.
    Returns:
        JSON object with adjustment_sets list (list of lists) or error.
    """
    from itertools import combinations

    G = nx.DiGraph()
    G.add_edges_from([tuple(e) for e in edges])

    all_nodes = set(G.nodes())
    candidates = sorted(all_nodes - {X, Y})

    G_bd = G.copy()
    G_bd.remove_edges_from(list(G.out_edges(X)))

    try:
        first = find_minimal_d_separator(G_bd, X, Y, restricted=set(candidates))
        if first is None:
            return json.dumps({
                "adjustment_sets": [],
                "status": "no valid adjustment set exists",
            })

        k = len(first)
        all_min = [
            sorted(combo)
            for combo in combinations(candidates, k)
            if is_d_separator(G_bd, {X}, {Y}, set(combo))
        ]
        return json.dumps({
            "adjustment_sets": all_min if all_min else [sorted(first)],
            "min_size": k,
            "status": "ok",
        })
    except Exception as exc:
        return f"Error finding adjustment sets: {exc}"


async def run_python(code: str, _ns: dict = None, _data_csv: str = None) -> str:
    """Execute Python code in a persistent session and return stdout + stderr.

    The session persists across calls within the same rollout.
    On first call: pandas (pd), numpy (np), scipy are pre-imported.
    If data is available: 'df' is pre-loaded as a pandas DataFrame with integer column names.

    Args:
        code: Python code to execute.
    Returns:
        stdout output; any errors are reported as tracebacks.
    """
    if _ns is None:
        _ns = {}

    if "_initialized" not in _ns:
        import numpy as np
        import pandas as pd
        import scipy

        _ns["np"] = np
        _ns["pd"] = pd
        _ns["scipy"] = scipy
        _ns["io"] = io
        _ns["_initialized"] = True

        if _data_csv:
            import pandas as _pd
            df_raw = _pd.read_csv(io.StringIO(_data_csv))
            # Convert column names to int where possible
            df_raw.columns = [
                int(c) if c.lstrip("-").isdigit() else c for c in df_raw.columns
            ]
            _ns["df"] = df_raw

    stdout_buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout_buf):
            exec(code, _ns)  # noqa: S102
        output = stdout_buf.getvalue()
        return output.rstrip() if output.strip() else "(no output)"
    except Exception:
        err = traceback.format_exc()
        captured = stdout_buf.getvalue()
        parts = []
        if captured.strip():
            parts.append(captured.rstrip())
        parts.append(f"Error:\n{err}")
        return "\n".join(parts)


async def load_data(format: str = "head", _data_csv: str = None) -> str:
    """Load the observational dataset for this problem.

    Args:
        format: 'head' (first 10 rows), 'describe' (summary statistics),
                or 'full' (all rows as CSV string).
    Returns:
        Requested view of the data, or a message if no data is available.
    """
    if _data_csv is None:
        return (
            "No observational data is available for this problem. "
            "Compute ATE analytically from the structural equations."
        )

    import pandas as _pd

    df = _pd.read_csv(io.StringIO(_data_csv))
    df.columns = [
        int(c) if c.lstrip("-").isdigit() else c for c in df.columns
    ]

    if format == "head":
        return df.head(10).to_string(index=False)
    elif format == "describe":
        return df.describe().to_string()
    elif format == "full":
        return _data_csv
    else:
        return f"Invalid format '{format}'. Choose from: 'head', 'describe', 'full'."


# ─────────────────────────────────────────────────────────────────────────────
# Reward functions
# ─────────────────────────────────────────────────────────────────────────────


async def format_compliance(completion) -> float:
    """Reward: response contains exactly one parseable <answer> block."""
    content = completion[-1]["content"] if completion else ""
    return 1.0 if parse_answer_2(content) is not None else 0.0


async def status_check(completion, info) -> float:
    """Reward: correct identifiability status declared.

    SCM-only problems are always identifiable; reward 1.0 if model does NOT
    declare not_identifiable.
    Data-only problems: reward 1.0 if declared status matches true status.
    """
    content = completion[-1]["content"] if completion else ""
    parsed = parse_answer_2(content)
    if parsed is None:
        return 0.0

    subcase = info.get("subcase", "B")
    true_status = info.get("identifiability_status", "identifiable")

    if subcase == "A":
        # SCM-only is always identifiable
        return 1.0 if parsed["status"] == "identifiable" else 0.0

    # Data-only: match declared status to true status
    if true_status == "not_identifiable":
        return 1.0 if parsed["status"] == "not_identifiable" else 0.0
    else:
        return 1.0 if parsed["status"] == "identifiable" else 0.0


async def formula_quality(completion, info) -> float:
    """Reward: correct <ate_type> and <extra_nodes> tags.

    Scoring (0.5 + 0.5):
      - not_identifiable declared correctly: 1.0 (tags omitted, no penalty)
      - not_identifiable declared incorrectly: 0.0
      - ate_type (0.5): "exact" for subcase A, "empirical" for subcase B
      - extra_nodes (0.5): {} for subcase A; adjustment set for backdoor; mediator for frontdoor
    """
    content = completion[-1]["content"] if completion else ""
    parsed = parse_answer_2(content)
    if parsed is None:
        return 0.0

    true_status = info.get("identifiability_status", "identifiable")

    # not_identifiable: no method tags needed
    if true_status == "not_identifiable":
        return 1.0 if parsed["status"] == "not_identifiable" else 0.0

    if parsed["status"] != "identifiable":
        return 0.0

    subcase = info.get("subcase", "B")

    # ── ate_type score (0.5) ──────────────────────────────────────────────────
    ate_type = parse_ate_type(content)
    if ate_type is None:
        type_score = 0.0
    elif subcase == "A":
        type_score = 0.5 if ate_type == "exact" else 0.25
    else:
        type_score = 0.5 if ate_type == "empirical" else 0.0

    # ── extra_nodes score (0.5) ───────────────────────────────────────────────
    cond_vars = parse_extra_nodes(content)
    if cond_vars is None:
        cond_score = 0.0
    elif subcase == "A":
        # X is root node: no conditioning needed → expect {}
        cond_score = 0.5 if cond_vars == [] else 0.0
    else:
        problem_type = info.get("problem_type", "")
        if problem_type == "frontdoor":
            mediator = info.get("mediator_node")
            if mediator is None:
                cond_score = 0.25
            else:
                cond_score = 0.5 if cond_vars == [mediator] else 0.0
        else:
            # Backdoor cases
            adj_set = info.get("adjustment_set") or []
            if set(cond_vars) == set(adj_set):
                cond_score = 0.5
            else:
                correct = len(set(cond_vars) & set(adj_set))
                total = len(set(cond_vars) | set(adj_set))
                cond_score = (correct / total * 0.5) if total > 0 else 0.0

    return type_score + cond_score


async def answer_quality(completion, info) -> float:
    """Reward shaping: graded answer quality (0.70 weight in rubric).

    SCM-only (Sub-case A):
      max(0, 1 − |ATE_hat − ATE_true| / (0.10 · |ATE_true|))
      Special case ATE_true=0: full credit iff |ATE_hat| ≤ 0.05

    Data-only (Sub-case B), identifiable:
      max(0, 1 − |ATE_hat − data_ATE| / (0.30 · |data_ATE|))
      Special case data_ATE=0: full credit iff |ATE_hat| ≤ 0.05

    Data-only, not_identifiable:
      1.0 if declared correctly; 0.0 if numeric estimate produced.
    """
    content = completion[-1]["content"] if completion else ""
    parsed = parse_answer_2(content)
    if parsed is None:
        return 0.0

    subcase = info.get("subcase", "B")
    true_status = info.get("identifiability_status", "identifiable")

    # ── SCM-only ──────────────────────────────────────────────────────────────
    if subcase == "A":
        if parsed["status"] != "identifiable":
            return 0.0
        ate_hat = parsed["ATE"]
        if ate_hat is None:
            return 0.0
        ate_true = info.get("true_ATE", 0.0) or 0.0

        if abs(ate_true) <= 1e-6:
            return 1.0 if abs(ate_hat) <= 0.05 else 0.0
        return max(0.0, 1.0 - abs(ate_hat - ate_true) / abs(ate_true) / 0.10)

    # ── Data-only ─────────────────────────────────────────────────────────────
    if true_status == "not_identifiable":
        if parsed["status"] == "identifiable":
            return 0.0
        return 1.0 if parsed["status"] == "not_identifiable" else 0.0

    if parsed["status"] != "identifiable":
        return 0.0

    ate_hat = parsed["ATE"]
    if ate_hat is None:
        return 0.0

    data_ate = info.get("data_ATE")
    if data_ate is None:
        return 0.0

    if abs(data_ate) <= 1e-6:
        return 1.0 if abs(ate_hat) <= 0.05 else 0.0
    return max(0.0, 1.0 - abs(ate_hat - data_ate) / abs(data_ate) / 0.30)


# ─────────────────────────────────────────────────────────────────────────────
# Stateful tool environment
# ─────────────────────────────────────────────────────────────────────────────


class Flavor2Env(vf.StatefulToolEnv):
    """Multi-turn tool environment for Flavor 2 (ATE estimation).

    Provides check_d_separation, find_adjustment_sets, run_python, and
    load_data tools. Per-rollout state stores a persistent Python namespace
    (across run_python calls) and data injection for load_data / run_python.
    """

    def __init__(self, **kwargs):
        # Initialise with no tools; add them with args_to_skip
        super().__init__(tools=[], max_turns=10, **kwargs)
        self.add_tool(check_d_separation)
        self.add_tool(find_adjustment_sets)
        self.add_tool(run_python, args_to_skip=["_ns", "_data_csv"])
        self.add_tool(load_data, args_to_skip=["_data_csv"])

    async def setup_state(self, state: vf.State) -> vf.State:
        """Initialise per-rollout persistent Python namespace."""
        state["_python_ns"] = {}
        return state

    def update_tool_args(
        self,
        tool_name: str,
        tool_args: dict,
        messages: vf.Messages,
        state: vf.State,
        **kwargs,
    ) -> dict:
        """Inject per-problem data into load_data and run_python calls."""
        args = dict(tool_args)
        info = state.get("info") or {}
        data_csv = info.get("data_csv")

        if tool_name == "load_data":
            args["_data_csv"] = data_csv
        elif tool_name == "run_python":
            args["_ns"] = state.get("_python_ns", {})
            args["_data_csv"] = data_csv

        return args


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


_HF_DATASET_ID = "irfanjamil/causal-reasoning-flavor2"


def load_flavor2() -> vf.Environment:
    """Load the Flavor 2 environment (ATE estimation).

    Attempts to load datasets from HuggingFace Hub.
    Falls back to local generation if the Hub dataset is not available.
    """
    from datasets import load_dataset

    try:
        dataset = load_dataset(_HF_DATASET_ID)
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]
    except Exception:
        # Fall back: generate a small dataset locally for testing
        import sys
        import os
        sys.path.insert(0, os.path.dirname(__file__))
        from data_generation.flavor2_gen import generate_flavor2_problems, build_dataset

        train_problems, eval_problems = generate_flavor2_problems(
            n_train=100, n_eval=40, seed=42
        )
        train_dataset = build_dataset(train_problems, format_problem)
        eval_dataset = build_dataset(eval_problems, format_problem)

    rubric = vf.Rubric(
        funcs=[format_compliance, status_check, formula_quality, answer_quality],
        weights=[0.05, 0.10, 0.15, 0.70],
    )

    return Flavor2Env(
        dataset=train_dataset,
        eval_dataset=eval_dataset,
        system_prompt=SYSTEM_PROMPT,
        rubric=rubric,
    )
