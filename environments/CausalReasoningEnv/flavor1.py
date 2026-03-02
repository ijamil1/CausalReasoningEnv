"""Flavor 1 — Minimal Adjustment Set Identification.

Given a randomly generated DAG with a designated treatment node X and
outcome node Y, the model must identify the minimal adjustment set Z:
the smallest set of non-descendants of X whose conditioning blocks all
backdoor paths between X and Y (via d-separation in the backdoor graph).

Environment type: vf.SingleTurnEnv. The DAG is presented to the model
as a textual description only (nodes, edges, adjacency, observed/latent status).
"""

import pathlib
import re

import networkx as nx
import verifiers as vf
from networkx.algorithms.d_separation import is_d_separator

from prompts import build_system_prompt


# ─────────────────────────────────────────────────────────────────────────────
# Prompt construction
# ─────────────────────────────────────────────────────────────────────────────


_F1_INTRO = """\
You will be given a Directed Acyclic Graph (DAG) representing a structural \
causal model. Nodes are variables. Nodes may be observed or latent/unobserved. \
A directed edge A→B means A is a direct cause of B.\
"""

_F1_TASK = """\
Given the DAG, determine whether the average treatment effect ATE = E[Y | do(X=1)] − E[Y | do(X=0)] \
is identifiable, and if so, identify the method \
(backdoor adjustment set or front-door mediator set) and the required variable set.\
"""

_F1_RESPONSE_FORMAT = """\
RESPONSE FORMAT
────────────────
You must follow the following format exactly. Write all reasoning inside a <reasoning> block. After </reasoning>, write \
exactly one <answer> block using one of these three forms:

  If identifiable via a backdoor adjustment set (including empty set):
      <answer>{N1, N2, ...}</answer>   or   <answer>{}</answer>

  If identifiable via the front-door criterion (for mediator M, use set notation even if M is a single node):
      <answer>frontdoor: {M1, M2, ...}</answer>

  If not identifiable:
      <answer>not_identifiable</answer>

Rules: do not write <answer> inside <reasoning>; use integer node IDs.
"""

SYSTEM_PROMPT = build_system_prompt(_F1_INTRO, _F1_RESPONSE_FORMAT, task=_F1_TASK)


def format_problem(
    edges: list,
    nodes: list,
    observed_nodes: list,
    latent_nodes: list,
    X: int,
    Y: int,
) -> str:
    """Render a DAG problem as a readable string for the model."""
    parents: dict[int, list[int]] = {n: [] for n in nodes}
    children: dict[int, list[int]] = {n: [] for n in nodes}
    for u, v in edges:
        children[u].append(v)
        parents[v].append(u)

    edge_str = ", ".join(f"{u}→{v}" for u, v in sorted(edges))
    node_str = ", ".join(str(n) for n in sorted(nodes))
    obs_str = ", ".join(str(n) for n in sorted(observed_nodes))
    lat_str = ", ".join(str(n) for n in sorted(latent_nodes)) if latent_nodes else "none"

    adj_lines = []
    for n in sorted(nodes):
        pa = sorted(parents[n])
        ch = sorted(children[n])
        kind = "latent" if n in latent_nodes else "observed"
        adj_lines.append(
            f"  Node {n} ({kind}): parents=[{', '.join(map(str, pa))}], "
            f"children=[{', '.join(map(str, ch))}]"
        )
    adj_str = "\n".join(adj_lines)

    return (
        f"DAG INFORMATION\n"
        f"───────────────\n"
        f"Nodes:    {node_str}\n"
        f"Observed: {obs_str}\n"
        f"Latent:   {lat_str}\n"
        f"Edges:    {edge_str}\n\n"
        f"Adjacency:\n{adj_str}\n\n"
        f"Treatment (X): {X}\n"
        f"Outcome   (Y): {Y}\n\n"
        f"QUESTION\n"
        f"────────\n"
        f"Is ATE = E[Y | do(X=1)] − E[Y | do(X=0)] identifiable from the causal model implied by this DAG? "
        f"If yes, state the identification method and the required variable set. "
        f"If not, respond with not_identifiable. "
        f"Respond according to the response format specified in the system prompt."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Answer parsing
# ─────────────────────────────────────────────────────────────────────────────


def parse_answer(content: str) -> dict | None:
    """Parse the model's <answer> tag into a structured result dict.

    Valid forms and returned dicts:
      <answer>{N1, N2, ...}</answer>         → {"type": "backdoor", "set": frozenset({N1, N2})}
      <answer>{}</answer>                    → {"type": "backdoor", "set": frozenset()}
      <answer>frontdoor: {M1, M2}</answer>   → {"type": "frontdoor", "mediators": frozenset({M1, M2})}
      <answer>not_identifiable</answer>      → {"type": "not_identifiable"}

    Returns None if no valid form is found or the content is malformed.
    """
    stripped = re.sub(r"<reasoning>.*?</reasoning>", "", content, flags=re.DOTALL)
    matches = re.findall(r"<answer>\s*(.*?)\s*</answer>", stripped, flags=re.DOTALL)
    if len(matches) != 1:
        return None
    inner = matches[0].strip()

    # not_identifiable — exact string match only
    if inner == "not_identifiable":
        return {"type": "not_identifiable"}

    # frontdoor: {M1, M2, ...}
    m = re.match(r"^frontdoor:\s*\{([^}]*)\}$", inner, re.IGNORECASE)
    if m:
        body = m.group(1).strip()
        if body == "":
            return None  # empty mediator set is invalid
        try:
            return {"type": "frontdoor", "mediators": frozenset(int(x.strip()) for x in body.split(","))}
        except ValueError:
            return None

    # {N1, N2, ...} or {} — backdoor adjustment set
    m = re.match(r"^\{([^}]*)\}$", inner)
    if m:
        body = m.group(1).strip()
        if body == "":
            return {"type": "backdoor", "set": frozenset()}
        try:
            return {"type": "backdoor", "set": frozenset(int(x.strip()) for x in body.split(","))}
        except ValueError:
            return None

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Reward functions
# ─────────────────────────────────────────────────────────────────────────────


async def format_compliance(completion) -> float:
    """Reward: response contains exactly one parseable <answer> block (0.05 weight)."""
    content = completion[-1]["content"]
    return 1.0 if parse_answer(content) is not None else 0.0


async def status_check(completion, info) -> float:
    """Reward: correct identification method declared

    For backdoor problems:   predicted type must be 'backdoor' 
    For frontdoor problems:  predicted type must be 'frontdoor'.
    For not_identifiable:    predicted type must be 'not_identifiable'.

    Returns 1.0 on full credit, 0.0 otherwise.
    """
    content = completion[-1]["content"]
    predicted = parse_answer(content)
    if predicted is None:
        return 0.0

    true_status = info["identifiability_status"]
    # Map ground-truth status → expected predicted type
    expected_type = {
        "identifiable":          "backdoor",
        "empty":                 "backdoor",
        "identifiable_frontdoor": "frontdoor",
        "not_identifiable":      "not_identifiable",
    }.get(true_status)

    if predicted["type"] != expected_type:
        return 0.0

    return 1.0


async def answer_correctness(completion, info) -> float:
    """Reward: exact correctness of the answer (0.80 weight).

    Scoring per identifiability status:
      identifiable (backdoor, non-empty):
        1.0  — predicted set matches any element of minimal_adjustment_sets
        0.25 — predicted set is valid but larger than minimum size
        0.0  — invalid set, wrong type, or unparseable
      empty (backdoor, empty set):
        1.0  — predicted set is {}
        0.0  — anything else
      identifiable_frontdoor:
        1.0  — predicted mediator matches mediator_node
        0.0  — wrong mediator or wrong type
      not_identifiable:
        1.0  — predicted type is 'not_identifiable'
        0.0  — anything else
    """
    content = completion[-1]["content"]
    predicted = parse_answer(content)
    if predicted is None:
        return 0.0

    true_status = info["identifiability_status"]

    if true_status == "not_identifiable":
        return 1.0 if predicted["type"] == "not_identifiable" else 0.0

    if true_status == "identifiable_frontdoor":
        if predicted["type"] != "frontdoor":
            return 0.0
        gold_mediators = frozenset({info["mediator_node"]})
        return 1.0 if predicted["mediators"] == gold_mediators else 0.0

    if true_status == "empty":
        if predicted["type"] != "backdoor":
            return 0.0
        return 1.0 if predicted["set"] == frozenset() else 0.0

    # true_status == "identifiable" (non-empty backdoor)
    if predicted["type"] != "backdoor":
        return 0.0

    predicted_set = predicted["set"]
    gold_sets = [frozenset(s) for s in info["minimal_adjustment_sets"]]

    # Exact match against any minimum-size set
    if predicted_set in gold_sets:
        return 1.0

    # Valid but non-minimal: check d-separation and no descendants
    X, Y = info["X"], info["Y"]
    G = nx.DiGraph()
    G.add_nodes_from(info["nodes"])
    G.add_edges_from(info["edges"])
    if predicted_set & nx.descendants(G, X):
        return 0.0
    G_bd = G.copy()
    G_bd.remove_edges_from(list(G.out_edges(X)))
    if is_d_separator(G_bd, {X}, {Y}, predicted_set):
        return 0.5  # valid but non-minimal

    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


_NUM_TRAIN = 250
_NUM_EVAL = 100
_MIN_NODES = 8
_MAX_NODES = 12
_SEED = 42

_DATASET_DIR = pathlib.Path(__file__).parent / "datasets" / "flavor1"
_TRAIN_DATASET_PATH = _DATASET_DIR / "train"
_EVAL_DATASET_PATH = _DATASET_DIR / "eval"


_HF_DATASET_ID = "irfanjamil/causal-reasoning-flavor1"


def load_flavor1() -> Flavor1Env:
    """Load the Flavor 1 environment (adjustment set identification).

    Datasets are loaded from HuggingFace Hub: irfanjamil/causal-reasoning-flavor1.
    """
    from datasets import load_dataset

    dataset = load_dataset(_HF_DATASET_ID)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    rubric = vf.Rubric(
        funcs=[format_compliance, status_check, answer_correctness],
        weights=[0.1, 0.1, 0.80],
    )

    return vf.SingleTurnEnv(
        dataset=train_dataset,
        eval_dataset=eval_dataset,
        system_prompt=SYSTEM_PROMPT,
        rubric=rubric,
    )


if __name__ == "__main__":
    from collections import Counter
    from data_generation.flavor1_gen import (
        build_dataset,
        generate_stratified_dag_problems,
    )

    print(f"Generating stratified problems (seed={_SEED}, nodes={_MIN_NODES}–{_MAX_NODES})…")
    train_problems, eval_problems = generate_stratified_dag_problems(
        n_train=_NUM_TRAIN,
        n_eval=_NUM_EVAL,
        min_nodes=_MIN_NODES,
        max_nodes=_MAX_NODES,
        seed=_SEED,
    )
    print(f"  Train: {len(train_problems)} problems")
    print(f"  Eval:  {len(eval_problems)} problems")
    for split_name, probs in [("Train", train_problems), ("Eval", eval_problems)]:
        counts = Counter(p["problem_type"] for p in probs)
        print(f"  {split_name} type distribution: {dict(counts)}")

    train_dataset = build_dataset(train_problems, format_problem)
    eval_dataset = build_dataset(eval_problems, format_problem)

    _DATASET_DIR.mkdir(parents=True, exist_ok=True)
    train_dataset.save_to_disk(str(_TRAIN_DATASET_PATH))
    eval_dataset.save_to_disk(str(_EVAL_DATASET_PATH))
    print(f"\nTrain dataset ({len(train_dataset)} rows) saved → {_TRAIN_DATASET_PATH}")
    print(f"Eval  dataset ({len(eval_dataset)} rows) saved → {_EVAL_DATASET_PATH}\n")
