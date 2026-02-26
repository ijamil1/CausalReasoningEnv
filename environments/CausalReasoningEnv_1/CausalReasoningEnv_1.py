# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "networkx>=3.0",
#     "verifiers>=0.1.9.post3",
#     "datasets",
#     "matplotlib>=3.7",
# ]
# ///

"""CausalReasoningEnv_1 — Minimal Adjustment Set Identification.

Given a randomly generated DAG with a designated treatment node X and
outcome node Y, the model must identify the minimal adjustment set Z:
the smallest set of non-descendants of X whose conditioning blocks all
backdoor paths between X and Y (via d-separation in the backdoor graph).

Environment type: SingleTurnEnv subclass. The DAG is rendered as a PNG
at rollout start and injected into the prompt alongside the text description.
"""

import base64
import io
import json
import pathlib
import random
import re

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; must be set before pyplot import
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import verifiers as vf
from datasets import Dataset
from networkx.algorithms.d_separation import find_minimal_d_separator, is_d_separator


# ─────────────────────────────────────────────────────────────────────────────
# DAG rendering
# ─────────────────────────────────────────────────────────────────────────────


def _dag_layout(G: nx.DiGraph) -> dict:
    """Layer-by-layer topological layout (sources at top, sinks at bottom)."""
    pos = {}
    for depth, layer in enumerate(nx.topological_generations(G)):
        layer = sorted(layer)
        for i, node in enumerate(layer):
            pos[node] = ((i - (len(layer) - 1) / 2.0), -float(depth))
    return pos


def _render_dag_b64(G: nx.DiGraph, X: int, Y: int, figsize=(8, 6), dpi=100) -> str:
    """Render a DAG as a base64-encoded PNG string.

    X (treatment) is drawn in blue, Y (outcome) in orange, all other nodes
    in light gray. Labels use white text on colored nodes, black on gray.
    Layout is topological so causal flow reads top-to-bottom.
    """
    pos = _dag_layout(G)

    node_colors = ["#4C72B0" if n == X else "#DD8452" if n == Y else "#C8C8C8"
                   for n in G.nodes()]
    font_colors = {n: "white" if n in (X, Y) else "black" for n in G.nodes()}

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    node_size = 700
    nx.draw_networkx_nodes(G, pos=pos, ax=ax, node_color=node_colors, node_size=node_size)
    # connectionstyle curves edges so collinear nodes don't obscure long-range arrows
    nx.draw_networkx_edges(G, pos=pos, ax=ax, arrows=True, arrowsize=20,
                           edge_color="#555555", width=1.5,
                           node_size=node_size,
                           connectionstyle="arc3,rad=0.15")
    for node, (x, y) in pos.items():
        ax.text(x, y, str(node), ha="center", va="center",
                fontsize=10, color=font_colors[node], fontweight="bold")

    # Place legend outside the axes to avoid covering nodes
    ax.legend(handles=[
        mpatches.Patch(color="#4C72B0", label=f"X = {X}  (treatment)"),
        mpatches.Patch(color="#DD8452", label=f"Y = {Y}  (outcome)"),
        mpatches.Patch(color="#C8C8C8", label="other nodes"),
    ], loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, fontsize=9)
    ax.set_title("Causal DAG", fontsize=12)
    ax.axis("off")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode()


# ─────────────────────────────────────────────────────────────────────────────
# DAG generation
# ─────────────────────────────────────────────────────────────────────────────


def _make_dag(n: int, edge_prob: float, rng: random.Random) -> nx.DiGraph:
    """Generate a random DAG by keeping only forward edges from an Erdos-Renyi graph."""
    nodes = list(range(n))
    edges = [
        (u, v)
        for u in nodes
        for v in nodes
        if u < v and rng.random() < edge_prob
    ]
    return nx.DiGraph(edges)


def _try_sample_problem(
    rng: random.Random,
    min_nodes: int,
    max_nodes: int,
    edge_prob: float,
) -> dict | None:
    """Attempt to sample one valid causal adjustment-set problem.

    Each accepted problem satisfies:
      - Y is a descendant of X (a causal path exists).
      - Y is a leaf node (no outgoing edges).
      - At least 4 backdoor paths exist, with at least one of length ≥ 5 nodes.
      - A minimal d-separator (adjustment set) exists.

    Returns a problem dict (including a temporary "G" key for the nx.DiGraph
    used by _classify_problem) or None if any filter fails.
    """
    n = rng.randint(min_nodes, max_nodes)
    G = _make_dag(n, edge_prob, rng)
    nodes_list = list(G.nodes())
    if len(nodes_list) < 2:
        return None

    X, Y = rng.sample(nodes_list, 2)

    # Y must be reachable from X
    if not nx.has_path(G, X, Y):
        return None
    # Y must be a leaf
    if G.out_degree(Y) > 0:
        return None
    # No back-path (should be impossible in a DAG, but guard anyway)
    if nx.has_path(G, Y, X):
        return None

    # Backdoor graph: remove all edges out of X
    G_bd = G.copy()
    G_bd.remove_edges_from(list(G.out_edges(X)))

    try:
        bd_paths = list(nx.all_simple_paths(G_bd.to_undirected(), X, Y))
        if len(bd_paths) < 4 or not any(len(p) >= 5 for p in bd_paths):
            return None
        min_set = find_minimal_d_separator(G_bd, X, Y)
        if min_set is None:
            return None
    except Exception:
        return None

    return {
        "G": G,  # temporary; removed before dataset serialisation
        "edges": [(int(u), int(v)) for u, v in G.edges()],
        "nodes": [int(nd) for nd in G.nodes()],
        "X": int(X),
        "Y": int(Y),
        "minimal_adjustment_set": sorted(int(nd) for nd in min_set),
        "num_nodes": len(nodes_list),
        "num_backdoor_paths": len(bd_paths),
    }


def _classify_problem(problem: dict) -> str:
    """Classify a problem by the mechanism behind its ratio (|min_set|/|parents(X)|).

    Uses the temporary "G" key placed by _try_sample_problem.

    Classifications:
      "standard" — |min_set| >= |parents(X)|  (all parents needed; ratio ≥ 1)
      "ancestor" — ratio < 1 AND there exists a dropped parent p such that some
                   z ∈ min_set has a directed path z → … → p in G. This means
                   z is an ancestor of p, and conditioning on z blocks p's
                   backdoor contribution without conditioning on p itself.
      "collider" — ratio < 1 AND no dropped parent has any ancestor in min_set.
                   The redundancy arises from a collider structure on the backdoor
                   path(s) through that parent, not from ancestor absorption.
    """
    G: nx.DiGraph = problem["G"]
    X = problem["X"]
    parents_X = set(G.predecessors(X))
    min_set = set(problem["minimal_adjustment_set"])

    if len(min_set) >= len(parents_X):
        return "standard"

    # ratio < 1: at least one parent was dropped from the minimal set
    dropped_parents = parents_X - min_set
    for p in dropped_parents:
        ancestors_of_p = nx.ancestors(G, p)
        if ancestors_of_p & min_set:
            # Some node in min_set is an ancestor of dropped parent p —
            # conditioning on that ancestor blocks p's confounding path too.
            return "ancestor"

    return "collider"


def generate_stratified_dag_problems(
    n_train: int = 250,
    n_eval: int = 100,
    min_nodes: int = 8,
    max_nodes: int = 12,
    edge_prob: float = 0.41,
    seed: int = 42,
    target_ratio_lt1: float = 0.40,
    target_ancestor_fraction: float = 0.50,
    exclude: set[tuple] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Generate stratified train and eval problem pools with controlled difficulty.

    Distribution targets (applied to both train and eval via stratified split):
      - ~target_ratio_lt1 of all problems have |min_set| < |parents(X)|.
      - Within those, ~target_ancestor_fraction are "ancestor" type (the rest
        are "collider" type).

    Args:
        exclude: Optional set of (frozenset(edges), X, Y) signatures to reject,
                 used to guarantee disjointness from an existing problem pool.

    Returns:
        (train_problems, eval_problems): two lists of problem dicts with keys:
        edges, nodes, X, Y, minimal_adjustment_set, num_nodes,
        num_backdoor_paths, num_parents_X, problem_type.
    """
    rng = random.Random(seed)
    n_total = n_train + n_eval

    n_lt1 = round(target_ratio_lt1 * n_total)
    n_ancestor = round(target_ancestor_fraction * n_lt1)
    n_collider = n_lt1 - n_ancestor
    n_standard = n_total - n_lt1

    targets = {"standard": n_standard, "ancestor": n_ancestor, "collider": n_collider}
    buckets: dict[str, list[dict]] = {"standard": [], "ancestor": [], "collider": []}
    seen: set[tuple] = set(exclude) if exclude else set()

    while any(len(buckets[t]) < targets[t] for t in targets):
        prob = _try_sample_problem(rng, min_nodes, max_nodes, edge_prob)
        if prob is None:
            continue

        sig = (frozenset((int(u), int(v)) for u, v in prob["edges"]), prob["X"], prob["Y"])
        if sig in seen:
            continue

        ptype = _classify_problem(prob)
        if len(buckets[ptype]) >= targets[ptype]:
            continue

        G: nx.DiGraph = prob.pop("G")
        parents_X = set(G.predecessors(prob["X"]))
        prob["num_parents_X"] = len(parents_X)
        prob["problem_type"] = ptype
        buckets[ptype].append(prob)
        seen.add(sig)

    # Stratified split: each bucket contributes proportionally to train and eval
    train_frac = n_train / n_total if n_total > 0 else 1.0
    train_problems: list[dict] = []
    eval_problems: list[dict] = []

    for ptype in ("standard", "ancestor", "collider"):
        probs = buckets[ptype]
        rng.shuffle(probs)
        n_to_train = round(len(probs) * train_frac)
        train_problems.extend(probs[:n_to_train])
        eval_problems.extend(probs[n_to_train:])

    rng.shuffle(train_problems)
    rng.shuffle(eval_problems)

    return train_problems, eval_problems


# ─────────────────────────────────────────────────────────────────────────────
# Prompt construction
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT_BASE = """\
You are an expert in causal inference and graphical models.

You will be given a Directed Acyclic Graph (DAG) representing a structural causal model. \
You will be given both a textual and an image representation of the DAG. \
The textual representation describes the graph as a list of \
directed edges, a treatment node X, and an outcome node Y.

BACKGROUND
----------
A **backdoor path** from X to Y is any path in the graph that begins with \
an arrow INTO X (i.e., it "sneaks around" the front-door causal path). \
Backdoor paths create confounding bias when estimating the causal effect of X on Y.

An **adjustment set** Z is a set of non-descendants of X such that, when we \
condition on Z, all backdoor paths between X and Y are blocked (d-separated \
in the graph with X's outgoing edges removed). Conditioning on Z allows us \
to compute the unconfounded causal effect P(Y | do(X)).

The **minimal adjustment set** is the smallest valid adjustment set — the \
fewest nodes needed. If X and Y are already d-separated in the backdoor \
graph (no open backdoor paths), the minimal set is empty: {}.

IMPORTANT RULES
---------------
- Never include descendants of X in Z (this would open new biases).
- A collider node on a path BLOCKS that path unless conditioned on; \
  conditioning on a collider (or its descendant) OPENS the path — avoid this.
- Your answer must be a subset of the non-descendant, non-X, non-Y nodes.

RESPONSE FORMAT (strict)
------------------------
Your response MUST follow this structure exactly — no exceptions:

1. Open a <reasoning> block in which you write your analysis and explanation.
2. Close the </reasoning> block.
3. Immediately after </reasoning>, write exactly one <answer> block.

CRITICAL RULES for formatting:
- You MUST close </reasoning> before writing <answer>.
- Do NOT write <answer> tags anywhere inside the <reasoning> block.
- There must be EXACTLY ONE <answer> tag in your entire response.
- Use integer node IDs inside curly braces, comma-separated.
- For an empty adjustment set use <answer>{}</answer>."""

_ICL_EXAMPLE_1 = """
Here is a worked example to illustrate the expected reasoning and format.

Example (nodes 0–10; edges 0→1, 0→2, 0→6, 0→8, 0→9, 1→4, 1→8, 1→10, 2→3, 2→4, 2→7, 2→9, \
2→10, 3→4, 3→7, 4→7, 4→9, 5→8, 5→10, 6→7, 6→8, 7→9, 8→9, 8→10; X=8, Y=9):

<reasoning>
Causal paths from X=8 to Y=9: 8→9 (direct).

Descendants of X=8: {9, 10}. Non-descendants eligible for Z: {0, 1, 2, 3, 4, 5, 6, 7}.

Backdoor graph: remove edges 8→9 and 8→10.

Parents of X=8 — the only possible first steps of any backdoor path: {0, 1, 5, 6}.

Verify each parent can reach Y=9 through an open path in the (undirected) backdoor graph:

  Parent 0: 8←0→9 (direct edge 0→9; node 0 is a fork non-collider; open) — must condition on 0.

  Parent 1: 8←1→4→9 (at node 1: fork non-collider; at 4: chain non-collider; open) — must condition on 1.

  Parent 5: Consider all undirected paths starting 8–5.
    Node 5 connects only to 8 (via 5→8) and 10 (via 5→10).
    Every path from 5 toward Y must pass through node 10.
    At node 10: arrows arrive from multiple parents (5→10, 1→10, 2→10). Node 10 is a COLLIDER
    (all arrows point INTO it). A collider blocks the path by default unless we condition on it
    or one of its descendants. Node 10 is a descendant of X=8 (8→10), so we cannot condition on it.
    Therefore ALL backdoor paths through parent 5 are blocked by the un-activatable collider at 10.
    Parent 5 is not a confounder and need not appear in Z.

  Parent 6: 8←6→7→9 (at node 6: fork non-collider; at 7: chain non-collider; open) — must condition on 6.

Active backdoor routes require blocking via parents 0, 1, and 6.

Check that no proper subset of {0, 1, 6} suffices:
  Drop 0: path 8←0→9 remains open.
  Drop 1: path 8←1→4→9 remains open.
  Drop 6: path 8←6→7→9 remains open.

No colliders are inadvertently activated — each of {0, 1, 6} is a non-collider on every path it
lies on, and none is a descendant of X=8.

The minimal adjustment set is therefore {0, 1, 6}.
</reasoning>
<answer>{0, 1, 6}</answer>"""

_ICL_EXAMPLE_2 = """
Here is another worked example. The rendered DAG for this example is shown in the first image immediately following this system message.

Example (nodes 0–8; edges 0→2, 0→4, 0→6, 0→7, 1→2, 1→3, 1→4, 1→7, 2→8, 3→4, 3→5, 4→5, \
4→6, 4→8, 5→8; X=4, Y=8):

<reasoning>
Causal paths from X=4 to Y=8: 4→8 (direct) and 4→5→8.

Descendants of X=4: {5, 6, 8}. Non-descendants eligible for Z: {0, 1, 2, 3, 7}.

Backdoor graph: remove edges 4→5, 4→6, 4→8.

Parents of X=4 — the only possible first steps of any backdoor path: {0, 1, 3}.

Verify each parent independently reaches Y=8 in the backdoor graph:
  Parent 0: 4←0→2→8  (at node 0: fork non-collider; at 2: chain non-collider; open)
  Parent 1: 4←1→2→8  (at node 1: fork non-collider; at 2: chain non-collider; open)
  Parent 3: 4←3→5→8  (at node 3: fork non-collider; at 5: chain non-collider; open)
    Note: node 5 is a descendant of X=4, but since we do NOT condition on 5, the path remains open.
    We must never include descendants of X in Z — but they may still appear on open backdoor paths.

Check for collider shortcuts. Path 4←0→7←1→2→8:
  At node 7: arrows from both 0 (0→7) and 1 (1→7) point INTO 7 — node 7 is a COLLIDER.
  Node 7 is not in Z and not a descendant of X=4, so the collider is not activated; this path
  is blocked. No adjustment for this path is needed.

All three parents open independent backdoor routes and there is no shortcut: each path via parent 0,
1, and 3 reaches Y through a different intermediate chain (node 2 for parents 0 and 1, node 5 for
parent 3).

Check that no proper subset of {0, 1, 3} suffices:
  Drop 0: path 4←0→2→8 remains open.
  Drop 1: path 4←1→2→8 remains open.
  Drop 3: path 4←3→5→8 remains open.

No colliders are inadvertently activated — each of {0, 1, 3} is a non-collider on every path it
lies on, and none is a descendant of X=4.

The minimal adjustment set is therefore {0, 1, 3}.
</reasoning>
<answer>{0, 1, 3}</answer>"""

SYSTEM_PROMPT_1 = _SYSTEM_PROMPT_BASE + _ICL_EXAMPLE_1 + _ICL_EXAMPLE_2

# Pre-render the second ICL example's DAG at module load time so it can be
# injected into the system message at rollout time without re-computing it.
_ICL2_G = nx.DiGraph([
    (0, 2), (0, 4), (0, 6), (0, 7),
    (1, 2), (1, 3), (1, 4), (1, 7),
    (2, 8),
    (3, 4), (3, 5),
    (4, 5), (4, 6), (4, 8),
    (5, 8),
])
_ICL_EXAMPLE_2_B64 = _render_dag_b64(_ICL2_G, X=4, Y=8)


def format_problem(edges: list, nodes: list, X: int, Y: int) -> str:
    """Render a DAG problem as a readable string for the model."""
    # Build parent/child lookup
    parents: dict[int, list[int]] = {n: [] for n in nodes}
    children: dict[int, list[int]] = {n: [] for n in nodes}
    for u, v in edges:
        children[u].append(v)
        parents[v].append(u)

    edge_str = ", ".join(f"{u}→{v}" for u, v in sorted(edges))
    node_str = ", ".join(str(n) for n in sorted(nodes))

    adj_lines = []
    for n in sorted(nodes):
        pa = sorted(parents[n])
        ch = sorted(children[n])
        adj_lines.append(
            f"  Node {n}: parents=[{', '.join(map(str, pa))}]  "
            f"children=[{', '.join(map(str, ch))}]"
        )
    adj_str = "\n".join(adj_lines)

    return (
        f"Here is the textual representation of the DAG: \n"
        f"Nodes: {node_str}\n"
        f"Edges: {edge_str}\n\n"
        f"Adjacency:\n{adj_str}\n\n"
        f"Treatment (X): {X}\n"
        f"Outcome   (Y): {Y}\n\n"
        f"What is the minimal adjustment set Z that blocks all backdoor "
        f"paths from {X} to {Y}?\n"
        f"The rendered image of this DAG is also provided "
        f"(blue node = treatment X, orange node = outcome Y).\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Answer parsing
# ─────────────────────────────────────────────────────────────────────────────


def parse_answer(content: str) -> set[int] | None:
    """Extract the adjustment set from the model's <answer> tag.

    Returns a set of ints, or None if the format is invalid.
    """
    stripped = re.sub(r"<reasoning>.*?</reasoning>", "", content, flags=re.DOTALL)
    matches = re.findall(r"<answer>\s*(.*?)\s*</answer>", stripped, flags=re.DOTALL)
    if len(matches) != 1:
        return None
    inner = matches[0].strip()
    m = re.match(r"^\{(.*)\}$", inner, re.DOTALL)
    if not m:
        return None
    body = m.group(1).strip()
    if body == "":
        return set()
    try:
        return {int(x.strip()) for x in body.split(",")}
    except ValueError:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Reward functions
# ─────────────────────────────────────────────────────────────────────────────


async def adjustment_set_accuracy(completion, info) -> float:
    """Reward: correctness of the predicted minimal adjustment set.

    Scoring:
      - Exact match → 1.0
      - Partial match → Jaccard similarity (intersection / union)
      - Unparseable answer → 0.0
    """
    content = completion[-1]["content"]
    predicted = parse_answer(content)
    if predicted is None:
        return 0.0
    gold = set(info["minimal_adjustment_set"])
    if predicted == gold:
        return 1.0
    union = len(predicted | gold)
    if union == 0:
        return 1.0  # both empty
    return len(predicted & gold) / union


async def format_compliance(completion) -> float:
    """Reward: whether the response follows the required XML format.

    Returns 1.0 if <answer>{...}</answer> is correctly present, else 0.0.
    """
    content = completion[-1]["content"]
    return 1.0 if parse_answer(content) is not None else 0.0


async def valid_adjustment_set(completion, info) -> float:
    """Reward: whether the predicted set is a valid (not necessarily minimal) adjustment set.

    A set Z is valid if:
      1. No node in Z is a descendant of X (post-treatment bias check).
      2. Z d-separates X from Y in the backdoor graph (G with X's outgoing edges removed).

    Returns 1.0 if valid, 0.0 otherwise (including unparseable answers).
    """
    content = completion[-1]["content"]
    predicted = parse_answer(content)
    if predicted is None:
        return 0.0

    X = info["X"]
    Y = info["Y"]

    G = nx.DiGraph()
    G.add_nodes_from(info["nodes"])
    G.add_edges_from(info["edges"])

    # Descendant check: no node in Z may be a descendant of X
    if predicted & nx.descendants(G, X):
        return 0.0

    # Build backdoor graph: remove all outgoing edges from X
    G_bd = G.copy()
    G_bd.remove_edges_from(list(G.out_edges(X)))

    # D-separation check
    return 1.0 if is_d_separator(G_bd, {X}, {Y}, predicted) else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Dataset builder
# ─────────────────────────────────────────────────────────────────────────────


def build_dataset(problems: list[dict]) -> Dataset:
    """Convert a list of problem dicts into a HuggingFace Dataset."""
    rows = []
    for p in problems:
        rows.append({
            "question": format_problem(p["edges"], p["nodes"], p["X"], p["Y"]),
            "info": json.dumps({
                "minimal_adjustment_set": p["minimal_adjustment_set"],
                "X": p["X"],
                "Y": p["Y"],
                "edges": p["edges"],
                "nodes": p["nodes"],
                "num_nodes": p["num_nodes"],
                "num_backdoor_paths": p["num_backdoor_paths"],
                "num_parents_X": p.get("num_parents_X"),
                "problem_type": p.get("problem_type"),
            }),
        })
    return Dataset.from_list(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Environment subclass
# ─────────────────────────────────────────────────────────────────────────────


class CausalReasoningEnv(vf.SingleTurnEnv):
    """SingleTurnEnv that renders each DAG as a PNG and injects it into the prompt.

    setup_state runs once at the start of each rollout. It reconstructs the
    graph from state["info"], renders it as a base64 PNG, and replaces the
    last user message's plain-string content with a [text, image_url] list —
    the multimodal format expected by vision-capable models.
    """

    async def setup_state(self, state: vf.State, **kwargs) -> vf.State:
        info = state["info"]
        G = nx.DiGraph()
        G.add_nodes_from(info["nodes"])
        G.add_edges_from(info["edges"])

        b64 = _render_dag_b64(G, info["X"], info["Y"])

        prompt = list(state["prompt"])

        # Upgrade system message to multimodal: append the ICL example 2 DAG image
        # so the model sees the visual alongside the worked example text.
        sys_idx = next((i for i, m in enumerate(prompt) if m["role"] == "system"), None)
        if sys_idx is not None and isinstance(prompt[sys_idx]["content"], str):
            prompt[sys_idx] = {
                "role": "system",
                "content": [
                    {"type": "text", "text": prompt[sys_idx]["content"]},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_ICL_EXAMPLE_2_B64}"}},
                ],
            }

        # Locate the last user message and upgrade its content to [text, image]
        last_user_idx = max(i for i, m in enumerate(prompt) if m["role"] == "user")
        original_text = prompt[last_user_idx]["content"]
        prompt[last_user_idx] = {
            "role": "user",
            "content": [
                {"type": "text", "text": original_text},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ],
        }
        state["prompt"] = prompt

        return await super().setup_state(state, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


_NUM_TRAIN = 250
_NUM_EVAL = 100
_MIN_NODES = 8
_MAX_NODES = 12
_SEED = 42

# Paths where __main__ saves and load_environment() can load from.
_DATASET_DIR = pathlib.Path(__file__).parent / "datasets"
_TRAIN_DATASET_PATH = _DATASET_DIR / "train"
_EVAL_DATASET_PATH = _DATASET_DIR / "eval"


def load_environment(
    train_dataset: Dataset | None = None,
    eval_dataset: Dataset | None = None,
) -> vf.Environment:
    """Load the CausalReasoningEnv_1 environment.

    If datasets are not passed, attempts to auto-load pre-built datasets from
    disk. If no datasets exist on disk, generates them from scratch (slow; run
    __main__ to pre-build and save them).

    Args:
        train_dataset: Pre-built HuggingFace Dataset for training.
        eval_dataset: Pre-built HuggingFace Dataset for evaluation.

    Auto-load example (reads from disk if available):
        env = load_environment()
    """
    from datasets import load_from_disk

    if train_dataset is None and _TRAIN_DATASET_PATH.exists():
        train_dataset = load_from_disk(str(_TRAIN_DATASET_PATH))
    if eval_dataset is None and _EVAL_DATASET_PATH.exists():
        eval_dataset = load_from_disk(str(_EVAL_DATASET_PATH))

    if train_dataset is None or eval_dataset is None:
        train_problems, eval_problems = generate_stratified_dag_problems(
            n_train=_NUM_TRAIN,
            n_eval=_NUM_EVAL,
            min_nodes=_MIN_NODES,
            max_nodes=_MAX_NODES,
            seed=_SEED,
        )
        if train_dataset is None:
            train_dataset = build_dataset(train_problems)
        if eval_dataset is None:
            eval_dataset = build_dataset(eval_problems)

    rubric = vf.Rubric(
        funcs=[adjustment_set_accuracy, valid_adjustment_set, format_compliance],
        weights=[0.9, 0.0, 0.1],
    )

    return CausalReasoningEnv(
        dataset=train_dataset,
        eval_dataset=eval_dataset,
        system_prompt=SYSTEM_PROMPT_1,
        rubric=rubric,
    )


if __name__ == "__main__":
    import base64 as _b64
    from collections import Counter

    # ── Generate stratified train + eval pools ────────────────────────────────
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

    # ── Build and save HuggingFace Datasets to disk ───────────────────────────
    train_dataset = build_dataset(train_problems)
    eval_dataset = build_dataset(eval_problems)

    _DATASET_DIR.mkdir(parents=True, exist_ok=True)
    train_dataset.save_to_disk(str(_TRAIN_DATASET_PATH))
    eval_dataset.save_to_disk(str(_EVAL_DATASET_PATH))
    print(f"\nTrain dataset ({len(train_dataset)} rows) saved → {_TRAIN_DATASET_PATH}")
    print(f"Eval  dataset ({len(eval_dataset)} rows) saved → {_EVAL_DATASET_PATH}\n")

    # ── Build signature set for ICL exclusion ────────────────────────────────
    all_sigs: set[tuple] = {
        (frozenset((int(u), int(v)) for u, v in p["edges"]), p["X"], p["Y"])
        for p in train_problems + eval_problems
    }

    # ── Generate 2 ICL examples disjoint from train+eval ─────────────────────
    _ICL_SEED = 77
    print(f"Generating ICL examples (seed={_ICL_SEED})…")
    icl_rng = random.Random(_ICL_SEED)
    icl_problems: list[dict] = []
    while len(icl_problems) < 2:
        prob = _try_sample_problem(icl_rng, _MIN_NODES, _MAX_NODES, 0.41)
        if prob is None:
            continue
        sig = (frozenset((int(u), int(v)) for u, v in prob["edges"]), prob["X"], prob["Y"])
        if sig in all_sigs:
            continue
        ptype = _classify_problem(prob)
        G_icl: nx.DiGraph = prob.pop("G")
        parents_X_icl = set(G_icl.predecessors(prob["X"]))
        prob["num_parents_X"] = len(parents_X_icl)
        prob["problem_type"] = ptype
        icl_problems.append(prob)
        all_sigs.add(sig)

    print(f"ICL examples: {len(icl_problems)} unique problems (seed={_ICL_SEED})\n")

    for i, p in enumerate(icl_problems):
        edges_str = ", ".join(f"{u}→{v}" for u, v in sorted(p["edges"]))
        print(f"{'='*70}")
        print(f"ICL Example {i + 1}")
        print(f"{'='*70}")
        print(f"  nodes                  = {sorted(p['nodes'])}")
        print(f"  edges                  = {edges_str}")
        print(f"  X={p['X']}, Y={p['Y']}")
        print(f"  minimal_adjustment_set = {p['minimal_adjustment_set']}")
        print(f"  num_parents_X={p['num_parents_X']}, problem_type={p['problem_type']}")
        print(f"  num_nodes={p['num_nodes']}, num_backdoor_paths={p['num_backdoor_paths']}")
        print(f"  formatted problem:")
        print(format_problem(p["edges"], p["nodes"], p["X"], p["Y"]))

        G_print = nx.DiGraph()
        G_print.add_nodes_from(p["nodes"])
        G_print.add_edges_from(p["edges"])
        img_path = f"./icl_example_{i + 1}.png"
        with open(img_path, "wb") as fh:
            fh.write(_b64.b64decode(_render_dag_b64(G_print, p["X"], p["Y"])))
        print(f"  DAG image saved → {img_path}\n")
