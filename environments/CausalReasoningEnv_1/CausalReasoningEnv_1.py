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


def generate_dag_problems(
    n_graphs: int = 200,
    min_nodes: int = 7,
    max_nodes: int = 14,
    edge_prob: float = 0.35,
    seed: int = 42,
    exclude: set[frozenset] | None = None,
) -> list[dict]:
    """Generate a list of causal adjustment-set problems via rejection sampling.

    Each problem satisfies:
      - Y is a descendant of X (a causal path exists).
      - Y is a leaf node (no outgoing edges).
      - At least 2 backdoor paths exist, with at least one of length > 3.
      - No two problems share the same (edges, X, Y) signature (uniqueness).

    Args:
        exclude: Optional set of (frozenset(edges), X, Y) signatures to reject,
                 used to guarantee disjointness from an existing problem pool.

    Returns a list of dicts with keys:
        edges, nodes, X, Y, minimal_adjustment_set, num_nodes, num_backdoor_paths.
    """
    rng = random.Random(seed)
    problems: list[dict] = []
    seen: set[tuple] = set(exclude) if exclude else set()

    while len(problems) < n_graphs:
        n = rng.randint(min_nodes, max_nodes)
        G = _make_dag(n, edge_prob, rng)
        nodes_list = list(G.nodes())
        if len(nodes_list) < 2:
            continue

        X, Y = rng.sample(nodes_list, 2)

        # Y must be reachable from X
        if not nx.has_path(G, X, Y):
            continue
        # Y must be a leaf
        if G.out_degree(Y) > 0:
            continue
        # No cycles (implicit in DAG construction, but guard anyway)
        if nx.has_path(G, Y, X):
            continue

        # Backdoor graph: remove all edges out of X
        G_bd = G.copy()
        G_bd.remove_edges_from(list(G.out_edges(X)))

        try:
            bd_paths = list(nx.all_simple_paths(G_bd.to_undirected(), X, Y))
            if len(bd_paths) < 2 or not any(len(p) > 3 for p in bd_paths):
                continue
            min_set = find_minimal_d_separator(G_bd, X, Y)
        except Exception:
            continue

        # Uniqueness check
        sig = (frozenset((int(u), int(v)) for u, v in G.edges()), int(X), int(Y))
        if sig in seen:
            continue
        seen.add(sig)

        problems.append({
            "edges": [(int(u), int(v)) for u, v in G.edges()],
            "nodes": [int(n) for n in G.nodes()],
            "X": int(X),
            "Y": int(Y),
            "minimal_adjustment_set": sorted(int(n) for n in min_set),
            "num_nodes": len(nodes_list),
            "num_backdoor_paths": len(bd_paths),
        })

    return problems


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

Example (nodes 0,1,3,4,5; edges 0→1, 0→3, 1→3, 1→5, 3→5, 4→5; X=3, Y=5):

<reasoning>
Causal path from X=3 to Y=5: 3→5. This is the front-door path; it is not a backdoor path.

Descendants of X=3: node 5 (via 3→5). All other nodes (0, 1, 4) are non-descendants and are candidates for Z.

Backdoor paths start with an arrow INTO X=3. Node 3 has two parents: 0 and 1.

Enumerate all backdoor paths:
  Path 1: 3←1→5. Node 1 is a non-collider on this path, so the path is open.
  Path 2: 3←0→1→5. Nodes 0 and 1 are non-colliders, so this path is open.
  (Node 4→5 creates no backdoor path because 4 is not a parent of X=3.)

To block Path 1 (3←1→5) I must condition on node 1.
Conditioning on node 1 also blocks Path 2 (3←0→1→5) because 1 lies on that path too.

Could {0} alone work? Node 0 blocks Path 2 but does NOT block Path 1 (3←1→5), so {0} is insufficient.

The minimal adjustment set is therefore {1}.
</reasoning>
<answer>{1}</answer>"""

_ICL_EXAMPLE_2 = """
Here is another worked example. The rendered DAG for this example is shown in the first image immediately following this system message.

Example (nodes 0,1,2,3,4,5,6,7; edges 0→1, 0→2, 0→3, 0→6, 0→7, 1→5, 2→4, 3→7, 4→6, 5→6; X=2, Y=6):

<reasoning>
Causal path from X=2 to Y=6: 2→4→6. This is the front-door path; it is not a backdoor path.

Descendants of X=2: nodes 4 (via 2→4) and 6 (via 2→4→6). Non-descendants are 0, 1, 3, 5, 7.

Backdoor paths start with an arrow INTO X=2. Node 2 has exactly one parent: node 0.

Enumerate all backdoor paths (all must start 2←0):
  Path 1: 2←0→6. Node 0 is a non-collider; path is open.
  Path 2: 2←0→1→5→6. Nodes 0, 1, 5 are all non-colliders; path is open.
  (0→3→7 and 0→7 are dead ends that do not reach Y=6.)

Both paths pass through node 0 — the sole parent of X. Conditioning on 0 blocks both simultaneously.
Node 0 is a non-descendant of X=2, so it is a valid conditioning node.

Could a different single node work?
  {1} blocks Path 2 but not Path 1 (2←0→6 bypasses 1). Insufficient.
  {5} blocks Path 2 but not Path 1. Insufficient.

The minimal adjustment set is therefore {0}.
</reasoning>
<answer>{0}</answer>"""

SYSTEM_PROMPT_1 = _SYSTEM_PROMPT_BASE + _ICL_EXAMPLE_1 + _ICL_EXAMPLE_2

# Pre-render the second ICL example's DAG at module load time so it can be
# injected into the system message at rollout time without re-computing it.
_ICL2_G = nx.DiGraph([
    (0, 1), (0, 2), (0, 3), (0, 6), (0, 7),
    (1, 5), (2, 4), (3, 7), (4, 6), (5, 6),
])
_ICL_EXAMPLE_2_B64 = _render_dag_b64(_ICL2_G, X=2, Y=6)


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
_MIN_NODES = 5
_MAX_NODES = 8
_SEED = 42


def load_environment() -> vf.Environment:
    """Load the CausalReasoningEnv_1 environment.

    Generates disjoint train/eval splits of DAG adjustment-set problems.
    All parameters are fixed to ensure identical datasets across every run.
    """
    all_problems = generate_dag_problems(
        n_graphs=_NUM_TRAIN + _NUM_EVAL,
        min_nodes=_MIN_NODES,
        max_nodes=_MAX_NODES,
        seed=_SEED,
    )
    train_dataset = build_dataset(all_problems[:_NUM_TRAIN])
    eval_dataset = build_dataset(all_problems[_NUM_TRAIN:])

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

    # ── Reproduce the fixed train+eval pool ──────────────────────────────────
    all_problems = generate_dag_problems(
        n_graphs=_NUM_TRAIN + _NUM_EVAL,
        min_nodes=_MIN_NODES,
        max_nodes=_MAX_NODES,
        seed=_SEED,
    )

    # Build a set of their signatures for exclusion
    train_eval_sigs = {
        (frozenset((int(u), int(v)) for u, v in p["edges"]), p["X"], p["Y"])
        for p in all_problems
    }
    print(f"Train+eval pool: {len(all_problems)} unique problems (seed={_SEED})")

    # ── Generate 2 ICL examples disjoint from train+eval ────────────────────
    _ICL_SEED = 77
    icl_problems = generate_dag_problems(
        n_graphs=2,
        min_nodes=_MIN_NODES,
        max_nodes=_MAX_NODES,
        seed=_ICL_SEED,
        exclude=train_eval_sigs,
    )

    print(f"ICL examples: {len(icl_problems)} unique problems (seed={_ICL_SEED})\n")

    for i, p in enumerate(icl_problems):
        edges_str = ", ".join(f"{u}→{v}" for u, v in sorted(p["edges"]))
        print(f"{'='*70}")
        print(f"ICL Example {i + 1}")
        print(f"{'='*70}")
        print(f"  nodes                 = {sorted(p['nodes'])}")
        print(f"  edges                 = {edges_str}")
        print(f"  X={p['X']}, Y={p['Y']}")
        print(f"  minimal_adjustment_set = {p['minimal_adjustment_set']}")
        print(f"  num_nodes={p['num_nodes']}, num_backdoor_paths={p['num_backdoor_paths']}")
        print(f"  formatted problem:")
        print(format_problem(p["edges"], p["nodes"], p["X"], p["Y"]))

        G = nx.DiGraph()
        G.add_nodes_from(p["nodes"])
        G.add_edges_from(p["edges"])
        img_path = f"/tmp/icl_example_{i + 1}.png"
        with open(img_path, "wb") as fh:
            fh.write(_b64.b64decode(_render_dag_b64(G, p["X"], p["Y"])))
        print(f"  DAG image saved → {img_path}\n")
