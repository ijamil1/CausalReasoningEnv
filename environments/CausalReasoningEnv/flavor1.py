"""Flavor 1 — Minimal Adjustment Set Identification.

Given a randomly generated DAG with a designated treatment node X and
outcome node Y, the model must identify the minimal adjustment set Z:
the smallest set of non-descendants of X whose conditioning blocks all
backdoor paths between X and Y (via d-separation in the backdoor graph).

Environment type: SingleTurnEnv subclass. The DAG is rendered as a PNG
at rollout start and injected into the prompt alongside the text description.
"""

import base64
import io
import pathlib
import re

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; must be set before pyplot import
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import verifiers as vf
from networkx.algorithms.d_separation import is_d_separator

from data_generation.flavor1_gen import (
    build_dataset,
    generate_stratified_dag_problems,
)
from prompts import build_system_prompt


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


def _render_dag_b64(
    G: nx.DiGraph,
    X: int,
    Y: int,
    latent_nodes: set[int] | None = None,
    figsize=(8, 6),
    dpi=100,
) -> str:
    """Render a DAG as a base64-encoded PNG string.

    X (treatment) is blue, Y (outcome) is orange, latent nodes are light
    purple with an italic "(L)" label suffix, observed nodes are gray.
    Layout is topological so causal flow reads top-to-bottom.
    """
    latent_nodes = set(latent_nodes) if latent_nodes else set()
    pos = _dag_layout(G)

    def _color(n):
        if n == X:       return "#4C72B0"
        if n == Y:       return "#DD8452"
        if n in latent_nodes: return "#D4B8E0"  # light purple
        return "#C8C8C8"

    node_colors = [_color(n) for n in G.nodes()]
    font_colors = {n: "white" if n in (X, Y) else "black" for n in G.nodes()}
    labels = {n: f"{n}(L)" if n in latent_nodes else str(n) for n in G.nodes()}

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    node_size = 700
    nx.draw_networkx_nodes(G, pos=pos, ax=ax, node_color=node_colors, node_size=node_size)
    nx.draw_networkx_edges(G, pos=pos, ax=ax, arrows=True, arrowsize=20,
                           edge_color="#555555", width=1.5,
                           node_size=node_size,
                           connectionstyle="arc3,rad=0.15")
    for node, (x, y) in pos.items():
        ax.text(x, y, labels[node], ha="center", va="center",
                fontsize=9, color=font_colors[node], fontweight="bold")

    legend_handles = [
        mpatches.Patch(color="#4C72B0", label=f"X = {X}  (treatment)"),
        mpatches.Patch(color="#DD8452", label=f"Y = {Y}  (outcome)"),
        mpatches.Patch(color="#C8C8C8", label="observed"),
    ]
    if latent_nodes:
        legend_handles.append(mpatches.Patch(color="#D4B8E0", label="latent (L)"))
    ax.legend(handles=legend_handles, loc="upper left",
              bbox_to_anchor=(1.02, 1), borderaxespad=0, fontsize=9)
    ax.set_title("Causal DAG", fontsize=12)
    ax.axis("off")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode()


# ─────────────────────────────────────────────────────────────────────────────
# Prompt construction
# ─────────────────────────────────────────────────────────────────────────────


_F1_INTRO = """\
You will be given a Directed Acyclic Graph (DAG) representing a structural \
causal model, along with a visual rendering of the same graph. Nodes are \
variables. Nodes may be observed or latent/unobserved. A directed edge A→B means A is a direct cause of B.\
"""

_F1_RESPONSE_FORMAT = """\
RESPONSE FORMAT
────────────────
You must follow the following format exactly. Write all reasoning inside a <reasoning> block. After </reasoning>, write \
exactly one <answer> block using one of these three forms:

  Backdoor (adjustment set, including empty set):
      <answer>{N1, N2, ...}</answer>   or   <answer>{}</answer>

  Front-door (mediator set M — use even if M is a single node):
      <answer>frontdoor: {M1, M2, ...}</answer>

  Not identifiable:
      <answer>not_identifiable</answer>

Rules: do not write <answer> inside <reasoning>; use integer node IDs.
"""

SYSTEM_PROMPT = build_system_prompt(_F1_INTRO, _F1_RESPONSE_FORMAT)


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
        f"DAG:\n"
        f"Nodes:    {node_str}\n"
        f"Observed: {obs_str}\n"
        f"Latent:   {lat_str}\n"
        f"Edges:    {edge_str}\n\n"
        f"Adjacency:\n{adj_str}\n\n"
        f"Treatment (X): {X}\n"
        f"Outcome   (Y): {Y}\n\n"
        f"Determine whether the average treatment effect "
        f"ATE = E[Y | do(X=1)] − E[Y | do(X=0)] is identifiable from the "
        f"observed variables. If it is, state the identification method and "
        f"the required variables. If it is not, respond with not_identifiable in the answer tags. RESPOND ACCORDING TO THE RESPONSE FORMAT SPECIFIED EARLIER.\n\n"
        f"A visual rendering of this DAG is also provided "
        f"(blue = X, orange = Y, gray = observed, purple = latent)."
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
# Environment class
# ─────────────────────────────────────────────────────────────────────────────


class Flavor1Env(vf.SingleTurnEnv):
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

        b64 = _render_dag_b64(
            G, info["X"], info["Y"],
            latent_nodes=set(info.get("latent_nodes", [])),
        )

        prompt = list(state["prompt"])

        # Upgrade last user message to [text, image].
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

    return Flavor1Env(
        dataset=train_dataset,
        eval_dataset=eval_dataset,
        system_prompt=SYSTEM_PROMPT,
        rubric=rubric,
    )


if __name__ == "__main__":
    from collections import Counter

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
