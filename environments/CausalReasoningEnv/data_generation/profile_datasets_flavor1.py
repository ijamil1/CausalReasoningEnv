"""Data profile of the Flavor 1 train and eval datasets.

Usage:
    uv run profile_datasets.py

Produces:
    - Console tables for all distributions
    - profile_datasets_flavor1.png  (multi-panel figure)
"""

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "datasets",
#   "networkx>=3.0",
#   "matplotlib>=3.7",
#   "pandas",
# ]
# ///

import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from datasets import load_from_disk

sys.path.insert(0, str(pathlib.Path(__file__).parent))

# ── paths ─────────────────────────────────────────────────────────────────────

_DIR = pathlib.Path(__file__).parent / "datasets" / "flavor1"
train_ds = load_from_disk(str(_DIR / "train"))
eval_ds  = load_from_disk(str(_DIR / "eval"))

splits = {"train": train_ds, "eval": eval_ds}

# ── build DataFrames ──────────────────────────────────────────────────────────

def to_df(ds) -> pd.DataFrame:
    rows = []
    for ex in ds:
        info = json.loads(ex["info"])
        n_total  = info["num_nodes"]
        n_latent = len(info["latent_nodes"])
        min_sets = info["minimal_adjustment_sets"]  # list of lists or None
        min_set_size = len(min_sets[0]) if min_sets is not None else None
        n_parents_X  = info.get("num_parents_X")
        ratio = min_set_size / n_parents_X if (min_set_size is not None and n_parents_X) else None
        rows.append({
            "num_nodes":              n_total,
            "latent_ratio":           n_latent / n_total,
            "identifiability_status": info["identifiability_status"],
            "min_set_size":           min_set_size,
            "num_parents_X":          n_parents_X,
            "min_set_over_parents":   ratio,
        })
    return pd.DataFrame(rows)

dfs = {name: to_df(ds) for name, ds in splits.items()}

# ── console report ────────────────────────────────────────────────────────────

STATUS_ORDER = [
    "identifiable",
    "identifiable_frontdoor",
    "empty",
    "not_identifiable",
]

for name, df in dfs.items():
    print(f"\n{'='*60}")
    print(f"  {name.upper()}  ({len(df)} examples)")
    print(f"{'='*60}")

    print("\n── num_nodes distribution ──")
    print(df["num_nodes"].value_counts().sort_index().to_string())

    print("\n── latent/total ratio distribution (rounded to 1 dp) ──")
    print(df["latent_ratio"].round(1).value_counts().sort_index().to_string())

    print("\n── identifiability_status distribution ──")
    print(df["identifiability_status"].value_counts().to_string())

    sub = df[df["identifiability_status"] == "identifiable"].dropna(subset=["min_set_over_parents"])
    if not sub.empty:
        print("\n── min_set_size / num_parents_X  (identifiable only) ──")
        print(sub["min_set_over_parents"].round(2).value_counts().sort_index().to_string())

# ── DAG plotting helpers ───────────────────────────────────────────────────────

def _dag_layout(G: nx.DiGraph) -> dict:
    pos = {}
    for depth, layer in enumerate(nx.topological_generations(G)):
        layer = sorted(layer)
        for i, node in enumerate(layer):
            pos[node] = ((i - (len(layer) - 1) / 2.0), -float(depth))
    return pos


def _plot_dag(ax, ex, title: str):
    info         = json.loads(ex["info"])
    edges        = info["edges"]
    nodes        = info["nodes"]
    latent_nodes = set(info["latent_nodes"])
    X, Y         = info["X"], info["Y"]

    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    pos = _dag_layout(G)

    def _color(n):
        if n == X:            return "#4C72B0"
        if n == Y:            return "#DD8452"
        if n in latent_nodes: return "#D4B8E0"
        return "#C8C8C8"

    node_colors = [_color(n) for n in G.nodes()]
    labels      = {n: f"{n}(L)" if n in latent_nodes else str(n) for n in G.nodes()}
    font_colors = {n: "white" if n in (X, Y) else "black" for n in G.nodes()}

    node_size = 600
    nx.draw_networkx_nodes(G, pos=pos, ax=ax, node_color=node_colors, node_size=node_size)
    nx.draw_networkx_edges(G, pos=pos, ax=ax, arrows=True, arrowsize=18,
                           edge_color="#555555", width=1.5, node_size=node_size,
                           connectionstyle="arc3,rad=0.15")
    for node, (x, y) in pos.items():
        ax.text(x, y, labels[node], ha="center", va="center",
                fontsize=8, color=font_colors[node], fontweight="bold")

    status = info["identifiability_status"]
    min_sets = info.get("minimal_adjustment_sets")
    med      = info.get("mediator_node")
    if min_sets is not None:
        subtitle = f"adj={list(min_sets[0])}"
    elif med is not None:
        subtitle = f"mediator={med}"
    else:
        subtitle = ""
    ax.set_title(f"{title}\n{status}\n{subtitle}", fontsize=8)
    ax.axis("off")


# ── select one example per status ────────────────────────────────────────────

# Use train split for examples
examples_by_status: dict[str, dict] = {}
for ex in train_ds:
    s = json.loads(ex["info"])["identifiability_status"]
    if s not in examples_by_status:
        examples_by_status[s] = ex

# ── build figure ──────────────────────────────────────────────────────────────

# Layout: 4 rows × 4 cols
#   Row 0-1: distribution bar charts (across both splits)
#   Row 2-3: example DAGs

fig = plt.figure(figsize=(18, 20))
gs  = fig.add_gridspec(4, 4, hspace=0.5, wspace=0.4)

# Helper: bar chart on an axis
def bar_chart(ax, series_dict: dict, title: str, xlabel: str, ylabel: str = "count"):
    """series_dict: {split_name: pd.Series(value→count)}"""
    all_vals = sorted(set().union(*[s.index for s in series_dict.values()]))
    x        = range(len(all_vals))
    width    = 0.35
    colors   = {"train": "#4C72B0", "eval": "#DD8452"}
    for i, (split, series) in enumerate(series_dict.items()):
        heights = [series.get(v, 0) for v in all_vals]
        offset  = (i - 0.5) * width
        ax.bar([xi + offset for xi in x], heights, width, label=split, color=colors[split], alpha=0.85)
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(v) for v in all_vals], rotation=30, ha="right", fontsize=7)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.legend(fontsize=7)
    ax.tick_params(axis="y", labelsize=7)

# ── Row 0, cols 0-1: num_nodes distribution ──────────────────────────────────
ax_nodes = fig.add_subplot(gs[0, 0:2])
bar_chart(
    ax_nodes,
    {name: df["num_nodes"].value_counts().sort_index() for name, df in dfs.items()},
    title="Number of nodes",
    xlabel="num_nodes",
)

# ── Row 0, cols 2-3: latent ratio distribution ───────────────────────────────
ax_lat = fig.add_subplot(gs[0, 2:4])
bar_chart(
    ax_lat,
    {name: df["latent_ratio"].round(2).value_counts().sort_index() for name, df in dfs.items()},
    title="Latent / total nodes ratio",
    xlabel="ratio (rounded to 2 dp)",
)

# ── Row 1, cols 0-1: identifiability_status distribution ─────────────────────
ax_status = fig.add_subplot(gs[1, 0:2])
bar_chart(
    ax_status,
    {name: df["identifiability_status"].value_counts() for name, df in dfs.items()},
    title="Identifiability status",
    xlabel="status",
)

# ── Row 1, cols 2-3: min_set_size / num_parents_X (identifiable only) ────────
ax_ratio = fig.add_subplot(gs[1, 2:4])
ratio_series = {}
for name, df in dfs.items():
    sub = df[df["identifiability_status"] == "identifiable"].dropna(subset=["min_set_over_parents"])
    ratio_series[name] = sub["min_set_over_parents"].round(2).value_counts().sort_index()
bar_chart(
    ax_ratio,
    ratio_series,
    title="min_set_size / num_parents_X\n(identifiable problems only)",
    xlabel="ratio",
)

# ── Rows 2-3: one example DAG per status ─────────────────────────────────────
dag_axes = [
    fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1]),
    fig.add_subplot(gs[2, 2]), fig.add_subplot(gs[2, 3]),
]
for ax, status in zip(dag_axes, STATUS_ORDER):
    ex = examples_by_status.get(status)
    if ex is not None:
        _plot_dag(ax, ex, title=status)
    else:
        ax.set_title(f"{status}\n(no example)", fontsize=8)
        ax.axis("off")

# Legend row (row 3 col 0)
ax_legend = fig.add_subplot(gs[3, 0])
ax_legend.axis("off")
handles = [
    mpatches.Patch(color="#4C72B0", label="X (treatment)"),
    mpatches.Patch(color="#DD8452", label="Y (outcome)"),
    mpatches.Patch(color="#C8C8C8", label="observed"),
    mpatches.Patch(color="#D4B8E0", label="latent (L)"),
]
ax_legend.legend(handles=handles, loc="center", fontsize=9, title="Node legend", title_fontsize=9)

fig.suptitle("Flavor 1 dataset profile", fontsize=14, fontweight="bold")

out_path = pathlib.Path(__file__).parent / "profile_datasets_flavor1.png"
fig.savefig(str(out_path), dpi=120, bbox_inches="tight")
print(f"\nFigure saved → {out_path}")
