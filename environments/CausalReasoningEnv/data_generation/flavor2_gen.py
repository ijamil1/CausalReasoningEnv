"""Flavor 2 data generation — ATE Estimation (Analytical + Nonparametric).

SCM-only problems (~20% of problems): Linear SCM, binary X root node.
  Model computes exact ATE via directed path-tracing (Wright's rule).
  No data provided.

Data-only problems (~80% of problems): Discrete CPT SCM with observational data.
  Model identifies whether ATE is identifiable, then estimates ATE
  nonparametrically from the data.

─────────────────────────────────────────────────────────────────────────────
SCM-ONLY PROBLEM TYPES (subcase="A")
─────────────────────────────────────────────────────────────────────────────
  "standard"  (~40%): 1–2 directed X→Y paths; ATE ≠ 0; no sign cancellation.
  "mediated"  (~30%): ≥2 directed paths through mediators; no direct X→Y edge.
  "canceling" (~20%): ≥2 opposing-sign paths; each |contribution| ≥ 0.4;
                       |ATE| ≤ 0.05.
  "no_path"   (~10%): no directed X→Y path; ATE = 0.

─────────────────────────────────────────────────────────────────────────────
DATA-ONLY PROBLEM TYPES (subcase="B")
─────────────────────────────────────────────────────────────────────────────
  "backdoor_standard" (~35%): non-empty minimal observed adjustment set;
                               full support; model estimates ATE and CATE(z₀).
  "backdoor_empty"    (~20%): empty adjustment set; model estimates ATE and CATE.
  "frontdoor"         (~20%): latent U→X, U→Y; valid frontdoor mediator M.
                               Model applies two-step frontdoor formula.
  "not_identifiable"  (~25%): latent U→X, U→Y; no valid backdoor or frontdoor.

─────────────────────────────────────────────────────────────────────────────
DATA FIELDS (per problem dict)
─────────────────────────────────────────────────────────────────────────────
  subcase               "A" or "B"
  problem_type          specific sub-type string
  edges                 list of (u, v) directed edge pairs
  nodes                 all node ids (observed + latent)
  X, Y                  treatment and outcome node ids
  observed_nodes        sorted list of observed node ids
  latent_nodes          sorted list of latent node ids

  Sub-case A only:
    structural_equations_text  human-readable structural equations
    coefficients               dict "u,v" → float
    sigmas                     dict str(node) → float
    X_prob                     Bernoulli parameter for X

  Sub-case B only:
    data_csv             CSV string of observational data (observed columns only)
    adjustment_set       minimal adjustment set (list) or None
    mediator_node        mediator node id for frontdoor, else None
  Both:
    identifiability_status  "identifiable" | "not_identifiable"
    true_ATE             exact ATE (from path-tracing for A; CPT enumeration for B)
    data_ATE             ATE from frequency counting on sample (None for not_identifiable)
"""

import json
import random
from itertools import combinations, product as itertools_product

import networkx as nx
import numpy as np
import pandas as pd
from datasets import Dataset
from networkx.algorithms.d_separation import find_minimal_d_separator, is_d_separator


# ─────────────────────────────────────────────────────────────────────────────
# Shared DAG utilities
# ─────────────────────────────────────────────────────────────────────────────


def _make_dag(n: int, edge_prob: float, rng: random.Random) -> nx.DiGraph:
    """Generate a random forward-only DAG (Erdős–Rényi, topological ordering).

    All n nodes (0..n-1) are added explicitly so isolated nodes are included.
    """
    nodes = list(range(n))
    edges = [(u, v) for u in nodes for v in nodes if u < v and rng.random() < edge_prob]
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    return G


def _make_backdoor_graph(G: nx.DiGraph, X: int) -> nx.DiGraph:
    """Return copy of G with all outgoing edges from X removed."""
    G_bd = G.copy()
    G_bd.remove_edges_from(list(G.out_edges(X)))
    return G_bd


def _add_latent_confounder(G: nx.DiGraph, X: int, Y: int, n: int) -> int:
    """Append latent node L=n to G with edges L→X and L→Y. Returns L."""
    G.add_node(n)
    G.add_edge(n, X)
    G.add_edge(n, Y)
    return n


# ─────────────────────────────────────────────────────────────────────────────
# Sub-case A — Linear SCM generation
# ─────────────────────────────────────────────────────────────────────────────


def _nonround_coeff(rng: random.Random) -> float:
    """Sample a non-round coefficient from Uniform([0.2,1.5] ∪ [-1.5,-0.2])."""
    mag = round(rng.uniform(0.2, 1.5), 2)
    # Avoid values very close to 0.5, 1.0 to prevent "round" numbers
    while round(mag * 2) / 2 == mag or round(mag) == mag:
        mag = round(rng.uniform(0.2, 1.5), 2)
    return mag if rng.random() < 0.5 else -mag


def _compute_ate_a(G: nx.DiGraph, X: int, Y: int, coeffs: dict) -> float:
    """Wright's rule: ATE = sum of (product of edge coefficients) over all directed X→Y paths."""
    ate = 0.0
    try:
        for path in nx.all_simple_paths(G, X, Y):
            contrib = 1.0
            for i in range(len(path) - 1):
                contrib *= coeffs[(path[i], path[i + 1])]
            ate += contrib
    except nx.NetworkXError:
        pass
    return round(ate, 6)


def _build_equations_text(
    G: nx.DiGraph,
    X: int,
    Y: int,
    coeffs: dict,
    sigmas: dict,
    X_prob: float,
    topo_order: list,
) -> str:
    """Render structural equations as readable text (shown to model)."""
    lines = []
    for nd in topo_order:
        parents = sorted(G.predecessors(nd))
        if nd == X:
            lines.append(f"  {nd}:  X ~ Bernoulli({X_prob})")
        elif not parents:
            lines.append(f"  {nd}:  {nd} ~ N(0, {sigmas[nd]})")
        else:
            terms = " + ".join(f"{coeffs[(pa, nd)]}·{pa}" for pa in parents)
            ylabel = "Y" if nd == Y else str(nd)
            lines.append(f"  {nd}:  {ylabel} = {terms} + N(0, {sigmas[nd]})")
    return "\n".join(lines)


def _problem_dict_a(
    G: nx.DiGraph,
    X: int,
    Y: int,
    coeffs: dict,
    sigmas: dict,
    X_prob: float,
    problem_type: str,
    ate: float,
) -> dict:
    """Build the standard Sub-case A problem dict."""
    topo = list(nx.topological_sort(G))
    return {
        "subcase": "A",
        "problem_type": problem_type,
        "edges": [(int(u), int(v)) for u, v in sorted(G.edges())],
        "nodes": [int(nd) for nd in sorted(G.nodes())],
        "X": int(X),
        "Y": int(Y),
        "observed_nodes": [int(nd) for nd in sorted(G.nodes())],
        "latent_nodes": [],
        "coefficients": {f"{u},{v}": c for (u, v), c in sorted(coeffs.items())},
        "sigmas": {str(nd): s for nd, s in sorted(sigmas.items())},
        "X_prob": X_prob,
        "structural_equations_text": _build_equations_text(
            G, X, Y, coeffs, sigmas, X_prob, topo
        ),
        "identifiability_status": "identifiable",
        "true_ATE": float(ate),
        "data_ATE": float(ate),   # same as true_ATE for Sub-case A
        "data_csv": None,
        "adjustment_set": None,
        "mediator_node": None,
    }


def _try_sample_a_no_path(rng: random.Random, min_nodes: int, max_nodes: int) -> dict | None:
    """Sample a Sub-case A 'no_path' problem (no directed X→Y path, ATE=0)."""
    n = rng.randint(min_nodes, max_nodes)
    G = _make_dag(n, 0.35, rng)

    roots = [nd for nd in G.nodes() if G.in_degree(nd) == 0]
    leaves = [nd for nd in G.nodes() if G.out_degree(nd) == 0]
    if not roots or not leaves:
        return None

    X = rng.choice(roots)
    non_reachable = [nd for nd in leaves if nd != X and not nx.has_path(G, X, nd)]
    if not non_reachable:
        return None
    Y = rng.choice(non_reachable)

    coeffs = {(u, v): _nonround_coeff(rng) for u, v in G.edges()}
    sigmas = {nd: round(rng.uniform(0.1, 0.5), 2) for nd in G.nodes() if nd != X}
    X_prob = round(rng.uniform(0.3, 0.7), 2)

    return _problem_dict_a(G, X, Y, coeffs, sigmas, X_prob, "no_path", 0.0)


def _try_sample_a_standard(rng: random.Random, min_nodes: int, max_nodes: int) -> dict | None:
    """Sample a Sub-case A 'standard' problem (1–2 paths, ATE ≠ 0)."""
    n = rng.randint(min_nodes, max_nodes)
    G = _make_dag(n, 0.4, rng)

    roots = [nd for nd in G.nodes() if G.in_degree(nd) == 0]
    if not roots:
        return None
    X = rng.choice(roots)

    leaves = [nd for nd in G.nodes() if G.out_degree(nd) == 0 and nd != X]
    reachable = [nd for nd in leaves if nx.has_path(G, X, nd)]
    if not reachable:
        return None
    Y = rng.choice(reachable)

    paths = list(nx.all_simple_paths(G, X, Y))
    if len(paths) == 0 or len(paths) > 2:
        return None

    coeffs = {(u, v): _nonround_coeff(rng) for u, v in G.edges()}
    ate = _compute_ate_a(G, X, Y, coeffs)

    # Standard: ATE ≠ 0, no cancellation
    if abs(ate) < 0.05:
        return None

    sigmas = {nd: round(rng.uniform(0.1, 0.5), 2) for nd in G.nodes() if nd != X}
    X_prob = round(rng.uniform(0.3, 0.7), 2)
    return _problem_dict_a(G, X, Y, coeffs, sigmas, X_prob, "standard", ate)


def _try_sample_a_mediated(rng: random.Random, min_nodes: int, max_nodes: int) -> dict | None:
    """Sample a Sub-case A 'mediated' problem (≥2 paths, no direct X→Y edge)."""
    n = rng.randint(max(5, min_nodes), max_nodes)
    G = _make_dag(n, 0.4, rng)

    roots = [nd for nd in G.nodes() if G.in_degree(nd) == 0]
    if not roots:
        return None
    X = rng.choice(roots)

    leaves = [nd for nd in G.nodes() if G.out_degree(nd) == 0 and nd != X]
    reachable = [nd for nd in leaves if nx.has_path(G, X, nd)]
    if not reachable:
        return None
    Y = rng.choice(reachable)

    # No direct X→Y edge
    if G.has_edge(X, Y):
        return None

    # ≥2 directed paths, all through mediators
    paths = list(nx.all_simple_paths(G, X, Y))
    if len(paths) < 2:
        return None

    coeffs = {(u, v): _nonround_coeff(rng) for u, v in G.edges()}
    ate = _compute_ate_a(G, X, Y, coeffs)

    if abs(ate) < 0.05:
        return None

    sigmas = {nd: round(rng.uniform(0.1, 0.5), 2) for nd in G.nodes() if nd != X}
    X_prob = round(rng.uniform(0.3, 0.7), 2)
    return _problem_dict_a(G, X, Y, coeffs, sigmas, X_prob, "mediated", ate)


def _try_sample_a_canceling(rng: random.Random, min_nodes: int, max_nodes: int) -> dict | None:
    """Sample a Sub-case A 'canceling' problem.

    Constructs a graph with exactly two X→Y paths via M1 and M2 with opposite-sign
    contributions that cancel to |ATE| ≤ 0.05, each |contrib| ≥ 0.4.
    Extra non-path nodes may be added for visual complexity.
    """
    n_extra = rng.randint(0, min(max_nodes - 4, 4))
    n_total = 4 + n_extra  # nodes: X=0, M1=1, M2=2, Y=3, extras=4..

    X, M1, M2, Y = 0, 1, 2, 3

    G = nx.DiGraph()
    G.add_nodes_from(range(n_total))
    G.add_edge(X, M1)
    G.add_edge(X, M2)
    G.add_edge(M1, Y)
    G.add_edge(M2, Y)

    # Add extra nodes as descendants of M1 or M2 (but NOT connecting to Y)
    for extra in range(4, n_total):
        parent = rng.choice([M1, M2, X] + list(range(4, extra)) if extra > 4 else [M1, M2])
        G.add_edge(parent, extra)

    # Generate canceling coefficients:
    # Path1: X→M1→Y with positive contribution c1 ≥ 0.4
    # Path2: X→M2→Y with negative contribution c2 ≈ -c1 (|c2| ≥ 0.4, |ATE| ≤ 0.05)
    for _ in range(50):  # up to 50 attempts to get valid canceling coefficients
        c_XM1 = round(rng.uniform(0.65, 1.2), 2)
        c_M1Y = round(rng.uniform(0.65, 1.2), 2)
        contrib1 = c_XM1 * c_M1Y

        if abs(contrib1) < 0.4:
            continue

        c_XM2 = round(rng.uniform(0.65, 1.2), 2)
        # Target: c_XM2 * c_M2Y ≈ -contrib1
        c_M2Y_target = -contrib1 / c_XM2
        c_M2Y = round(c_M2Y_target + rng.uniform(-0.03, 0.03), 2)

        # Must be in [-1.5, -0.2]
        if c_M2Y > -0.2 or c_M2Y < -1.5:
            continue

        contrib2 = c_XM2 * c_M2Y
        if abs(contrib2) < 0.4:
            continue

        coeffs = {
            (X, M1): c_XM1,
            (M1, Y): c_M1Y,
            (X, M2): c_XM2,
            (M2, Y): c_M2Y,
        }
        # Random coefficients for extra edges
        for u, v in G.edges():
            if (u, v) not in coeffs:
                coeffs[(u, v)] = _nonround_coeff(rng)

        ate = _compute_ate_a(G, X, Y, coeffs)
        if abs(ate) > 0.05:
            continue

        sigmas = {nd: round(rng.uniform(0.1, 0.5), 2) for nd in G.nodes() if nd != X}
        X_prob = round(rng.uniform(0.3, 0.7), 2)
        return _problem_dict_a(G, X, Y, coeffs, sigmas, X_prob, "canceling", ate)

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Sub-case B — Discrete CPT SCM generation
# ─────────────────────────────────────────────────────────────────────────────


def _gen_cpt(rng: random.Random, n_cat: int, parent_cats: list[int]) -> dict:
    """Generate a CPT for a node.

    Args:
        n_cat: Number of categories for this node (2=binary, 3=ternary).
        parent_cats: Number of categories for each parent node.

    Returns:
        dict mapping tuple(parent_values) → P(V=1|...) for binary, or
        list [P(V=0), P(V=1), P(V=2)] for ternary.
    """
    combos = list(itertools_product(*[range(k) for k in parent_cats])) if parent_cats else [()]

    cpt = {}
    for combo in combos:
        if n_cat == 2:
            cpt[combo] = round(rng.uniform(0.1, 0.9), 3)
        else:
            # Ternary: ensure all probs ≥ 0.1
            for _ in range(100):
                raw = [rng.uniform(0.15, 0.7) for _ in range(3)]
                s = sum(raw)
                probs = [round(r / s, 3) for r in raw]
                probs[2] = round(1.0 - probs[0] - probs[1], 3)
                if all(p >= 0.1 for p in probs):
                    break
            cpt[combo] = probs
    return cpt


def _sample_row(topo_order: list, parents_map: dict, cpts: dict, n_cats: dict, rng: random.Random) -> dict:
    """Sample one data row in topological order. Returns values for ALL nodes."""
    row = {}
    for nd in topo_order:
        pvals = tuple(row[pa] for pa in parents_map[nd])
        cpt_entry = cpts[nd][pvals]
        n = n_cats[nd]
        if n == 2:
            row[nd] = 1 if rng.random() < cpt_entry else 0
        else:
            r = rng.random()
            cum = 0.0
            chosen = n - 1
            for i, p in enumerate(cpt_entry):
                cum += p
                if r < cum:
                    chosen = i
                    break
            row[nd] = chosen
    return row


def _compute_data_ate_backdoor(df: pd.DataFrame, X: int, Y: int, adjustment_set: list) -> float | None:
    """Compute data_ATE using the backdoor adjustment formula."""
    if not adjustment_set:
        n_x1 = (df[X] == 1).sum()
        n_x0 = (df[X] == 0).sum()
        if n_x1 == 0 or n_x0 == 0:
            return None
        p_y1_x1 = ((df[X] == 1) & (df[Y] == 1)).sum() / n_x1
        p_y1_x0 = ((df[X] == 0) & (df[Y] == 1)).sum() / n_x0
        return round(float(p_y1_x1 - p_y1_x0), 6)

    Z = adjustment_set
    # Get unique strata
    strata = df[Z].drop_duplicates()
    ate = 0.0
    n_total = len(df)

    for _, z_row in strata.iterrows():
        z_dict = dict(z_row)
        mask_z = pd.Series(True, index=df.index)
        for col, val in z_dict.items():
            mask_z &= df[col] == val

        mask_x1_z = mask_z & (df[X] == 1)
        mask_x0_z = mask_z & (df[X] == 0)

        n_z = int(mask_z.sum())
        n_x1_z = int(mask_x1_z.sum())
        n_x0_z = int(mask_x0_z.sum())

        if n_x1_z == 0 or n_x0_z == 0:
            return None  # Missing support

        p_y1_x1_z = int((mask_x1_z & (df[Y] == 1)).sum()) / n_x1_z
        p_y1_x0_z = int((mask_x0_z & (df[Y] == 1)).sum()) / n_x0_z
        p_z = n_z / n_total

        ate += (p_y1_x1_z - p_y1_x0_z) * p_z

    return round(float(ate), 6)



def _compute_data_ate_frontdoor(df: pd.DataFrame, X: int, Y: int, M: int) -> float | None:
    """Compute data_ATE using the two-step frontdoor formula."""
    m_values = sorted(df[M].unique())
    x_values = [0, 1]
    n_total = len(df)

    p_x = {}
    for x in x_values:
        n_x = (df[X] == x).sum()
        if n_x == 0:
            return None
        p_x[x] = n_x / n_total

    ate = 0.0
    for x_treat, sign in [(1, 1.0), (0, -1.0)]:
        n_xtreat = (df[X] == x_treat).sum()
        if n_xtreat == 0:
            return None

        inner = 0.0
        for m in m_values:
            mask_xm = (df[X] == x_treat) & (df[M] == m)
            n_xm = int(mask_xm.sum())
            p_m_given_x = n_xm / n_xtreat

            # Σ_x' P(Y=1|X=x', M=m) · P(X=x')
            y_sum = 0.0
            for x_prime in x_values:
                mask_xp_m = (df[X] == x_prime) & (df[M] == m)
                n_xp_m = int(mask_xp_m.sum())
                if n_xp_m > 0:
                    p_y1_xp_m = int((mask_xp_m & (df[Y] == 1)).sum()) / n_xp_m
                else:
                    p_y1_xp_m = 0.0
                y_sum += p_y1_xp_m * p_x[x_prime]

            inner += p_m_given_x * y_sum

        ate += sign * inner

    return round(float(ate), 6)


def _sample_data_b(
    topo_order: list,
    parents_map: dict,
    cpts: dict,
    n_cats: dict,
    observed_nodes: set,
    latent_nodes: set,
    required_cells: list[tuple],  # list of (masks) to check min per cell
    min_per_cell: int,
    max_n: int,
    rng: random.Random,
    np_rng: np.random.Generator,
) -> pd.DataFrame | None:
    """Sample data until all required cells have min_per_cell observations.

    required_cells: list of (X_val, Z_vals_tuple, Z_nodes) for backdoor,
    or handled by caller checking cell counts directly.

    Returns DataFrame with observed columns only, or None if max_n exceeded.
    """
    rows = []
    while True:
        # Sample a batch
        batch_size = 1000
        for _ in range(batch_size):
            row = _sample_row(topo_order, parents_map, cpts, n_cats, rng)
            rows.append(row)

        # Build dataframe of ALL nodes (including latent) for cell checking
        df_all = pd.DataFrame(rows)

        # Check if all required cells are populated
        all_ok = True
        for cell_mask_fn in required_cells:
            if int(cell_mask_fn(df_all).sum()) < min_per_cell:
                all_ok = False
                break

        if all_ok:
            break

        if len(rows) >= max_n:
            return None

    # Return only observed columns
    obs_cols = sorted(observed_nodes)
    df_obs = df_all[obs_cols].copy()
    df_obs.columns = [str(c) for c in df_obs.columns]
    return df_obs


def _try_sample_b_backdoor(
    rng: random.Random,
    min_nodes: int,
    max_nodes: int,
    edge_prob: float,
    latent_prob: float,
    empty: bool = False,
) -> dict | None:
    """Sample a backdoor_standard or backdoor_empty Sub-case B problem."""
    n = rng.randint(min_nodes, max_nodes)
    G = _make_dag(n, edge_prob, rng)
    nodes_list = sorted(G.nodes())
    if len(nodes_list) < 3:
        return None

    X, Y = rng.sample(nodes_list, 2)
    if not nx.has_path(G, X, Y):
        return None
    if G.out_degree(Y) > 0:
        return None
    if nx.has_path(G, Y, X):
        return None

    # Assign latent nodes (X, Y always observed)
    other_nodes = [nd for nd in nodes_list if nd != X and nd != Y]
    latent_nodes = {nd for nd in other_nodes if rng.random() < latent_prob}
    observed_nodes = set(nodes_list) - latent_nodes

    G_bd = _make_backdoor_graph(G, X)

    try:
        min_set = find_minimal_d_separator(G_bd, X, Y, restricted=observed_nodes - {X, Y})
    except Exception:
        return None

    if min_set is None:
        return None

    if empty and len(min_set) != 0:
        return None
    if not empty and len(min_set) == 0:
        return None

    # Verify no descendants of X in min_set
    if min_set & nx.descendants(G, X):
        return None

    # Complexity filter for standard: ≥1 backdoor path
    G_bd_undir = G_bd.to_undirected()
    try:
        bd_paths = list(nx.all_simple_paths(G_bd_undir, X, Y))
    except Exception:
        bd_paths = []

    if not bd_paths:
        return None

    # Cap adjustment set size to ≤3 for tractability
    if len(min_set) > 3:
        return None

    adjustment_set = sorted(min_set)

    # Assign variable types
    n_cats = {}
    for nd in nodes_list:
        if nd == X or nd == Y:
            n_cats[nd] = 2  # X and Y always binary
        elif nd in latent_nodes:
            n_cats[nd] = 2  # latent nodes can be binary
        else:
            n_cats[nd] = 2 if rng.random() < 0.6 else 3

    # Generate CPTs
    topo_order = list(nx.topological_sort(G))
    parents_map = {nd: sorted(G.predecessors(nd)) for nd in G.nodes()}
    cpts = {}
    for nd in topo_order:
        pa = parents_map[nd]
        pa_cats = [n_cats[p] for p in pa]
        cpts[nd] = _gen_cpt(rng, n_cats[nd], pa_cats)

    # Build required cell masks for data sampling
    def _make_cell_masks_backdoor(adj_set):
        masks = []
        for x_val in [0, 1]:
            if not adj_set:
                masks.append(lambda df, xv=x_val: (df[X] == xv))
            else:
                # For each unique combination of adjustment set values
                adj_cats = [n_cats[nd] for nd in adj_set]
                for z_combo in itertools_product(*[range(k) for k in adj_cats]):
                    def _mask(df, xv=x_val, zc=z_combo, zs=adj_set):
                        m = df[X] == xv
                        for znode, zval in zip(zs, zc):
                            m = m & (df[znode] == zval)
                        return m
                    masks.append(_mask)
        return masks

    required_cells = _make_cell_masks_backdoor(adjustment_set)
    MIN_PER_CELL = 50
    MAX_N = 100_000

    df_obs = _sample_data_b(
        topo_order, parents_map, cpts, n_cats,
        observed_nodes, latent_nodes,
        required_cells, MIN_PER_CELL, MAX_N, rng, None,
    )
    if df_obs is None:
        return None

    # Compute data_ATE using frequency counting
    # Rename columns to string for pandas access
    df_int = df_obs.rename(columns=int)
    data_ate = _compute_data_ate_backdoor(df_int, X, Y, adjustment_set)
    if data_ate is None:
        return None

    # Compute true_ATE via large simulation
    true_ate = _simulate_true_ate(topo_order, parents_map, cpts, n_cats, X, Y, rng)

    ptype = "backdoor_empty" if empty else "backdoor_standard"

    return {
        "subcase": "B",
        "problem_type": ptype,
        "edges": [(int(u), int(v)) for u, v in sorted(G.edges())],
        "nodes": [int(nd) for nd in sorted(G.nodes())],
        "X": int(X),
        "Y": int(Y),
        "observed_nodes": sorted(int(nd) for nd in observed_nodes),
        "latent_nodes": sorted(int(nd) for nd in latent_nodes),
        "identifiability_status": "identifiable",
        "adjustment_set": [int(nd) for nd in adjustment_set],
        "mediator_node": None,
        "true_ATE": float(true_ate),
        "data_ATE": float(data_ate),
        "data_csv": df_obs.to_csv(index=False),
        "structural_equations_text": None,
        "coefficients": None,
        "sigmas": None,
        "X_prob": None,
    }


def _simulate_true_ate(
    topo_order: list,
    parents_map: dict,
    cpts: dict,
    n_cats: dict,
    X: int,
    Y: int,
    rng: random.Random,
    n_samples: int = 50_000,
) -> float:
    """Estimate true ATE via simulation under do(X=1) and do(X=0)."""
    y_means = []
    for x_val in [1, 0]:
        y_sum = 0.0
        for _ in range(n_samples):
            row = {}
            for nd in topo_order:
                if nd == X:
                    row[nd] = x_val  # intervention: do(X=x_val)
                    continue
                pvals = tuple(row[pa] for pa in parents_map[nd])
                cpt_entry = cpts[nd][pvals]
                n = n_cats[nd]
                if n == 2:
                    row[nd] = 1 if rng.random() < cpt_entry else 0
                else:
                    r = rng.random()
                    cum = 0.0
                    chosen = n - 1
                    for i, p in enumerate(cpt_entry):
                        cum += p
                        if r < cum:
                            chosen = i
                            break
                    row[nd] = chosen
            y_sum += row[Y]
        y_means.append(y_sum / n_samples)
    return round(y_means[0] - y_means[1], 6)


def _try_sample_b_frontdoor(
    rng: random.Random,
    min_nodes: int,
    max_nodes: int,
    edge_prob: float,
) -> dict | None:
    """Sample a frontdoor Sub-case B problem."""
    n = rng.randint(min_nodes, max_nodes)
    G = _make_dag(n, edge_prob, rng)

    # Find triples (X, M, Y) where X→M direct, M→Y direct, Y is leaf, no X→Y edge
    candidates = [
        (u, m, v)
        for u, m in G.edges()
        for v in G.successors(m)
        if G.out_degree(v) == 0 and u != v and not G.has_edge(u, v)
    ]
    if not candidates:
        return None

    X, M, Y = rng.choice(candidates)

    # Frontdoor condition 2: M intercepts ALL directed X→Y paths
    G_no_M = G.copy()
    G_no_M.remove_node(M)
    if nx.has_path(G_no_M, X, Y):
        return None

    # Add latent confounder L→X, L→Y (not L→M — preserves condition 1)
    L = _add_latent_confounder(G, X, Y, n)

    nodes_list = sorted(G.nodes())
    observed_nodes = set(range(n))
    latent_nodes = {L}

    # Assign variable types
    n_cats = {}
    for nd in nodes_list:
        if nd == X or nd == Y:
            n_cats[nd] = 2
        elif nd == L:
            n_cats[nd] = 2
        else:
            n_cats[nd] = 2 if rng.random() < 0.6 else 3

    # Generate CPTs
    topo_order = list(nx.topological_sort(G))
    parents_map = {nd: sorted(G.predecessors(nd)) for nd in G.nodes()}
    cpts = {}
    for nd in topo_order:
        pa = parents_map[nd]
        cpts[nd] = _gen_cpt(rng, n_cats[nd], [n_cats[p] for p in pa])

    # Required cells for frontdoor: (X=x, M=m) for all x ∈ {0,1}, m ∈ M values
    # Determine M's categories
    m_cats = n_cats[M]
    required_cells = []
    for x_val in [0, 1]:
        for m_val in range(m_cats):
            def _mask(df, xv=x_val, mv=m_val):
                return (df[X] == xv) & (df[M] == mv)
            required_cells.append(_mask)
    # Also need (X=x', M=m) for grading
    for x_prime in [0, 1]:
        for m_val in range(m_cats):
            def _mask2(df, xp=x_prime, mv=m_val):
                return (df[X] == xp) & (df[M] == mv)
            if _mask2 not in required_cells:
                required_cells.append(_mask2)

    MIN_PER_CELL = 50
    MAX_N = 100_000

    df_obs = _sample_data_b(
        topo_order, parents_map, cpts, n_cats,
        observed_nodes, latent_nodes,
        required_cells, MIN_PER_CELL, MAX_N, rng, None,
    )
    if df_obs is None:
        return None

    df_int = df_obs.rename(columns=int)
    data_ate = _compute_data_ate_frontdoor(df_int, X, Y, M)
    if data_ate is None:
        return None

    true_ate = _simulate_true_ate(topo_order, parents_map, cpts, n_cats, X, Y, rng)

    return {
        "subcase": "B",
        "problem_type": "frontdoor",
        "edges": [(int(u), int(v)) for u, v in sorted(G.edges())],
        "nodes": [int(nd) for nd in sorted(G.nodes())],
        "X": int(X),
        "Y": int(Y),
        "observed_nodes": sorted(int(nd) for nd in observed_nodes),
        "latent_nodes": sorted(int(nd) for nd in latent_nodes),
        "identifiability_status": "identifiable",
        "adjustment_set": None,
        "mediator_node": int(M),
        "true_ATE": float(true_ate),
        "data_ATE": float(data_ate),
        "data_csv": df_obs.to_csv(index=False),
        "structural_equations_text": None,
        "coefficients": None,
        "sigmas": None,
        "X_prob": None,
    }


def _try_sample_b_not_identifiable(
    rng: random.Random,
    min_nodes: int,
    max_nodes: int,
    edge_prob: float,
) -> dict | None:
    """Sample a not_identifiable Sub-case B problem.

    Constructs X→Y (direct) with latent L→X, L→Y.
    Verifies find_minimal_d_separator returns None on observed nodes.
    """
    n = rng.randint(min_nodes, max_nodes)
    G = _make_dag(n, edge_prob, rng)

    # Need a direct X→Y edge (rules out frontdoor)
    candidates = [(u, v) for u, v in G.edges() if G.out_degree(v) == 0]
    if not candidates:
        return None

    X, Y = rng.choice(candidates)

    L = _add_latent_confounder(G, X, Y, n)
    nodes_list = sorted(G.nodes())
    observed_nodes = set(range(n))
    latent_nodes = {L}

    G_bd = _make_backdoor_graph(G, X)
    try:
        check = find_minimal_d_separator(G_bd, X, Y, restricted=observed_nodes - {X, Y})
    except Exception:
        return None

    if check is not None:
        return None

    # Generate some data to show in the prompt (model must declare not_identifiable)
    n_cats = {nd: 2 for nd in nodes_list}
    topo_order = list(nx.topological_sort(G))
    parents_map = {nd: sorted(G.predecessors(nd)) for nd in G.nodes()}
    cpts = {}
    for nd in topo_order:
        pa = parents_map[nd]
        cpts[nd] = _gen_cpt(rng, n_cats[nd], [n_cats[p] for p in pa])

    # Sample a modest amount of data (no required_cells constraint)
    rows = [_sample_row(topo_order, parents_map, cpts, n_cats, rng) for _ in range(2000)]
    df_all = pd.DataFrame(rows)
    obs_cols = sorted(observed_nodes)
    df_obs = df_all[obs_cols].rename(columns=str)

    return {
        "subcase": "B",
        "problem_type": "not_identifiable",
        "edges": [(int(u), int(v)) for u, v in sorted(G.edges())],
        "nodes": [int(nd) for nd in sorted(G.nodes())],
        "X": int(X),
        "Y": int(Y),
        "observed_nodes": sorted(int(nd) for nd in observed_nodes),
        "latent_nodes": sorted(int(nd) for nd in latent_nodes),
        "identifiability_status": "not_identifiable",
        "adjustment_set": None,
        "mediator_node": None,
        "true_ATE": None,
        "data_ATE": None,
        "data_csv": df_obs.to_csv(index=False),
        "structural_equations_text": None,
        "coefficients": None,
        "sigmas": None,
        "X_prob": None,
    }



# ─────────────────────────────────────────────────────────────────────────────
# Main generator
# ─────────────────────────────────────────────────────────────────────────────


def generate_flavor2_problems(
    n_train: int = 250,
    n_eval: int = 100,
    seed: int = 42,
    min_nodes_a: int = 4,
    max_nodes_a: int = 8,
    min_nodes_b: int = 5,
    max_nodes_b: int = 9,
    edge_prob_b: float = 0.38,
    latent_prob_b: float = 0.10,
    # Sub-case A type fractions (sum to 1.0 within Sub-case A)
    a_standard_frac: float = 0.40,
    a_mediated_frac: float = 0.30,
    a_canceling_frac: float = 0.20,
    # a_no_path_frac: remainder (~0.10)
    # Sub-case B type fractions (sum to 1.0 within Sub-case B)
    b_backdoor_std_frac: float = 0.35,
    b_backdoor_empty_frac: float = 0.20,
    b_frontdoor_frac: float = 0.20,
    # b_not_identifiable_frac: remainder (~0.25)
    # Overall Sub-case A vs B split
    subcase_a_frac: float = 0.20,
) -> tuple[list[dict], list[dict]]:
    """Generate stratified train and eval problem pools for Flavor 2.

    Returns:
        (train_problems, eval_problems): two lists of problem dicts.
    """
    rng = random.Random(seed)
    n_total = n_train + n_eval

    # Overall sub-case split
    n_a = round(subcase_a_frac * n_total)
    n_b = n_total - n_a

    # Sub-case A buckets
    n_a_standard = round(a_standard_frac * n_a)
    n_a_mediated = round(a_mediated_frac * n_a)
    n_a_canceling = round(a_canceling_frac * n_a)
    n_a_no_path = n_a - n_a_standard - n_a_mediated - n_a_canceling

    # Sub-case B buckets
    n_b_bd_std = round(b_backdoor_std_frac * n_b)
    n_b_bd_empty = round(b_backdoor_empty_frac * n_b)
    n_b_fd = round(b_frontdoor_frac * n_b)
    n_b_not_id = n_b - n_b_bd_std - n_b_bd_empty - n_b_fd

    targets_a = {
        "standard": n_a_standard,
        "mediated": n_a_mediated,
        "canceling": n_a_canceling,
        "no_path": max(0, n_a_no_path),
    }
    targets_b = {
        "backdoor_standard": n_b_bd_std,
        "backdoor_empty": n_b_bd_empty,
        "frontdoor": n_b_fd,
        "not_identifiable": max(0, n_b_not_id),
    }

    buckets: dict[str, list[dict]] = {k: [] for k in {**targets_a, **targets_b}}

    _sampler_a = {
        "standard": lambda: _try_sample_a_standard(rng, min_nodes_a, max_nodes_a),
        "mediated": lambda: _try_sample_a_mediated(rng, min_nodes_a, max_nodes_a),
        "canceling": lambda: _try_sample_a_canceling(rng, min_nodes_a, max_nodes_a),
        "no_path": lambda: _try_sample_a_no_path(rng, min_nodes_a, max_nodes_a),
    }
    _sampler_b = {
        "backdoor_standard": lambda: _try_sample_b_backdoor(
            rng, min_nodes_b, max_nodes_b, edge_prob_b, latent_prob_b, empty=False
        ),
        "backdoor_empty": lambda: _try_sample_b_backdoor(
            rng, min_nodes_b, max_nodes_b, edge_prob_b, latent_prob_b, empty=True
        ),
        "frontdoor": lambda: _try_sample_b_frontdoor(
            rng, min_nodes_b, max_nodes_b, edge_prob_b
        ),
        "not_identifiable": lambda: _try_sample_b_not_identifiable(
            rng, min_nodes_b, max_nodes_b, edge_prob_b
        ),
    }

    all_targets = {**targets_a, **targets_b}
    all_samplers = {**_sampler_a, **_sampler_b}

    # Collect problems for each bucket
    for ptype, target in all_targets.items():
        sampler = all_samplers[ptype]
        attempts = 0
        while len(buckets[ptype]) < target:
            attempts += 1
            if attempts > target * 500:
                # Soft failure: log and continue with fewer samples
                break
            prob = sampler()
            if prob is not None:
                buckets[ptype].append(prob)

    # Stratified train/eval split
    train_frac = n_train / n_total if n_total > 0 else 1.0
    train_problems: list[dict] = []
    eval_problems: list[dict] = []

    for ptype in all_targets:
        probs = buckets[ptype]
        rng.shuffle(probs)
        n_to_train = round(len(probs) * train_frac)
        train_problems.extend(probs[:n_to_train])
        eval_problems.extend(probs[n_to_train:])

    rng.shuffle(train_problems)
    rng.shuffle(eval_problems)

    return train_problems, eval_problems


# ─────────────────────────────────────────────────────────────────────────────
# Dataset builder
# ─────────────────────────────────────────────────────────────────────────────


def build_dataset(problems: list[dict], format_fn) -> Dataset:
    """Convert a list of problem dicts into a HuggingFace Dataset.

    Args:
        problems: List of problem dicts from generate_flavor2_problems.
        format_fn: Callable(problem_dict) -> str — renders the problem text.
    """
    rows = []
    for p in problems:
        rows.append({
            "question": format_fn(p),
            "info": json.dumps({
                "subcase": p["subcase"],
                "problem_type": p["problem_type"],
                "X": p["X"],
                "Y": p["Y"],
                "edges": p["edges"],
                "nodes": p["nodes"],
                "observed_nodes": p["observed_nodes"],
                "latent_nodes": p["latent_nodes"],
                "identifiability_status": p["identifiability_status"],
                "true_ATE": p.get("true_ATE"),
                "data_ATE": p.get("data_ATE"),
                "adjustment_set": p.get("adjustment_set"),
                "mediator_node": p.get("mediator_node"),
                "data_csv": p.get("data_csv"),
            }),
        })
    return Dataset.from_list(rows)
