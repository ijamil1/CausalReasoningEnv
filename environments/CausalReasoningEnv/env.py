"""CausalATEEnv — two-phase StatefulToolEnv for ATE estimation.

Phase 1 (declaration turn): Model reasons about the DAG and writes a <set> tag
declaring its identification set. Scored independently of computation.

Phase 2 (tool use + answer): Model calls tools and writes <answer> to end the episode.
A global MAX_TOOL_CALLS cap enforces minimal tool use.

Reward components (weights: 0.05 / 0.30 / 0.15 / 0.50):
  format_compliance / set_valid / minimality / ate_accuracy
"""

import json
import re
from itertools import product as itertools_product

import networkx as nx
import verifiers as vf

from prompts import SYSTEM_PROMPT

MAX_TOOL_CALLS = 5  # global cap; optimal max is 2
MAX_TURNS = 7       # 1 declaration + up to 5 tool calls + 1 answer
ATE_THRESHOLD = 0.01


# ─────────────────────────────────────────────────────────────────────────────
# Exact probability inference (shared by both tools)
# ─────────────────────────────────────────────────────────────────────────────


def _joint_marginal(
    int_assignments: dict,
    cpts: dict,
    domains: dict,
    topo_order: list,
    parents_map: dict,
) -> float:
    """Compute P(assignments) by summing over all consistent full configurations."""
    prob_sum = 0.0
    for vals in itertools_product(*[range(len(domains[nd])) for nd in topo_order]):
        config = dict(zip(topo_order, vals))
        if any(config[k] != v for k, v in int_assignments.items()):
            continue
        joint = 1.0
        for nd in topo_order:
            pa = parents_map[nd]
            pa_vals = tuple(config[p] for p in pa)
            cpt_entry = cpts[nd][pa_vals]
            v = config[nd]
            if len(domains[nd]) == 2:
                joint *= cpt_entry if v == 1 else (1.0 - cpt_entry)
            else:
                joint *= cpt_entry[v]
        prob_sum += joint
    return prob_sum


# ─────────────────────────────────────────────────────────────────────────────
# Tool functions (hidden args injected by update_tool_args)
# ─────────────────────────────────────────────────────────────────────────────


async def marginal(
    variables: list,
    _cpts: dict,
    _domains: dict,
    _topo_order: list,
    _parents_map: dict,
    _latent_nodes: list,
) -> str:
    """Return the full joint PMF P(V1, V2, ...) for the given observed variables.

    Args:
        variables: List of node IDs (as strings) to compute the joint marginal for.
                   Example: ["2", "3"] returns P(node2, node3) for all value combos.
    """
    int_vars = [int(v) for v in variables]
    for v in int_vars:
        if v in _latent_nodes:
            return f"Error: node {v} is latent and not observable."
    lines = []
    for vals in itertools_product(*[range(len(_domains[v])) for v in int_vars]):
        int_assignments = dict(zip(int_vars, vals))
        prob = _joint_marginal(int_assignments, _cpts, _domains, _topo_order, _parents_map)
        lhs = ", ".join(f"node{v}={val}" for v, val in zip(int_vars, vals))
        lines.append(f"P({lhs}) = {round(prob, 6)}")
    return "\n".join(lines)


async def conditional(
    query: list,
    given: list,
    _cpts: dict,
    _domains: dict,
    _topo_order: list,
    _parents_map: dict,
    _latent_nodes: list,
) -> str:
    """Return the full conditional PMF P(query | given) for all strata of given variables.

    Args:
        query: List of node IDs (strings) for the query variables.
        given: List of node IDs (strings) for the conditioning variables.
               Returns P(query=q | given=g) for every combination of q and g values.
    """
    int_query = [int(v) for v in query]
    int_given = [int(v) for v in given]
    for v in int_query + int_given:
        if v in _latent_nodes:
            return f"Error: node {v} is latent and not observable."
    lines = []
    for given_vals in itertools_product(*[range(len(_domains[v])) for v in int_given]):
        given_assignments = dict(zip(int_given, given_vals))
        denom = _joint_marginal(given_assignments, _cpts, _domains, _topo_order, _parents_map)
        if denom < 1e-10:
            given_str = ", ".join(f"node{v}={val}" for v, val in zip(int_given, given_vals))
            lines.append(f"P(... | {given_str}) = undefined (zero probability)")
            continue
        for query_vals in itertools_product(*[range(len(_domains[v])) for v in int_query]):
            query_assignments = dict(zip(int_query, query_vals))
            numer = _joint_marginal(
                {**query_assignments, **given_assignments},
                _cpts, _domains, _topo_order, _parents_map,
            )
            query_str = ", ".join(f"node{v}={val}" for v, val in zip(int_query, query_vals))
            given_str = ", ".join(f"node{v}={val}" for v, val in zip(int_given, given_vals))
            lines.append(f"P({query_str} | {given_str}) = {round(numer / denom, 6)}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────


def _parse_answer(messages: list) -> str | None:
    """Extract the content of the last <answer> block from the last assistant message."""
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            matches = re.findall(r"<answer>\s*(.*?)\s*</answer>", content, re.DOTALL)
            return matches[-1].strip() if matches else None
    return None


def _parse_set(messages: list) -> list[int] | None:
    """Extract declared identification set from first assistant message with <set> tag.

    Returns:
        list[int]  — declared node IDs (may be empty for empty identification set)
        None       — no <set> tag found
    """
    for msg in messages:
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            set_m = re.search(r"<set>\s*(.*?)\s*</set>", content, re.DOTALL)
            if set_m:
                inner = set_m.group(1).strip()
                if not inner or inner == "{}":
                    return []  # empty identification set
                return [
                    int(x.strip()) for x in inner.split(",")
                    if x.strip().lstrip("-").isdigit()
                ]
    return None  # no <set> tag found


def _reconstruct_graph(info: dict) -> nx.DiGraph:
    """Rebuild NetworkX DiGraph from stored edges in info dict."""
    G = nx.DiGraph()
    G.add_nodes_from(info["nodes"])
    G.add_edges_from([tuple(e) for e in info["edges"]])
    return G


def _check_frontdoor_conditions(G: nx.DiGraph, X: int, Y: int, M: set) -> bool:
    """Check all three frontdoor conditions for mediator set M."""
    M = set(M)
    if not M:
        return False

    # Condition 1: Y not reachable from X without going through M
    G_minus_M = G.subgraph(set(G.nodes()) - M)
    try:
        if nx.has_path(G_minus_M, X, Y):
            return False
    except nx.NetworkXError:
        return False

    # Condition 2: X d-separated from M in backdoor graph given ∅
    G_xbar = G.copy()
    G_xbar.remove_edges_from(list(G.out_edges(X)))
    try:
        if not nx.d_separated(G_xbar, {X}, M, set()):
            return False
    except Exception:
        return False

    # Condition 3: M d-separated from Y by X in graph with M's outgoing edges removed
    G_mbar = G.copy()
    for m in M:
        G_mbar.remove_edges_from(list(G.out_edges(m)))
    try:
        if not nx.d_separated(G_mbar, M, {Y}, {X}):
            return False
    except Exception:
        return False

    return True


# ─────────────────────────────────────────────────────────────────────────────
# Reward functions
# ─────────────────────────────────────────────────────────────────────────────


async def format_compliance(completion, **kwargs) -> float:
    """1.0 if a valid <answer> block is present."""
    answer = _parse_answer(completion)
    if answer is None:
        return 0.0
    if answer == "not_identifiable":
        return 1.0
    if re.match(r"^ATE=[-+]?\d+(\.\d+)?$", answer):
        return 1.0
    return 0.0


async def set_valid(completion, info, **kwargs) -> float:
    """1.0 if declared set satisfies the identification criterion for this problem type."""
    if isinstance(info, str):
        info = json.loads(info)
    problem_type = info.get("problem_type", "")

    if problem_type == "not_identifiable":
        return 1.0  # no set needed; answer block checked by ate_accuracy

    declared_set = _parse_set(completion)
    if declared_set is None:
        return 0.0  # identifiable problem but no <set> tag declared

    latent_nodes = set(info.get("latent_nodes", []))
    if any(n in latent_nodes for n in declared_set):
        return 0.0

    X, Y = info["X"], info["Y"]
    G = _reconstruct_graph(info)

    if problem_type in ("backdoor_empty", "backdoor_standard"):
        Z = set(declared_set)
        if Z & nx.descendants(G, X):  # Z contains a descendant of X
            return 0.0
        G_back = G.copy()
        G_back.remove_edges_from(list(G.out_edges(X)))
        try:
            return 1.0 if nx.d_separated(G_back, {X}, {Y}, Z) else 0.0
        except Exception:
            return 0.0

    if problem_type == "frontdoor":
        M = set(declared_set)
        if not M:
            return 0.0
        return 1.0 if _check_frontdoor_conditions(G, X, Y, M) else 0.0

    return 0.0


async def minimality(completion, info, **kwargs) -> float:
    """Graded: 1.0 if declared set is minimal size, k/|declared| if valid superset."""
    if isinstance(info, str):
        info = json.loads(info)
    problem_type = info.get("problem_type", "")
    if problem_type == "not_identifiable":
        return 1.0  # full credit; no set to minimize
    # Gate on validity
    if await set_valid(completion, info, **kwargs) == 0.0:
        return 0.0
    declared_set = _parse_set(completion) or []
    minimal_set = info.get("minimal_set")
    if minimal_set is None:
        return 1.0
    k = len(minimal_set)
    declared_size = len(declared_set)
    if declared_size == 0:
        return 1.0 if k == 0 else 0.0
    return min(1.0, k / declared_size)


async def ate_accuracy(completion, info, **kwargs) -> float:
    """1.0 if final answer matches true_ATE within tolerance or correctly states not_identifiable."""
    if isinstance(info, str):
        info = json.loads(info)
    answer = _parse_answer(completion)
    if answer is None:
        return 0.0
    if info.get("identifiability_status") == "not_identifiable":
        return 1.0 if answer == "not_identifiable" else 0.0
    if answer == "not_identifiable":
        return 0.0
    m = re.match(r"^ATE=([-+]?\d+(?:\.\d+)?)$", answer)
    if not m:
        return 0.0
    try:
        ate_hat = float(m.group(1))
    except ValueError:
        return 0.0
    true_ate = info.get("true_ATE")
    if true_ate is None:
        return 0.0
    return 1.0 if abs(ate_hat - true_ate) <= ATE_THRESHOLD else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Environment class
# ─────────────────────────────────────────────────────────────────────────────


class CausalATEEnv(vf.StatefulToolEnv):
    """Two-phase environment for ATE estimation via probability query tools."""

    def __init__(self, dataset: vf.Dataset, **kwargs):
        rubric = vf.Rubric(
            funcs=[format_compliance, set_valid, minimality, ate_accuracy],
            weights=[0.05, 0.30, 0.15, 0.50],
        )
        super().__init__(
            dataset=dataset,
            system_prompt=SYSTEM_PROMPT,
            tools=[],
            rubric=rubric,
            max_turns=MAX_TURNS,
            **kwargs,
        )
        self.add_tool(marginal, args_to_skip=["_cpts", "_domains", "_topo_order", "_parents_map", "_latent_nodes"])
        self.add_tool(conditional, args_to_skip=["_cpts", "_domains", "_topo_order", "_parents_map", "_latent_nodes"])

    async def setup_state(self, state: vf.State) -> vf.State:
        """Deserialize CPTs and domain info from the problem info dict."""
        info = state.get("info") or {}
        if isinstance(info, str):
            info = json.loads(info)

        # Deserialize CPTs: str node keys + str tuple keys → int keys + tuple keys
        cpts_raw = info.get("cpts", {})
        cpts = {}
        for nd_str, cpt_str in cpts_raw.items():
            nd = int(nd_str)
            cpts[nd] = {}
            for k_str, v in cpt_str.items():
                k = () if k_str == "" else tuple(int(x) for x in k_str.split("|"))
                cpts[nd][k] = v

        # Domains: str node keys → int keys
        domains_raw = info.get("domains", {})
        domains = {int(k): v for k, v in domains_raw.items()}

        topo_order = [int(nd) for nd in info.get("topo_order", [])]
        parents_map = {int(k): [int(p) for p in v] for k, v in info.get("parents_map", {}).items()}
        latent_nodes = list(info.get("latent_nodes", []))

        state["_cpts"] = cpts
        state["_domains"] = domains
        state["_topo_order"] = topo_order
        state["_parents_map"] = parents_map
        state["_latent_nodes"] = latent_nodes
        state["tool_calls_used"] = 0
        return state

    def update_tool_args(
        self,
        tool_name: str,
        tool_args: dict,
        messages: vf.Messages,
        state: vf.State,
        **kwargs,
    ) -> dict:
        """Inject per-rollout CPT state into every tool call."""
        args = dict(tool_args)
        args["_cpts"] = state["_cpts"]
        args["_domains"] = state["_domains"]
        args["_topo_order"] = state["_topo_order"]
        args["_parents_map"] = state["_parents_map"]
        args["_latent_nodes"] = state["_latent_nodes"]
        return args

    @vf.stop
    def end_of_phase(self, messages: vf.Messages, state: vf.State, **kwargs) -> bool:
        """Stop conditions for the two-phase rollout.

        Turn 1 (declaration): stop only if <answer> is present.
        Turn 2+ (tool use / answer): stop if <answer> present OR no tool call made.
        """
        last_assistant = next(
            (m for m in reversed(messages) if m.get("role") == "assistant"), None
        )
        if not last_assistant:
            return False
        content = last_assistant.get("content", "") or ""
        has_answer = bool(re.search(r"<answer>\s*.+?\s*</answer>", content, re.DOTALL))
        has_tool_call = bool(last_assistant.get("tool_calls"))
        assistant_turn = sum(1 for m in messages if m.get("role") == "assistant")
        if assistant_turn == 1:
            # Declaration turn: only stop if answer is present (e.g. not_identifiable)
            return has_answer
        # Tool use / answer phase: stop if answer present OR no tool call made
        return has_answer or not has_tool_call

    async def env_response(
        self,
        messages: vf.Messages,
        state: vf.State,
        **kwargs,
    ) -> vf.Messages:
        """Execute tool calls with limit enforcement."""
        last_assistant = next(
            (m for m in reversed(messages) if m.get("role") == "assistant"), None
        )
        has_tool_call = bool(last_assistant and last_assistant.get("tool_calls"))

        if not has_tool_call:
            return []

        tool_calls_used = state.get("tool_calls_used", 0)

        if tool_calls_used >= MAX_TOOL_CALLS:
            # Model is still calling tools after limit — return error
            return [{"role": "tool", "content": "Error: tool call limit reached. Provide your answer in <answer> tags."}]

        # Execute the tool via parent class
        result_messages = await super().env_response(messages, state, **kwargs)

        tool_calls_used += 1
        state["tool_calls_used"] = tool_calls_used

        if tool_calls_used >= MAX_TOOL_CALLS and result_messages:
            # Append limit warning to last tool result
            last_result = dict(result_messages[-1])
            current_content = last_result.get("content", "")
            last_result["content"] = (
                current_content
                + f"\n\n[You have used all {MAX_TOOL_CALLS} available tool calls. Provide your final answer now.]"
            )
            result_messages = list(result_messages[:-1]) + [last_result]

        return result_messages


# ─────────────────────────────────────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────────────────────────────────────


def load_environment(dataset: vf.Dataset | None = None, **kwargs) -> CausalATEEnv:
    """Instantiate CausalATEEnv from a pre-built HuggingFace Dataset.

    If dataset is None, loads from HuggingFace Hub (irfanjamil/causal-reasoning-ate).
    """
    if dataset is None:
        from datasets import load_dataset
        ds = load_dataset("irfanjamil/causal-reasoning-ate", split="train")
        dataset = ds
    return CausalATEEnv(dataset=dataset, **kwargs)
