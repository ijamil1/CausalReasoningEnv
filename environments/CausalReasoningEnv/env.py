"""CausalATEEnv — two-turn StatefulToolEnv for ATE estimation.

Turn 1 (declaration + tools): Model reasons about the DAG, writes a <set> tag,
  and makes all needed tool calls in the same response.
  - not_identifiable problems write <set></set><answer>not_identifiable</answer>
    with no tool calls — end_of_phase fires immediately.
Turn 2 (answer): Model receives tool results and writes <answer>ATE=...</answer>.

Reward components (weights: 0.05 / 0.30 / 0.15 / 0.50):
  format_compliance / set_valid / minimality / ate_accuracy

true_ATE is computed in gen.py using the same 6dp rounding as the tools, then
rounded to 4dp. A perfect model gets exact equality; ate_accuracy uses 5e-5
tolerance to absorb float representation differences.
"""

import json
import re
from itertools import product as itertools_product

import networkx as nx
import verifiers as vf
from datasets import Dataset, load_dataset
from networkx.algorithms.d_separation import is_d_separator

from prompts import SYSTEM_PROMPT

MAX_PARALLEL_TOOL_CALLS = 3  # frontdoor needs at most 3
MAX_TURNS = 2                # declaration + tools / answer


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
    """Extract declared identification set from the first assistant message only.

    Returns:
        list[int]  — declared node IDs (may be empty for empty identification set)
        None       — no <set> tag found in the first assistant message
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
            return None  # first assistant message found but no <set> tag
    return None  # no assistant messages


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
        if not is_d_separator(G_xbar, {X}, M, set()):
            return False
    except Exception:
        return False

    # Condition 3: M d-separated from Y by X in graph with M's outgoing edges removed
    G_mbar = G.copy()
    for m in M:
        G_mbar.remove_edges_from(list(G.out_edges(m)))
    try:
        if not is_d_separator(G_mbar, M, {Y}, {X}):
            return False
    except Exception:
        return False

    return True


# ─────────────────────────────────────────────────────────────────────────────
# Termination metrics (weight=0 — observability only)
# ─────────────────────────────────────────────────────────────────────────────


async def terminated_too_many_parallel_tool_calls(completion) -> float:
    """1.0 if rollout terminated because more than the max parallel tool calls were made."""
    for msg in completion:
        if msg.get("role") == "tool":
            content = msg.get("content", "")
            if "Error" in content and "parallel tool calls" in content:
                return 1.0
    return 0.0



# ─────────────────────────────────────────────────────────────────────────────
# Reward functions
# ─────────────────────────────────────────────────────────────────────────────


async def format_compliance(completion, state) -> float:
    """0.5 if <set> tag present in first assistant turn, 0.5 if valid <answer> tag present."""
    score = 0.0
    if state.get("declared_set") is not None:
        score += 0.5
    answer = _parse_answer(completion)
    if answer == "not_identifiable" or (answer and re.match(r"^ATE=[-+]?\d+(\.\d+)?$", answer)):
        score += 0.5
    return score


async def set_valid(info, state) -> float:
    """1.0 if declared set satisfies the identification criterion for this problem type."""
    if isinstance(info, str):
        info = json.loads(info)
    problem_type = info.get("problem_type", "")

    declared_set = state.get("declared_set")
    if declared_set is None:
        return 0.0  # identifiable problem but no <set> tag declared
    
    if problem_type == "not_identifiable":
        if len(declared_set) > 0:
            return 0
        return 1.0  # no set needed; answer block checked by ate_accuracy

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
            return 1.0 if is_d_separator(G_back, {X}, {Y}, Z) else 0.0
        except Exception:
            return 0.0

    if problem_type == "frontdoor":
        M = set(declared_set)
        if not M:
            return 0.0
        return 1.0 if _check_frontdoor_conditions(G, X, Y, M) else 0.0

    return 0.0


async def minimality(info, state) -> float:
    """Graded: 1.0 if declared set is minimal size, k/|declared| if valid superset."""
    if isinstance(info, str):
        info = json.loads(info)
    problem_type = info.get("problem_type", "")
    
    # Gate on validity
    validity = await set_valid(info, state)
    if validity == 0.0:
        return 0.0
    
    if problem_type == "not_identifiable":
        return 1.0  # full credit; no set to minimize
    
    declared_set = state.get("declared_set") or []
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
    return 1.0 if abs(ate_hat - true_ate) < 5e-5 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Environment class
# ─────────────────────────────────────────────────────────────────────────────


class CausalATEEnv(vf.StatefulToolEnv):
    """Two-phase environment for ATE estimation via probability query tools."""

    def __init__(self, dataset: Dataset, eval_dataset: Dataset | None = None, **kwargs):
        rubric = vf.Rubric(
            funcs=[format_compliance, set_valid, minimality, ate_accuracy],
            weights=[0.05, 0.30, 0.15, 0.50],
        )
        super().__init__(
            dataset=dataset,
            eval_dataset=eval_dataset,
            system_prompt=SYSTEM_PROMPT,
            tools=[],
            rubric=rubric,
            max_turns=MAX_TURNS,
            **kwargs,
        )
        self.add_tool(marginal, args_to_skip=["_cpts", "_domains", "_topo_order", "_parents_map", "_latent_nodes"])
        self.add_tool(conditional, args_to_skip=["_cpts", "_domains", "_topo_order", "_parents_map", "_latent_nodes"])
        rubric.add_metric(terminated_too_many_parallel_tool_calls)

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
    async def no_tools_called(self, state: vf.State) -> bool:
        return False

    @vf.stop
    async def end_of_phase(self, state: vf.State) -> bool:
        """Stop conditions for the two-turn rollout.

        Turn 1 (declaration + tools):
          Valid continuation (return False — proceed to env_response):
            tool calls present, no answer, <set> tag present.
          Stop (return True):
            - no <set> tag → format error
            - tool calls + answer → mutually exclusive, error
            - no tool calls + no answer → incomplete response, error
            - no tool calls + answer → valid early exit (e.g. not_identifiable); sets declared_set
        Turn 2 (answer): always stop.
        """
        trajectory = state.get("trajectory", [])
        if not trajectory:
            return False
        last_step = trajectory[-1]
        messages = list(last_step["prompt"]) + list(last_step["completion"])

        last_assistant = next(
            (m for m in reversed(messages) if m.get("role") == "assistant"), None
        )
        if not last_assistant:
            return False
        content = last_assistant.get("content", "") or ""
        has_answer = bool(re.search(r"<answer>\s*.+?\s*</answer>", content, re.DOTALL))
        has_tool_call = bool(last_assistant.get("tool_calls"))
        has_set = bool(re.search(r"<set>.*?</set>", content, re.DOTALL))
        assistant_turn = sum(1 for m in messages if m.get("role") == "assistant")

        if assistant_turn == 1:
            if not has_set:
                return True  # error: missing <set> tag
            if has_tool_call and has_answer:
                return True  # error: tool calls and answer are mutually exclusive
            if not has_tool_call and not has_answer:
                return True  # error: incomplete — neither tool calls nor answer
            if has_answer and not has_tool_call:
                state["declared_set"] = _parse_set(messages)
                return True  # valid early exit (not_identifiable)
            return False  # valid: tool calls present, no answer → execute tools

        return True  # turn 2: always stop

    async def env_response(
        self,
        messages: vf.Messages,
        state: vf.State,
        **kwargs,
    ) -> vf.Messages:
        """Turn 1 only: parse declared set, enforce tool call limit, execute tools.

        Only reached when end_of_phase returned False (tool calls present, no answer,
        <set> tag present). Turn 2 always stops via end_of_phase before reaching here.
        """
        last_assistant = next(
            (m for m in reversed(messages) if m.get("role") == "assistant"), None
        )
        tool_calls = last_assistant.get("tool_calls", []) if last_assistant else []

        # Parse and store declared set
        state["declared_set"] = _parse_set(messages)

        # Enforce parallel tool call limit
        if len(tool_calls) > MAX_PARALLEL_TOOL_CALLS:
            termination = [
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": f"Error: exceeded maximum of {MAX_PARALLEL_TOOL_CALLS} parallel tool calls. Rollout terminated.",
                }
                for tc in tool_calls
            ]
            state["final_env_response"] = termination
            return termination

        tool_messages = await super().env_response(messages, state, **kwargs)
        tool_messages.append({
            "role": "user",
            "content": (
                "Tool results are above. Now use this information and your domain knowledge to compute the ATE and write your final answer.\n"
                "You MUST end your response with exactly one <answer>ATE=...</answer> tag "
                "(or <answer>not_identifiable</answer>). Do NOT make any tool calls."
            ),
        })
        return tool_messages


# ─────────────────────────────────────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────────────────────────────────────


def load_environment(**kwargs) -> CausalATEEnv:
    """Instantiate CausalATEEnv from a pre-built HuggingFace Dataset.

    Load from HuggingFace Hub (irfanjamil/causal-reasoning-ate).
    """
    
    train_ds = load_dataset("irfanjamil/causal-reasoning-ate", split="train")
    eval_ds = load_dataset("irfanjamil/causal-reasoning-ate", split="eval")
    return CausalATEEnv(dataset=train_ds, eval_dataset=eval_ds, **kwargs)
