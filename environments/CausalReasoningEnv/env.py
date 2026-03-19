"""CausalATEEnv — two-turn StatefulToolEnv for ATE/LATE estimation.

Turn 1 (declaration + tools): Model reasons about the DAG, calls declare(method, nodes),
  and makes all needed probability tool calls in the same response.
  env_response validates the declaration and executes tools, then offers Turn 2.
  Rollout terminates if: no declare call, 0 probability calls, or parallel limit exceeded.
Turn 2 (answer): Model receives tool results and writes <answer>ATE=...</answer>
  or <answer>LATE=...</answer>.

Reward components (weights: 0.05 / 0.125 / 0.125 / 0.10 / 0.50 / 0.10):
  format_compliance / method_validity / set_validity / minimality /
  ate_accuracy_binary / ate_accuracy_l2
"""

import json
import re
from itertools import product as itertools_product

import networkx as nx
import verifiers as vf
from datasets import Dataset, load_dataset

from data_generation.gen import is_valid_backdoor_set, is_valid_frontdoor_set, is_valid_iv
from prompts import SYSTEM_PROMPT

MAX_PARALLEL_TOOL_CALLS = 4  # declare + up to 3 probability tools (frontdoor worst case)



# ─────────────────────────────────────────────────────────────────────────────
# Exact probability inference (shared by both tools)
# ─────────────────────────────────────────────────────────────────────────────


def _joint_marginal(
    assignments: dict,
    cpts: dict,
    domains: dict,
    topo_order: list,
    parents_map: dict,
) -> float:
    """Compute P(assignments) by summing over all consistent full configurations.

    assignments: dict mapping node_id -> actual domain value (not index).
    domains: dict mapping node_id -> list of actual domain values.
    cpts: dict mapping node_id -> {parent_value_tuple -> scalar or list}.
    """
    prob_sum = 0.0
    for vals in itertools_product(*[domains[nd] for nd in topo_order]):
        config = dict(zip(topo_order, vals))
        if any(config[k] != v for k, v in assignments.items()):
            continue
        joint = 1.0
        for nd in topo_order:
            pa = parents_map[nd]
            pa_vals = tuple(config[p] for p in pa)
            cpt_entry = cpts[nd][pa_vals]
            v = config[nd]
            dom = domains[nd]
            if len(dom) == 2:
                joint *= cpt_entry if v == dom[1] else (1.0 - cpt_entry)
            else:
                joint *= cpt_entry[dom.index(v)]
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
        variables: List of node IDs (as integers) to compute the joint marginal for.
                   Example: [2, 3] returns P(node2, node3) for all value combos.
    """
    int_vars = [int(v) for v in variables]
    for v in int_vars:
        if v in _latent_nodes:
            return f"Error: node {v} is latent and not observable."
    lines = []
    for vals in itertools_product(*[_domains[v] for v in int_vars]):
        assignments = dict(zip(int_vars, vals))
        prob = _joint_marginal(assignments, _cpts, _domains, _topo_order, _parents_map)
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
        query: List of node IDs (as integers) for the query variables.
               Example: [4] — computes P(node4=v | given) for all values v of node4 and all strata of given.
        given: List of node IDs (as integers) for the conditioning variables.
               Example: [0, 2] — conditions on all value combinations of node0 and node2.
    """
    int_query = [int(v) for v in query]
    int_given = [int(v) for v in given]
    for v in int_query + int_given:
        if v in _latent_nodes:
            return f"Error: node {v} is latent and not observable."
    lines = []
    for given_vals in itertools_product(*[_domains[v] for v in int_given]):
        given_assignments = dict(zip(int_given, given_vals))
        denom = _joint_marginal(given_assignments, _cpts, _domains, _topo_order, _parents_map)
        if denom < 1e-10:
            given_str = ", ".join(f"node{v}={val}" for v, val in zip(int_given, given_vals))
            lines.append(f"P(... | {given_str}) = undefined (zero probability)")
            continue
        for query_vals in itertools_product(*[_domains[v] for v in int_query]):
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
            matches = re.findall(
                r"<answer>\s*((?:ATE|LATE)=[-+]?\d+(?:\.\d+)?)\s*</answer>",
                content,
                re.DOTALL,
            )
            return matches[-1].strip() if matches else None
    return None


def _reconstruct_graph(info: dict) -> nx.DiGraph:
    """Rebuild NetworkX DiGraph from stored edges in info dict."""
    G = nx.DiGraph()
    G.add_nodes_from(info["nodes"])
    G.add_edges_from([tuple(e) for e in info["edges"]])
    return G


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


# Substrings that identify Turn-1 format-violation terminations (lines 450/461/466/479/493).
# Does NOT include the method/set invalidity termination (line 521) — that's scored separately.
_FORMAT_TERMINATION_SUBSTRINGS = [
    "declare() was not called",
    "could not parse declare() arguments",
    "unknown method",
    "no probability tool calls were made",
    "exceeded maximum of",
]


# ─────────────────────────────────────────────────────────────────────────────
# Reward functions
# ─────────────────────────────────────────────────────────────────────────────


async def format_compliance(completion, state) -> float:
    """0.0 if the rollout violated format constraints; 1.0 otherwise.

    Returns 0.0 if any of:
      - Rollout terminated due to a format violation (missing declare, unparseable args,
        unknown method, no probability calls, or exceeded parallel call limit).
      - First assistant message contains an <answer> tag (answer belongs in Turn 2).
      - set_validity_score == 1.0 but no valid <answer> tag in the final assistant message.
    """
    # Check for format-violation terminations
    final_env_response = state.get("final_env_response")
    if final_env_response:
        for msg in final_env_response:
            content = msg.get("content", "") or ""
            if any(sub in content for sub in _FORMAT_TERMINATION_SUBSTRINGS):
                return 0.0

    # Answer tags must not appear in Turn 1
    first_assistant = next(
        (m for m in completion if m.get("role") == "assistant"), None
    )
    if first_assistant and re.search(r"<answer>", first_assistant.get("content", "") or ""):
        return 0.0

    # If model reached Turn 2 (set_validity == 1), it must have written a valid answer
    if state.get("set_validity_score", 0.0) >= 1.0 and not _parse_answer(completion):
        return 0.0

    return 1.0


async def method_validity(state) -> float:
    """1.0 if the declared method matches the problem's identification method.
    Reads the score cached by env_response to avoid recomputation."""
    return state.get("method_validity_score", 0.0)


async def set_validity(state) -> float:
    """1.0 if the declared node set correctly identifies the causal effect for the declared method.
    Reads the score cached by env_response to avoid recomputation."""
    return state.get("set_validity_score", 0.0)


async def minimality(info, state) -> float:
    """Graded minimality: 1.0 if declared set equals minimal_set; k/|declared| if valid superset.
    For IV (single-node), 1.0 if correct instrument, 0.0 otherwise.
    Gated on method_validity_score=1.0 and set_validity_score=1.0."""
    if state.get("set_validity_score", 0.0) < 1.0:
        return 0.0
    if isinstance(info, str):
        info = json.loads(info)
    minimal_set = info.get("minimal_set")
    declared_nodes = state.get("declared_nodes", [])
    if minimal_set is None:
        return 0.0
    declared_set = set(declared_nodes)
    minimal = set(minimal_set)
    if declared_set == minimal:
        return 1.0
    k = len(minimal)
    d = len(declared_set)
    return round(k / d, 4) if d > 0 else 0.0


async def ate_accuracy_binary(completion, info) -> float:
    """1.0 if |answer - true_target| < 0.001, else 0.0. Works for both ATE and LATE."""
    if isinstance(info, str):
        info = json.loads(info)
    answer = _parse_answer(completion)
    if not answer or answer is None:
        return 0.0
    ptype = info.get("problem_type", "")
    prefix = "LATE=" if ptype == "iv" else "ATE="
    m = re.match(rf"^{prefix}([-+]?\d+(?:\.\d+)?)$", answer)
    if not m:
        return 0.0
    try:
        val_hat = float(m.group(1))
    except ValueError:
        return 0.0
    true_val = info.get("true_LATE") if ptype == "iv" else info.get("true_ATE")
    if true_val is None:
        return 0.0
    return 1.0 if abs(val_hat - true_val) < 0.001 else 0.0


def _prob_calls_from_turn1(completion: list) -> list[tuple[str, dict]]:
    """Return (name, parsed_args) for probability tool calls in the first assistant turn."""
    for msg in completion:
        if msg.get("role") == "assistant":
            result = []
            for tc in (msg.get("tool_calls") or []):
                name = tc.get("function", {}).get("name", "")
                if name in ("marginal", "conditional"):
                    try:
                        args = json.loads(tc.get("function", {}).get("arguments", "{}"))
                    except Exception:
                        args = {}
                    result.append((name, args))
            return result
    return []


def _to_int_set(arg) -> set:
    if not arg:
        return set()
    return {int(v) for v in arg}


def _has_marginal_of(calls, required: set) -> bool:
    """True if any marginal call's variables is a superset of required."""
    return any(
        name == "marginal" and required.issubset(_to_int_set(args.get("variables", [])))
        for name, args in calls
    )


def _has_conditional_of(calls, query: set, given: set) -> bool:
    """True if any conditional call covers at least query in query and given in given."""
    return any(
        name == "conditional"
        and query.issubset(_to_int_set(args.get("query", [])))
        and given.issubset(_to_int_set(args.get("given", [])))
        for name, args in calls
    )


async def process_correctness(completion, info, state) -> float:
    """Proportional reward for making tool calls that efficiently target the causal identification.

    Gated on set_validity_score=1.0. Uses the model's declared
    node set (not minimal_set) as the reference, so the check is relative to what the model
    actually committed to.

    Per method, defines a set of ideal probability tool calls. Score = (targets hit) / (total targets).

      backdoor_empty    (1 target):  conditional(Y | X)
      backdoor_standard (2 targets): marginal(Z), conditional(Y | X∪Z)
      frontdoor         (3 targets): conditional(M | X), marginal(X), conditional(Y | X∪M)
      iv                (2 targets): [marginal(Z,Y) or conditional(Y|Z)],
                                     [marginal(Z,X) or conditional(X|Z)]

    Calls covering a superset of the required nodes also satisfy the target.
    """
    if state.get("set_validity_score", 0.0) < 1.0:
        return 0.0

    if isinstance(info, str):
        info = json.loads(info)

    ptype = info.get("problem_type", "")
    declared_set = set(state.get("declared_nodes") or [])
    X = info.get("X")
    Y = info.get("Y")

    calls = _prob_calls_from_turn1(completion)
    if not calls:
        return 0.0

    if ptype == "backdoor_empty":
        # Target: conditional(Y | X)
        return 1.0 if _has_conditional_of(calls, {Y}, {X}) else 0.0

    elif ptype == "backdoor_standard":
        # Targets: marginal(Z), conditional(Y | X∪Z)
        hits = sum([
            _has_marginal_of(calls, declared_set),
            _has_conditional_of(calls, {Y}, {X} | declared_set),
        ])
        return hits / 2

    elif ptype == "frontdoor":
        # Targets: conditional(M | X), marginal(X), conditional(Y | X∪M)
        hits = sum([
            _has_conditional_of(calls, declared_set, {X}),
            _has_marginal_of(calls, {X}),
            _has_conditional_of(calls, {Y}, {X} | declared_set),
        ])
        return hits / 3

    elif ptype == "iv":
        # Targets: a call giving E[Y|Z], a call giving E[X|Z]
        # declared_set is a single-element set containing the instrument
        Z = next(iter(declared_set))
        hit_y = _has_marginal_of(calls, {Z, Y}) or _has_conditional_of(calls, {Y}, {Z})
        hit_x = _has_marginal_of(calls, {Z, X}) or _has_conditional_of(calls, {X}, {Z})
        return (hit_y + hit_x) / 2

    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Environment class
# ─────────────────────────────────────────────────────────────────────────────


class CausalATEEnv(vf.StatefulToolEnv):
    """Two-phase environment for ATE/LATE estimation via probability query tools."""

    def __init__(self, dataset: Dataset, eval_dataset: Dataset | None = None, max_turns: int = 2, **kwargs):
        rubric = vf.Rubric(
            funcs=[
                format_compliance,
                method_validity,
                set_validity,
                minimality,
                ate_accuracy_binary,
                process_correctness,
            ],
            weights=[0.07, 0.145, 0.145, 0.0, 0.50, 0.14],
        )
        super().__init__(
            dataset=dataset,
            eval_dataset=eval_dataset,
            system_prompt=SYSTEM_PROMPT,
            tools=[],
            rubric=rubric,
            max_turns=max_turns,
            **kwargs,
        )
        self.add_tool(self.declare)
        self.add_tool(marginal, args_to_skip=["_cpts", "_domains", "_topo_order", "_parents_map", "_latent_nodes"])
        self.add_tool(conditional, args_to_skip=["_cpts", "_domains", "_topo_order", "_parents_map", "_latent_nodes"])
        rubric.add_metric(terminated_too_many_parallel_tool_calls)

    async def setup_state(self, state: vf.State) -> vf.State:
        """Deserialize CPTs and domain info from the problem info dict."""
        info = state.get("info") or {}
        if isinstance(info, str):
            info = json.loads(info)

        # Deserialize CPTs: str node keys + pipe-delimited str tuple keys → int keys + tuple keys
        cpts_raw = info.get("cpts", {})
        cpts = {}
        for nd_str, cpt_str in cpts_raw.items():
            nd = int(nd_str)
            cpts[nd] = {}
            for k_str, v in cpt_str.items():
                k = () if k_str == "" else tuple(int(x) for x in k_str.split("|"))
                cpts[nd][k] = v

        # Domains: str node keys → int keys; values are lists of actual domain values
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

        # Per-rollout declaration state
        state["declared_method"] = None
        state["declared_nodes"] = None
        state["method_validity_score"] = 0.0
        state["set_validity_score"] = 0.0

        return state

    async def declare(self, method: str, nodes: list) -> str:
        """Declare your identification method and the relevant node set.

        Args:
            method: "backdoor", "frontdoor", or "iv". Example: "backdoor"
            nodes: adjustment set (backdoor), mediator set (frontdoor),
                   or instrumental variable set (iv). Pass node IDs as integers.
                   Example: [1, 3]
        """
        return f"Declaration received: method={method}, nodes={nodes}"

    def update_tool_args(
        self,
        tool_name: str,
        tool_args: dict,
        messages: vf.Messages,
        state: vf.State,
        **kwargs,
    ) -> dict:
        """Inject per-rollout CPT state into marginal/conditional tool calls."""
        if tool_name not in ("marginal", "conditional"):
            return tool_args
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

        Turn 1:
          Stop if <answer> tag found (answer belongs in Turn 2, not Turn 1).
          Otherwise proceed to env_response, which handles declare validation,
          probability call checks, and parallel limit enforcement.
        Turn 2: always stop.
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
        assistant_turn = sum(1 for m in messages if m.get("role") == "assistant")

        if assistant_turn == 1:
            if has_answer:
                return True  # error: answer belongs in Turn 2
            return False  # proceed to env_response

        return True  # Turn 2: always stop

    async def env_response(
        self,
        messages: vf.Messages,
        state: vf.State,
        **kwargs,
    ) -> vf.Messages:
        """Turn 1 only: validate declare call, enforce limits, execute tools.

        Flow:
          1. Parse all tool calls from Turn 1 assistant message.
          2. If no declare call → terminate.
          3. Validate method; store declared_method + declared_nodes in state.
          4. If 0 probability calls → terminate.
          5. If total calls > MAX_PARALLEL_TOOL_CALLS → terminate.
          6. Compute and store process_validity_score.
          7. Execute all tools via super().env_response().
          8. Record prob_tool_calls in state.
          9. Append Turn 2 prompt.
        """
        last_assistant = next(
            (m for m in reversed(messages) if m.get("role") == "assistant"), None
        )
        tool_calls = (last_assistant.get("tool_calls") or []) if last_assistant else []

        # Step 2: Find declare call
        declare_call = next(
            (tc for tc in tool_calls if tc.get("function", {}).get("name") == "declare"),
            None,
        )
        if declare_call is None:
            termination = [{"role": "user", "content": "Error: declare() was not called. Rollout terminated."}]
            state["final_env_response"] = termination
            return termination

        # Step 3: Parse and validate declare args
        try:
            declare_args = json.loads(declare_call["function"]["arguments"])
            method = declare_args["method"]
            nodes_arg = declare_args["nodes"]
            declared_nodes = [int(n) for n in nodes_arg]
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            termination = [{"role": "user", "content": "Error: could not parse declare() arguments. Rollout terminated."}]
            state["final_env_response"] = termination
            return termination

        if method not in ("backdoor", "frontdoor", "iv"):
            termination = [{"role": "user", "content": f"Error: unknown method '{method}'. Must be 'backdoor', 'frontdoor', or 'iv'. Rollout terminated."}]
            state["final_env_response"] = termination
            return termination

        state["declared_method"] = method
        state["declared_nodes"] = declared_nodes

        # Step 4: Check probability calls
        probability_calls = [
            tc for tc in tool_calls
            if tc.get("function", {}).get("name") in ("marginal", "conditional")
        ]
        if not probability_calls:
            termination = [
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": "Error: no probability tool calls were made. Rollout terminated.",
                }
                for tc in tool_calls
                if tc.get("id")
            ]
            state["final_env_response"] = termination
            return termination

        # Step 5: Enforce parallel limit
        if len(tool_calls) > MAX_PARALLEL_TOOL_CALLS:
            termination = [
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": f"Error: exceeded maximum of {MAX_PARALLEL_TOOL_CALLS} parallel tool calls. Rollout terminated.",
                }
                for tc in tool_calls
                if tc.get("id")
            ]
            state["final_env_response"] = termination
            return termination

        # Step 6: Compute and store method_validity_score and set_validity_score
        info = state.get("info") or {}
        if isinstance(info, str):
            info = json.loads(info)
        problem_method = info.get("identification_methods", [None])[0]
        mv = 1.0 if method == problem_method else 0.0
        state["method_validity_score"] = mv

        if mv == 1.0:
            G = _reconstruct_graph(info)
            observed = set(info["observed_nodes"])
            X, Y = info["X"], info["Y"]
            if method == "backdoor":
                sv = 1.0 if is_valid_backdoor_set(G, X, Y, observed, declared_nodes) else 0.0
            elif method == "frontdoor":
                sv = 1.0 if is_valid_frontdoor_set(G, X, Y, observed, declared_nodes) else 0.0
            elif method == "iv":
                sv = 1.0 if (len(declared_nodes) == 1 and is_valid_iv(G, X, Y, observed, declared_nodes[0])) else 0.0
            else:
                sv = 0.0
        else:
            sv = 0.0
        state["set_validity_score"] = sv

        if mv < 1.0 or sv < 1.0:
            termination = [
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": "Error: declared method or set is not valid for this problem. Rollout terminated.",
                }
                for tc in tool_calls
                if tc.get("id")
            ]
            state["final_env_response"] = termination
            return termination

        # Step 7: Execute all tools via parent
        tool_messages = await super().env_response(messages, state, **kwargs)

        # Step 8: Append Turn 2 prompt
        turn2_content = (
            "Tool results are above. Now use this information and your domain knowledge "
            "to compute the causal effect and write your final answer.\n"
            "You MUST end your response with exactly one <answer> tag "
            "(rounded to 4 decimal places). ie: <answer>LATE=0.3821</answer> or <answer>ATE=-0.5880</answer>. Do NOT make any tool calls."
        )
        tool_messages.append({"role": "user", "content": turn2_content})
        return tool_messages


# ─────────────────────────────────────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────────────────────────────────────


def load_environment(max_turns: int = 2, **kwargs) -> CausalATEEnv | None:
    """Instantiate CausalATEEnv from a pre-built HuggingFace Dataset.

    Load from HuggingFace Hub (irfanjamil/causal-reasoning-ate).
    """
    if max_turns != 2:
        return None
    train_ds = load_dataset("irfanjamil/causal-reasoning-ate", split="train")
    eval_ds = load_dataset("irfanjamil/causal-reasoning-ate", split="eval")
    return CausalATEEnv(dataset=train_ds, eval_dataset=eval_ds, max_turns=max_turns, **kwargs)
