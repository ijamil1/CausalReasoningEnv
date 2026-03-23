"""CausalATEEnv — single-turn SingleTurnEnv for causal identification.

The model receives a causal DAG description and must output:
  - A <declare method="..." nodes="..."/> tag specifying the identification method and node set
  - One or more <marginal variables="..."/> or <conditional query="..." given="..."/> tags
    specifying the probability queries needed to compute the causal effect

Reward components (weights: 0.05 / 0.35 / 0.35 / 0.05 / 0.20):
  format_compliance / method_validity / set_validity / minimality / process_correctness
"""

import json
import re

import networkx as nx
import verifiers as vf
from datasets import Dataset, load_dataset

from data_generation.gen import is_valid_backdoor_set, is_valid_frontdoor_set, is_valid_iv
from prompts import SYSTEM_PROMPT


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────


def _reconstruct_graph(info: dict) -> nx.DiGraph:
    """Rebuild NetworkX DiGraph from stored edges in info dict."""
    G = nx.DiGraph()
    G.add_nodes_from(info["nodes"])
    G.add_edges_from([tuple(e) for e in info["edges"]])
    return G


def _parse_xml_tool_calls(content: str) -> list[dict]:
    """Extract <declare/>, <marginal/>, <conditional/> self-closing tags from content."""
    calls = []
    for m in re.finditer(r'<declare\s+([^/>]*)/>', content):
        attrs = dict(re.findall(r'(\w+)="([^"]*)"', m.group(1)))
        nodes = [n.strip() for n in attrs.get("nodes", "").split(",") if n.strip()]
        calls.append({"name": "declare", "method": attrs.get("method", ""), "nodes": nodes})
    for m in re.finditer(r'<marginal\s+([^/>]*)/>', content):
        attrs = dict(re.findall(r'(\w+)="([^"]*)"', m.group(1)))
        variables = [n.strip() for n in attrs.get("variables", "").split(",") if n.strip()]
        calls.append({"name": "marginal", "variables": variables})
    for m in re.finditer(r'<conditional\s+([^/>]*)/>', content):
        attrs = dict(re.findall(r'(\w+)="([^"]*)"', m.group(1)))
        query = [n.strip() for n in attrs.get("query", "").split(",") if n.strip()]
        given = [n.strip() for n in attrs.get("given", "").split(",") if n.strip()]
        calls.append({"name": "conditional", "query": query, "given": given})
    return calls


def _parse_declaration(completion: list) -> tuple[str | None, list[int] | None]:
    """Return (method, nodes) from <declare/> tag in last assistant message, or (None, None)."""
    last = next((m for m in reversed(completion) if m.get("role") == "assistant"), None)
    if not last:
        return None, None
    content = last.get("content", "") or ""
    calls = _parse_xml_tool_calls(content)
    declare_calls = [c for c in calls if c["name"] == "declare"]
    if len(declare_calls) != 1:
        return None, None
    decl = declare_calls[0]
    if not decl:
        return None, None
    method = decl.get("method", "").strip().lower()
    if method not in ("backdoor", "frontdoor", "iv"):
        return None, None
    try:
        nodes = [int(n) for n in decl.get("nodes", [])]
    except (ValueError, TypeError):
        return None, None
    return method, nodes


# ─────────────────────────────────────────────────────────────────────────────
# Reward functions
# ─────────────────────────────────────────────────────────────────────────────


def _all_ints(*str_lists) -> bool:
    """Return True if every element in every list is parseable as an integer."""
    for lst in str_lists:
        for v in lst:
            try:
                int(v)
            except (ValueError, TypeError):
                return False
    return True


async def format_compliance(completion) -> float:
    """0.0 if the response is missing a valid <declare/>, has no probability queries,
    or any query/given/variables/nodes field contains non-integer values; 1.0 otherwise."""
    last = next((m for m in reversed(completion) if m.get("role") == "assistant"), None)
    if not last:
        return 0.0
    content = last.get("content", "") or ""
    calls = _parse_xml_tool_calls(content)
    method, nodes = _parse_declaration(completion)
    has_declare = method is not None
    has_prob = any(c["name"] in ("marginal", "conditional") for c in calls)
    if not has_declare or not has_prob:
        return 0.0
    
    # all probability query fields must contain integers
    prob_calls_ct = 0
    for c in calls:
        if c["name"] == "marginal":
            if not _all_ints(c.get("variables", [])):
                return 0.0
            prob_calls_ct += 1
        elif c["name"] == "conditional":
            if not _all_ints(c.get("query", []), c.get("given", [])):
                return 0.0
            prob_calls_ct += 1
    if prob_calls_ct > 3:
        return 0.0
    return 1.0


async def method_validity(completion, info) -> float:
    """1.0 if the declared method matches the problem's identification method."""
    if isinstance(info, str):
        info = json.loads(info)
    method, _ = _parse_declaration(completion)
    if method is None:
        return 0.0
    return 1.0 if method == info.get("identification_methods", [None])[0] else 0.0


async def set_validity(completion, info) -> float:
    """1.0 if the declared node set correctly identifies the causal effect for the declared method."""
    if isinstance(info, str):
        info = json.loads(info)
    method, nodes = _parse_declaration(completion)
    if method is None:
        return 0.0
    if method != info.get("identification_methods", [None])[0]:
        return 0.0
    G = _reconstruct_graph(info)
    observed = set(info["observed_nodes"])
    X, Y = info["X"], info["Y"]
    try:
        if method == "backdoor":
            return 1.0 if is_valid_backdoor_set(G, X, Y, observed, nodes) else 0.0
        elif method == "frontdoor":
            return 1.0 if is_valid_frontdoor_set(G, X, Y, observed, nodes) else 0.0
        elif method == "iv":
            return 1.0 if (len(nodes) == 1 and is_valid_iv(G, X, Y, observed, nodes[0])) else 0.0
    except Exception:
        return 0.0
    return 0.0


async def minimality(completion, info) -> float:
    """Graded minimality: 1.0 if declared set equals minimal_set; k/|declared| if valid superset.
    For IV (single-node), 1.0 if correct instrument, 0.0 otherwise.
    Gated on set_validity=1.0."""
    if isinstance(info, str):
        info = json.loads(info)
    sv = await set_validity(completion, info)
    if sv < 1.0:
        return 0.0
    method, nodes = _parse_declaration(completion)
    minimal_set = info.get("minimal_set")
    if minimal_set is None:
        return 0.0
    if method == "iv":
        iv_instrument = info.get("iv_instrument")
        return 1.0 if (nodes and nodes[0] == iv_instrument) else 0.0
    declared_set = set(nodes)
    minimal = set(minimal_set)
    if declared_set == minimal:
        return 1.0
    d = len(declared_set)
    return round(len(minimal) / d, 4) if d > 0 else 0.0


def _prob_calls_from_turn1(completion: list) -> list[dict]:
    """Return marginal/conditional call dicts parsed from the last assistant message."""
    last = next((m for m in reversed(completion) if m.get("role") == "assistant"), None)
    if not last:
        return []
    content = last.get("content", "") or ""
    return [c for c in _parse_xml_tool_calls(content) if c["name"] in ("marginal", "conditional")]


def _to_int_set(arg) -> set:
    if not arg:
        return set()
    return {int(v) for v in arg}


def _has_marginal_of(calls: list[dict], required: set, observed: set) -> bool:
    """True if any marginal call's variables is a superset of required and all variables are observed."""
    for c in calls:
        if c["name"] != "marginal":
            continue
        vars_set = _to_int_set(c.get("variables", []))
        if vars_set - observed:
            continue  # references non-observed nodes
        if required.issubset(vars_set):
            return True
    return False


def _has_conditional_of(calls: list[dict], query: set, given: set, observed: set) -> bool:
    """True if any conditional call has query as a subset of its query vars, exactly the required given vars,
    and all referenced nodes are observed.

    Superset is allowed for query (extra query vars can be marginalized out).
    Exact match is required for given (conditioning on a superset or subset changes the distribution).
    """
    for c in calls:
        if c["name"] != "conditional":
            continue
        query_set = _to_int_set(c.get("query", []))
        given_set = _to_int_set(c.get("given", []))
        if (query_set | given_set) - observed:
            continue  # references non-observed nodes
        if query.issubset(query_set) and given_set == given:
            return True
    return False


async def process_correctness(completion, info) -> float:
    """Proportional reward for specifying probability queries that efficiently target causal identification.

    Gated on set_validity=1.0. Uses the model's declared node set as the reference.

    Per method, defines a set of ideal probability queries. Score = (targets hit) / (total targets).

      backdoor_empty    (1 target):  conditional(Y | X)
      backdoor_standard (2 targets): marginal(Z), conditional(Y | X∪Z)
      frontdoor         (3 targets): conditional(M | X), marginal(X), conditional(Y | X∪M)
      iv                (2 targets): [marginal(Z,Y) or conditional(Y|Z)],
                                     [marginal(Z,X) or conditional(X|Z)]

    Calls covering a superset of the required nodes also satisfy the target.
    """
    if isinstance(info, str):
        info = json.loads(info)
    sv = await set_validity(completion, info)
    fc = await format_compliance(completion)
    if sv < 1.0 or fc < 1.0:
        return 0.0

    method, nodes = _parse_declaration(completion)
    declared_set = set(nodes) if nodes else set()
    ptype = info.get("problem_type", "")
    X = info.get("X")
    Y = info.get("Y")
    observed = set(info.get("observed_nodes", []))

    calls = _prob_calls_from_turn1(completion)
    if not calls:
        return 0.0

    if ptype == "backdoor_empty" and not len(declared_set):
        return 1.0 if _has_conditional_of(calls, {Y}, {X}, observed) else 0.0

    elif ptype == "backdoor_standard" or ptype == "backdoor_empty":
        hits = sum([
            _has_marginal_of(calls, declared_set, observed),
            _has_conditional_of(calls, {Y}, {X} | declared_set, observed),
        ])
        return hits / 2

    elif ptype == "frontdoor":
        hits = sum([
            _has_conditional_of(calls, declared_set, {X}, observed),
            _has_marginal_of(calls, {X}, observed),
            _has_conditional_of(calls, {Y}, {X} | declared_set, observed),
        ])
        return hits / 3

    elif ptype == "iv":
        Z = next(iter(declared_set))
        hit_y = _has_marginal_of(calls, {Z, Y}, observed) or _has_conditional_of(calls, {Y}, {Z}, observed)
        hit_x = _has_marginal_of(calls, {Z, X}, observed) or _has_conditional_of(calls, {X}, {Z}, observed)
        return (hit_y + hit_x) / 2

    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Environment class
# ─────────────────────────────────────────────────────────────────────────────


class CausalATEEnv(vf.SingleTurnEnv):
    """Single-turn environment for causal identification via XML-tag structured output."""

    def __init__(self, dataset: Dataset, eval_dataset: Dataset | None = None, **kwargs):
        rubric = vf.Rubric(
            funcs=[
                format_compliance,
                method_validity,
                set_validity,
                minimality,
                process_correctness,
            ],
            weights=[0.1, 0.3, 0.3, 0.0, 0.3],
        )
        super().__init__(
            dataset=dataset,
            eval_dataset=eval_dataset,
            system_prompt=SYSTEM_PROMPT,
            rubric=rubric,
            **kwargs,
        )


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
