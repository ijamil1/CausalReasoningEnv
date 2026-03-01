"""Shared prompt constants for all four CausalReasoningEnv flavors.

Each flavor's SYSTEM_PROMPT is composed via build_system_prompt():

    SYSTEM_PROMPT = build_system_prompt(
        flavor_intro=<one or two sentences describing what inputs the model receives>,
        response_format=<RESPONSE FORMAT section specific to this flavor>,
    )

The shared header (expert identity) and CAUSAL_KNOWLEDGE block (graph
concepts, ATE/do-calculus, backdoor/front-door criteria, non-identifiability)
are identical across all four flavors. Only the intro and response format differ.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Shared header
# ─────────────────────────────────────────────────────────────────────────────

_HEADER = """\
CAUSAL INFERENCE — REFERENCE KNOWLEDGE

You are an expert in probabilistic graphical models, Bayesian networks, and \
structural causal models."""


# ─────────────────────────────────────────────────────────────────────────────
# Shared causal knowledge block (flavor-independent)
# ─────────────────────────────────────────────────────────────────────────────

CAUSAL_KNOWLEDGE = """\
GRAPH CONCEPTS
──────────────
Observed node: a variable that can be measured and included in a conditioning set.
Latent node: an unmeasured variable present in the causal structure but not in \
the data. Latent nodes cannot be conditioned upon since they are unmeasured/unobserved.

Path: a sequence of nodes connected by edges, ignoring direction.
Directed path: a path that follows every edge in its stated direction.
Backdoor path from X to Y: any undirected path between X and Y that starts \
with an arrow pointing INTO X.

Collider on a path: a node C where both adjacent edges on the path point \
INTO C (e.g. A→C←B). A collider BLOCKS the path by default. Conditioning \
on a collider — or any descendant of a collider — OPENS the path.
Non-collider on a path: any other node (fork: A←C→B, or chain: A→C→B). \
Conditioning on a non-collider BLOCKS the path at that node.

d-separation: A set Z d-separates X from Y in a DAG if it contains no \
descendant of X and every path between X and Y is blocked given Z, following \
the collider/non-collider rules above. d-separation is a graphical criterion \
for conditional independence: if Z d-separates X from Y, then X ⊥ Y | Z in \
every probability distribution consistent with the DAG's Markov factorization.

Descendant of X: any node reachable from X by following directed edges forward.

ATE AND DO-CALCULUS
────────────────────
do(X=x): an intervention that sets X to value x while accounting for all \
confounding between X's causes and Y.

ATE = E[Y | do(X=1)] − E[Y | do(X=0)]: the average treatment effect — \
the expected change in Y caused by setting X=1 versus X=0 via intervention.

ATE is identifiable if it equals some function of the observed distribution \
P(observed variables only), without reference to latent quantities. \
Identifiability hinges on conditional independence: if the observed variables \
can be used to make X and Y conditionally independent through every non-causal \
path (blocking all confounding while leaving the causal path intact), then the \
interventional distribution E[Y | do(X)] collapses to an expression involving \
only observational quantities, isolating the pure treatment effect.

BACKDOOR CRITERION
──────────────────
The backdoor graph for (X, Y) is the DAG with all edges out of X removed.

A set Z of observed variables satisfies the backdoor criterion for (X, Y) if:
  1. Z contains no descendant of X.
  2. Z d-separates X from Y in the backdoor graph.

The minimal adjustment set is the smallest Z satisfying these conditions. \
Multiple minimal sets may exist; all are equally valid. An empty set Z={} \
satisfies backdoor when X and Y are already d-separated in the backdoor graph.

FRONT-DOOR CRITERION
─────────────────────
An observed mediator set M satisfies the front-door criterion for (X, Y) if \
ALL three conditions hold:
  1. Every directed path from X to Y passes through at least one node in M \
(M collectively intercepts all causal paths from X to Y).
  2. There are no unblocked backdoor paths from X to any node in M.
  3. All backdoor paths from M to Y are blocked by X.

M may be a single node or a set of nodes. M cannot contain X and cannot contain Y.

PRIORITY RULE
─────────────
If both a valid backdoor adjustment set and a valid front-door mediator set \
exist, prefer the backdoor criterion — respond with the adjustment set answer.

NON-IDENTIFIABILITY of ATE
────────────────────
The ATE is not identifiable when no function of the observed distribution equals E[Y | do(X)].\
In other words, there exists no subset of the observed variables which allows isolation of the treatment effect.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Composer
# ─────────────────────────────────────────────────────────────────────────────

def build_system_prompt(flavor_intro: str, response_format: str) -> str:
    """Compose a complete system prompt from the shared knowledge block.

    Args:
        flavor_intro:    One or two sentences describing what inputs the model
                         receives for this flavor (e.g. "You will be given a
                         DAG and observational data...").
        response_format: The RESPONSE FORMAT section specific to this flavor.
    """
    return f"{_HEADER}\n\n{flavor_intro}\n\n{CAUSAL_KNOWLEDGE}\n\n{response_format}"
