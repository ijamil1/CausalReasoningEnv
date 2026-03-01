"""Flavor 4 — DAG + Observational Data -> Estimate the SCM.

TODO: implement Flavor4Env and load_flavor4().

The model is given a DAG (structure only) and N=1000 rows of observational data.
It must estimate the structural equation for each node by regressing on its
causal parents (per the DAG) — not on all correlated variables.

The core causal test is variable selection: using the DAG to identify the correct
parent regressors vs. naively including all correlated variables.
"""

import verifiers as vf

from prompts import build_system_prompt

_F4_INTRO = """\
You will be given a Directed Acyclic Graph (DAG) and observational data. \
Nodes are variables. A directed edge A→B means A is a direct cause of B. \
Your task is to estimate the structural equation for the specified node by \
regressing it on its causal parents as given by the DAG — not on all \
correlated variables.\
"""

_F4_RESPONSE_FORMAT = """\
RESPONSE FORMAT
────────────────
Write all reasoning inside a <reasoning> block. After </reasoning>, write \
exactly one <answer> block in this form:

      <answer>[node] = [coeff1]·[parent1] + [coeff2]·[parent2] + ... + N(0, [sigma])</answer>

Rules: do not write <answer> inside <reasoning>; include one term per causal \
parent as identified from the DAG; report coefficients and sigma rounded to \
4 significant figures; use only the node's DAG parents as regressors.\
"""

SYSTEM_PROMPT = build_system_prompt(_F4_INTRO, _F4_RESPONSE_FORMAT)


def load_flavor4() -> vf.Environment:
    raise NotImplementedError("Flavor 4 environment not yet implemented.")
