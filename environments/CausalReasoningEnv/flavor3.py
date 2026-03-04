"""Flavor 3 — DAG + Observational Data → Estimate the SCM.

TODO: implement Flavor3Env and load_flavor3().

Given a DAG (structure only) and observational data, the model must estimate
the structural equations — specifically the functional form and parameters for
a target node (Y) given its causal parents.

What it tests: whether the model uses the DAG to select the correct parent
regressors rather than naively including all correlated variables.

Environment type: vf.ToolEnv (multi-turn).
Tools: load_data, run_python.

See BENCHMARK_DESIGN.md § Flavor 3 for the full spec.
"""

import verifiers as vf

from prompts import build_system_prompt

_F3_INTRO = """\
You will be given a Directed Acyclic Graph (DAG) and observational data \
(N=1000 rows). Nodes are variables. A directed edge A→B means A is a \
direct cause of B. The data was generated from a linear Gaussian SCM.\
"""

_F3_TASK = """\
Estimate the structural equation for node Y: regress Y on its causal \
parents as identified from the DAG (not on correlated non-parents). \
Use the run_python tool to perform the regression. Report the coefficient \
for each parent and the noise standard deviation.\
"""

_F3_RESPONSE_FORMAT = """\
RESPONSE FORMAT
────────────────
Write all reasoning inside a <reasoning> block. After </reasoning>, write \
exactly one <answer> block:

  <answer>Y = [coeff1]·[parent1] + [coeff2]·[parent2] + N(0, [sigma])</answer>

Rules: use only Y's causal parents from the DAG as regressors (no intercept \
unless specified); report coefficients and sigma rounded to 4 significant \
figures; if Y has no parents, write <answer>Y = N(0, [sigma])</answer>.\
"""

SYSTEM_PROMPT = build_system_prompt(_F3_INTRO, _F3_RESPONSE_FORMAT, task=_F3_TASK)


def load_flavor3() -> vf.Environment:
    raise NotImplementedError("Flavor 3 environment not yet implemented.")
