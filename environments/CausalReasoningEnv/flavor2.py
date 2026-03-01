"""Flavor 2 — DAG + Observational Data -> Estimate ATE via Counting.

TODO: implement Flavor2Env and load_flavor2().

The model is given a DAG and discrete observational data (N=2000 rows).
It must:
  (a) Determine whether ATE is estimable from the available data.
  (b) If yes, estimate ATE = E[Y(do(X=1))] - E[Y(do(X=0))] via nonparametric
      stratified counting over the valid adjustment set.
  (c) If yes, estimate CATE for specified covariate values.

All variables are discrete (X binary, Y binary, Z multi-category).
"""

import verifiers as vf

from prompts import build_system_prompt

_F2_INTRO = """\
You will be given a Directed Acyclic Graph (DAG) representing a structural \
causal model and a sample of observational data (discrete variables). Nodes \
are variables. A directed edge A→B means A is a direct cause of B.\
"""

_F2_RESPONSE_FORMAT = """\
RESPONSE FORMAT
────────────────
Write all reasoning inside a <reasoning> block. After </reasoning>, write \
exactly one <answer> block using one of these two forms:

  Estimable (ATE and CATE computed):
      <answer>estimable: yes, ATE=[value], CATE=[value]</answer>

  Not estimable:
      <answer>estimable: no, reason=[explanation]</answer>

Rules: do not write <answer> inside <reasoning>; report ATE and CATE as \
decimal numbers rounded to 4 significant figures; if a required adjustment \
variable is unobserved in the data or treatment support is missing, the ATE \
is not estimable — do not produce a numeric estimate.\
"""

SYSTEM_PROMPT = build_system_prompt(_F2_INTRO, _F2_RESPONSE_FORMAT)


def load_flavor2() -> vf.Environment:
    raise NotImplementedError("Flavor 2 environment not yet implemented.")
