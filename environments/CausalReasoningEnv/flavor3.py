"""Flavor 3 — DAG + Fully Specified SCM -> Compute ATE.

TODO: implement Flavor3Env and load_flavor3().

The model is given a DAG and complete structural equations (functional form,
parameter values, noise distributions). It must compute ATE analytically:
  - Linear SCMs (75%): exact numeric ATE via Wright's path-tracing.
  - Nonlinear SCMs (25%): substituted symbolic formula evaluable by the grader.

All problems must have active confounding (at least one backdoor path) so that
E[Y|X=x] != E[Y|do(X=x)]. ~15% of problems have canceling paths (ATE ~ 0).
"""

import verifiers as vf

from prompts import build_system_prompt

_F3_INTRO = """\
You will be given a Directed Acyclic Graph (DAG) and fully specified \
structural equations with parameter values and noise distributions. Nodes \
are variables. A directed edge A→B means A is a direct cause of B.\
"""

_F3_RESPONSE_FORMAT = """\
RESPONSE FORMAT
────────────────
Write all reasoning inside a <reasoning> block. After </reasoning>, write \
exactly one <answer> block using one of these two forms:

  Linear SCM — numeric ATE and CATE:
      <answer>ATE=[value], CATE=[value]</answer>

  Nonlinear SCM — symbolic formula for ATE and CATE:
      <answer>ATE=[formula], CATE=[formula]</answer>

Rules: do not write <answer> inside <reasoning>; for linear SCMs report \
numbers rounded to 4 significant figures; for nonlinear SCMs write an \
explicit expectation expression with the structural equations substituted in \
(e.g. ATE = E_{Z~N(0,1)}[f(1, Z) - f(0, Z)]).\
"""

SYSTEM_PROMPT = build_system_prompt(_F3_INTRO, _F3_RESPONSE_FORMAT)


def load_flavor3() -> vf.Environment:
    raise NotImplementedError("Flavor 3 environment not yet implemented.")
