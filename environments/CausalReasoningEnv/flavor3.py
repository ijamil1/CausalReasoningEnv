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


def load_flavor3() -> vf.Environment:
    raise NotImplementedError("Flavor 3 environment not yet implemented.")