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


def load_flavor2() -> vf.Environment:
    raise NotImplementedError("Flavor 2 environment not yet implemented.")