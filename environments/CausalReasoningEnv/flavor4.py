"""Flavor 4 — DAG + Observational Data -> Estimate the SCM.

TODO: implement Flavor4Env and load_flavor4().

The model is given a DAG (structure only) and N=1000 rows of observational data.
It must estimate the structural equation for each node by regressing on its
causal parents (per the DAG) — not on all correlated variables.

The core causal test is variable selection: using the DAG to identify the correct
parent regressors vs. naively including all correlated variables.
"""

import verifiers as vf


def load_flavor4() -> vf.Environment:
    raise NotImplementedError("Flavor 4 environment not yet implemented.")