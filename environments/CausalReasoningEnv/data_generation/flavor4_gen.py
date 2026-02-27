"""Flavor 4 data generation — DAG + Observational Data -> Estimate the SCM.

TODO: implement generate_flavor4_problems() and build_dataset().

Each problem gives the model a DAG and N=1000 rows of data. The model must
estimate structural equations by regressing each node on its causal parents
(per the DAG), not on all correlated variables.

Include distractor variables: nodes with high correlation to non-parents due
to shared ancestors.

Each problem dict should include:
  edges, data_csv (str), true_structural_equations: {node: {parent: coeff, noise_var: sigma}}.
"""


def generate_flavor4_problems(
    n_train: int = 250,
    n_eval: int = 100,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    raise NotImplementedError("Flavor 4 data generation not yet implemented.")


def build_dataset(problems: list[dict], format_fn) -> object:
    raise NotImplementedError("Flavor 4 dataset builder not yet implemented.")