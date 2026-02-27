"""Flavor 2 data generation — DAG + Observational Data -> Estimate ATE via Counting.

TODO: implement generate_flavor2_problems() and build_dataset().

Problem types to generate:
  (a) Standard: valid adjustment set exists, all vars observed, enough data for all strata.
  (b) Multi-dimensional Z: >=2 adjustment vars each with >=3 categories; some sparse strata.
  (c) Unobserved adjustment variable: one required var absent from the data CSV.
  (d) Missing treatment support: all rows have X=0 or X=1.
  (e) Auxiliary variables: 1-3 extra columns in data not present in the DAG.

Each problem dict should include:
  edges, data_csv (str), X, Y, problem_type, identifiability_status,
  true_ATE (None for types c/d), true_CATE_cases, adjustment_set.
"""


def generate_flavor2_problems(
    n_train: int = 250,
    n_eval: int = 100,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    raise NotImplementedError("Flavor 2 data generation not yet implemented.")


def build_dataset(problems: list[dict], format_fn) -> object:
    raise NotImplementedError("Flavor 2 dataset builder not yet implemented.")
