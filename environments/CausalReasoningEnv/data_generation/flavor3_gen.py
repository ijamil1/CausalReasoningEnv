"""Flavor 3 data generation — DAG + Fully Specified SCM -> Compute ATE.

TODO: implement generate_flavor3_problems() and build_dataset().

SCM types to generate:
  - Linear (75%): V = sum(beta_i * parent_i) + N(0, sigma). ATE via Wright's path-tracing.
  - Nonlinear (25%): V = tanh(sum(beta_i * parent_i)) + N(0, 0.2) or quadratic mixtures.
  - ~15% canceling paths: ATE ~ 0 due to opposing direct and indirect path coefficients.

Each problem dict should include:
  edges, scm_equations (str), scm_type ("linear" | "nonlinear"),
  true_ATE, true_CATE_cases, has_canceling_paths.
"""


def generate_flavor3_problems(
    n_train: int = 250,
    n_eval: int = 100,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    raise NotImplementedError("Flavor 3 data generation not yet implemented.")


def build_dataset(problems: list[dict], format_fn) -> object:
    raise NotImplementedError("Flavor 3 dataset builder not yet implemented.")