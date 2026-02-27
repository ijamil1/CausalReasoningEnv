"""CausalReasoningEnv — Multi-flavor causal reasoning training environment.

Entry point exposing load_environment(weights) which returns a vf.EnvGroup
combining up to four causal reasoning task flavors:

  Index 0 — Flavor 1: Minimal adjustment set identification (IMPLEMENTED)
  Index 1 — Flavor 3: DAG + full SCM -> compute ATE              (TODO)
  Index 2 — Flavor 2: DAG + observational data -> estimate ATE   (TODO)
  Index 3 — Flavor 4: DAG + observational data -> estimate SCM   (TODO)

The weight ordering [F1, F3, F2, F4] matches the curriculum progression
defined in configs/vf-rl/phase*.toml. Sub-environments with weight 0 are
not instantiated (lazy loading), so only active flavors load their datasets.

Usage:
    from CausalReasoningEnv import load_environment

    # Flavor 1 only (phase 1 default):
    env = load_environment()

    # Custom weights via prime eval:
    # prime eval run CausalReasoningEnv -a '{"weights": [0.4, 0.6, 0.0, 0.0]}'
"""

import verifiers as vf

from flavor1 import load_flavor1
from flavor2 import load_flavor2
from flavor3 import load_flavor3
from flavor4 import load_flavor4


def load_environment(weights: list[float] | None = None) -> vf.Environment:
    """Load the CausalReasoningEnv multi-flavor environment.

    Args:
        weights: List of four floats [w_F1, w_F3, w_F2, w_F4] controlling
                 which flavor sub-environments are active and their sampling
                 probability. Flavors with weight 0 are not instantiated.
                 Defaults to [1.0, 0.0, 0.0, 0.0] (Flavor 1 only).

    Returns:
        A single Flavor1Env if only Flavor 1 is active, otherwise a
        vf.EnvGroup combining all active flavors.
    """
    if weights is None:
        weights = [1.0, 0.0, 0.0, 0.0]

    if len(weights) != 4:
        raise ValueError(f"weights must have exactly 4 elements, got {len(weights)}")

    loaders = [load_flavor1, load_flavor3, load_flavor2, load_flavor4]

    active_envs = []
    active_weights = []
    for loader, w in zip(loaders, weights):
        if w > 0:
            active_envs.append(loader())
            active_weights.append(w)

    if not active_envs:
        raise ValueError("All weights are 0 — at least one flavor must be active.")

    if len(active_envs) == 1:
        return active_envs[0]

    return vf.EnvGroup(active_envs, weights=active_weights)