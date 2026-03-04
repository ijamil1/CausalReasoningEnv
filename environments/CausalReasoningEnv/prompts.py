"""Shared prompt constants for all four CausalReasoningEnv flavors.

Each flavor's SYSTEM_PROMPT is composed via build_system_prompt():

    SYSTEM_PROMPT = build_system_prompt(
        flavor_intro=<one or two sentences describing what inputs the model receives>,
        task=<one sentence describing what the model must do>,
        response_format=<RESPONSE FORMAT section specific to this flavor>,
    )

The shared header (expert identity) is identical across all four flavors. Only the intro and response format differ.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Shared header
# ─────────────────────────────────────────────────────────────────────────────

_HEADER = """\
You are an expert in probabilistic graphical models, Bayesian networks, and \
structural causal models."""
# ─────────────────────────────────────────────────────────────────────────────
# Composer
# ─────────────────────────────────────────────────────────────────────────────

def build_system_prompt(flavor_intro: str, response_format: str, task: str = "") -> str:
    """Compose a complete system prompt from the shared knowledge block.

    Args:
        flavor_intro:    One or two sentences describing what inputs the model
                         receives for this flavor (e.g. "You will be given a
                         DAG and observational data...").
        task:            One sentence describing what the model must do
                         (e.g. "Determine whether the ATE is identifiable...").
        response_format: The RESPONSE FORMAT section specific to this flavor.
    """
    task_block = f"\nTASK\n────\n{task}\n" if task else ""
    return f"{_HEADER}\n\n{flavor_intro}\n{task_block}\n\n{response_format}"
