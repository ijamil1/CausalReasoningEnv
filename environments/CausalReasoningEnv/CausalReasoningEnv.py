"""CausalReasoningEnv — ATE estimation via probability query tools.

Entry point exposing load_environment() which returns a CausalATEEnv.
"""

from env import load_environment

__all__ = ["load_environment"]
