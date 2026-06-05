"""
Agent (ego) policy registry.

After Phase 4, the only ego policy left is the unified ``RLAgent`` from the
safe-RL framework, since it is the only one with checkpoints under
``safebench/agent/model_ckpt/safe_rl/``. SAC / PPO / TD3 are differentiated
solely by the per-config ``policy_name`` and ``load_dir``.
"""

from safebench.agent.safe_rl.rl_agent import RLAgent


AGENT_POLICY_LIST = {
    "rl": RLAgent,
}
