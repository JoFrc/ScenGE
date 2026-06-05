"""
Scenario policy registry.

After Phase 2 cleanup, only the policies actually wired to the surviving
configs remain:

- ``standard`` / ``scenic``: dummy policy (no learnable scenario model).
- ``advsim`` / ``advtraj``: hard-coded policies driven by JSON parameters.
- ``lc``: REINFORCE-style continuous policy (Learning to Collide).
"""

from safebench.scenario.scenario_policy.dummy_policy import DummyPolicy
from safebench.scenario.scenario_policy.reinforce_continuous import REINFORCE
from safebench.scenario.scenario_policy.hardcode_policy import HardCodePolicy


SCENARIO_POLICY_LIST = {
    "standard": DummyPolicy,
    "scenic": DummyPolicy,
    "advsim": HardCodePolicy,
    "advtraj": HardCodePolicy,
    "lc": REINFORCE,
}
