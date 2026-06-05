"""Threat amplification: attention-based segment selection + PGD perturbation.

The four submodules cover one concern each:

- :mod:`safebench.scenge.threat.io` -- trajectory alignment / tensor
  packaging / CSV writers
- :mod:`safebench.scenge.threat.attention` -- masked self-attention with
  temporal decay used to score actors and pick the perturbation key frame
- :mod:`safebench.scenge.threat.losses` -- a single configurable
  ``ThreatLoss`` collecting every individual term used by the paper
- :mod:`safebench.scenge.threat.pgd` -- a small PGD optimizer that perturbs
  one segment of one actor at a time
"""

from safebench.scenge.threat.attention import SelfAttention, select_T
from safebench.scenge.threat.io import align_trajectories, process_trajectories
from safebench.scenge.threat.losses import ThreatLoss
from safebench.scenge.threat.pgd import perturb_segment

__all__ = [
    "SelfAttention",
    "select_T",
    "align_trajectories",
    "process_trajectories",
    "ThreatLoss",
    "perturb_segment",
]
