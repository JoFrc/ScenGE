"""Configurable threat loss for trajectory perturbation.

The original codebase shipped two near-duplicate ``ThreatLoss`` classes
(one in ``scripts/select_traj.py`` with 4 terms, one in the now-deleted
``scripts/perturb_traj.py`` with 6 terms). This module merges them into a
single class whose terms are toggled by their weights:

- ``w_ego``: pull the perturbed actor toward the ego trajectory
  (proximity term).
- ``w_occ``: encourage the perturbed actor to lie on the ego-adv segment
  (visual occlusion term).
- ``w_pressure``: time-discounted inverse distance to ego (decision-time
  pressure term, only meaningful when ``ego`` and the perturbed actor are
  the same length).
- ``w_proximity_adv``: pull the perturbed actor toward the adversary.
- ``w_smooth``: penalize jerky trajectories (second-order finite diff).
- ``w_dev``: penalize deviation from the original trajectory.
- ``w_yaw``: penalize unnatural yaw changes vs. the original yaw series.

A term is silently skipped when its weight is 0 *and* the relevant
reference signal (e.g. ``agent_orig_traj``) is omitted at the call site.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class ThreatLoss(nn.Module):
    def __init__(
        self,
        w_ego: float = 0.0,
        w_occ: float = 0.0,
        w_pressure: float = 0.0,
        w_proximity_adv: float = 0.0,
        w_smooth: float = 0.0,
        w_dev: float = 0.0,
        w_yaw: float = 0.0,
    ):
        super().__init__()
        self.w_ego = w_ego
        self.w_occ = w_occ
        self.w_pressure = w_pressure
        self.w_proximity_adv = w_proximity_adv
        self.w_smooth = w_smooth
        self.w_dev = w_dev
        self.w_yaw = w_yaw

    # ---- term primitives ----------------------------------------------

    @staticmethod
    def _proximity(traj_a: torch.Tensor, traj_b: torch.Tensor) -> torch.Tensor:
        return (traj_a - traj_b).norm(dim=1).mean()

    @staticmethod
    def _occlusion(
        agent_traj: torch.Tensor, ego_traj: torch.Tensor, adv_traj: torch.Tensor
    ) -> torch.Tensor:
        """Distance from the perturbed actor to the ego-adv segment."""
        ab = adv_traj - ego_traj
        ap = agent_traj - ego_traj
        ab_norm_sq = (ab ** 2).sum(dim=1, keepdim=True) + 1e-6
        t = ((ap * ab).sum(dim=1, keepdim=True)) / ab_norm_sq
        t = t.clamp(0.0, 1.0)
        projection = ego_traj + t * ab
        return torch.norm(agent_traj - projection, dim=1).pow(2).mean()

    @staticmethod
    def _pressure(agent_traj: torch.Tensor, ego_traj: torch.Tensor) -> torch.Tensor:
        T = len(agent_traj)
        timesteps = torch.arange(T, device=agent_traj.device)
        distance = torch.norm(agent_traj - ego_traj, dim=1)
        time_factor = torch.exp(-(T - timesteps) / T)
        return (time_factor / (distance + 1e-6)).mean()

    @staticmethod
    def _proximity_adv_min(
        agent_traj: torch.Tensor, adv_traj: torch.Tensor
    ) -> torch.Tensor:
        return torch.min(torch.norm(agent_traj - adv_traj, dim=1))

    @staticmethod
    def _smoothness(traj: torch.Tensor) -> torch.Tensor:
        return torch.norm(traj[2:] - 2 * traj[1:-1] + traj[:-2], dim=1).mean()

    @staticmethod
    def _deviation(traj: torch.Tensor, orig_traj: torch.Tensor) -> torch.Tensor:
        return torch.norm(traj - orig_traj, dim=1).mean()

    @staticmethod
    def _yaw(traj: torch.Tensor, orig_yaw: torch.Tensor) -> torch.Tensor:
        diff = traj[1:] - traj[:-1]
        traj_yaw = torch.atan2(diff[:, 1], diff[:, 0])
        return torch.abs(traj_yaw - orig_yaw[:-1]).mean()

    # ---- aggregated forward -------------------------------------------

    def forward(
        self,
        traj: torch.Tensor,
        ego_traj: torch.Tensor,
        adv_traj: torch.Tensor,
        orig_yaw: torch.Tensor,
        orig_traj: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Aggregate every enabled term into a single scalar."""
        loss = traj.new_tensor(0.0)

        if self.w_ego:
            loss = loss + self.w_ego * self._proximity(traj, ego_traj)

        if self.w_occ:
            loss = loss + self.w_occ * self._occlusion(traj, ego_traj, adv_traj)

        if self.w_pressure:
            # negative: original code maximized inverse distance
            loss = loss - self.w_pressure * self._pressure(traj, ego_traj)

        if self.w_proximity_adv:
            # negative: encourage minimum distance to adv to shrink
            loss = loss - self.w_proximity_adv * self._proximity_adv_min(traj, adv_traj)

        if self.w_smooth:
            loss = loss + self.w_smooth * self._smoothness(traj)

        if self.w_dev and orig_traj is not None:
            loss = loss + self.w_dev * self._deviation(traj, orig_traj)

        if self.w_yaw:
            loss = loss + self.w_yaw * self._yaw(traj, orig_yaw)

        return loss


# Two presets matching the original code paths.

DEFAULT_SELECT_WEIGHTS = dict(
    w_ego=0.3,
    w_occ=0.45,
    w_smooth=0.15,
    w_yaw=0.1,
)
"""Weights used by the original ``scripts/select_traj.py``."""

DEFAULT_PERTURB_WEIGHTS = dict(
    w_occ=1.0,
    w_pressure=1.0,
    w_proximity_adv=0.2,
    w_smooth=5.0,
    w_dev=2.0,
    w_yaw=0.5,
)
"""Weights used by the original ``scripts/perturb_traj.py``."""
