"""PGD optimizer that perturbs one trajectory segment of one actor.

Algorithm (matches the original ``scripts/select_traj.py`` exactly):

1. Slice ``[start_t, end_t]`` from the env actor's trajectory.
2. Min-max normalize the segment to ``[0, 1]^2``.
3. Initialize an additive perturbation ``delta ~ N(0, init_sigma)``.
4. Adam-optimize ``-loss(traj_norm + delta)`` (gradient ascent on threat).
5. After each step, project ``delta`` to the L-infinity ball ``[-eps, eps]``.
6. Splice the optimized segment back into the full trajectory and return.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PGDConfig:
    epsilon: float = 0.05
    init_sigma: float = 0.01
    lr: float = 0.005
    steps: int = 100
    log_every: int = 20


def perturb_segment(
    env_traj_full: torch.Tensor,
    start_t: int,
    end_t: int,
    ego_traj: torch.Tensor,
    adv_traj: torch.Tensor,
    orig_yaw: torch.Tensor,
    loss_fn,
    cfg: PGDConfig = PGDConfig(),
    on_step=None,
) -> torch.Tensor:
    """Run PGD on a single ``[start_t, end_t]`` segment and return the
    optimized full-length trajectory tensor.

    ``loss_fn`` should follow the :class:`ThreatLoss` signature
    ``loss_fn(traj, ego, adv, orig_yaw, orig_traj=...)``.
    """
    orig_segment = env_traj_full[start_t:end_t].cpu()
    min_xy = orig_segment.min(dim=0).values
    max_xy = orig_segment.max(dim=0).values
    traj_norm = (orig_segment - min_xy) / (max_xy - min_xy + 1e-6)

    delta = torch.randn_like(traj_norm) * cfg.init_sigma
    delta.requires_grad = True
    optimizer = torch.optim.Adam([delta], lr=cfg.lr, betas=(0.5, 0.999))

    for step in range(cfg.steps):
        optimizer.zero_grad()

        candidate_norm = traj_norm + delta
        candidate_denorm = candidate_norm * (max_xy - min_xy) + min_xy

        perturbed = env_traj_full.clone()
        perturbed[start_t:end_t] = candidate_denorm

        loss = loss_fn(
            perturbed,
            ego_traj,
            adv_traj,
            orig_yaw,
            orig_traj=env_traj_full,
        )
        # gradient ascent: minimize -loss
        (-loss).backward()
        optimizer.step()
        delta.data = torch.clamp(delta.data, -cfg.epsilon, cfg.epsilon)

        if on_step is not None and step % cfg.log_every == 0:
            on_step(step, float(loss.item()))

    optimized_segment = (traj_norm + delta).detach() * (max_xy - min_xy) + min_xy
    final = env_traj_full.clone()
    final[start_t:end_t] = optimized_segment
    return final
