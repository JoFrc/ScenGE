"""Masked self-attention used to score actors and pick the key frame.

This is the same attention block as the original ``scripts/select_traj.py``;
factor-of-two cleanups only.

Score derivation (with ``N`` env actors over ``T`` timesteps):

1. Build queries from the ego + adversary trajectories (shape ``(2*T, D)``)
   and keys from the env actors (shape ``(N*T, D)``).
2. Apply a causal mask plus a multiplicative time-decay ``gamma**|t-t_j|``
   so future env steps cannot influence past ego/adv steps and the
   contribution of stale steps decays exponentially.
3. ``select_T`` reduces the resulting ``Att`` over the two query rows
   (``ego``, ``adv``), picks the top-k env actors and the most-attended
   timestep, and returns that timestep index.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """Causal self-attention with temporal decay weighting."""

    def __init__(self, N: int, T: int, dim: int, gamma: float):
        super().__init__()
        self.N = N
        self.T = T
        self.dim = dim
        self.gamma = gamma

    def forward(self, agent_tensor: torch.Tensor, env_tensor: torch.Tensor) -> torch.Tensor:
        Q = agent_tensor.view(-1, self.dim)  # (2*T, D)
        K = env_tensor.view(-1, self.dim)  # (N*T, D)

        t_j = torch.arange(self.N * self.T) % self.T
        t = torch.arange(2 * self.T) % self.T
        mask = t.unsqueeze(1) < t_j.unsqueeze(0)
        M = torch.zeros((2 * self.T, self.N * self.T), dtype=torch.float32)
        M = M.masked_fill_(mask, float("-inf"))

        delta = torch.clamp(t.unsqueeze(1) - t_j.unsqueeze(0), min=0)
        D = self.gamma ** delta

        attn = Q @ K.transpose(0, 1)  # (2*T, N*T)
        mean = attn.mean(dim=-1, keepdim=True)
        std = attn.std(dim=-1, keepdim=True, unbiased=False) + 1e-6
        attn = (attn - mean) / std
        return F.softmax(attn + M + torch.log(D), dim=1)


def select_T(
    att: torch.Tensor,
    N: int,
    T: int,
    la_ego: float,
    la_adv: float,
    top_k: int,
) -> int:
    """Pick the most threat-relevant timestep ``t`` from an attention map."""
    att = att.view(2, T, N, T)
    a_ego = att[0].view(T, N, T)
    a_adv = att[1:].view(T, N, T)

    # step 1: pick top-k perturbation targets
    s_ego = a_ego.sum(dim=(0, 2))
    s_adv = a_adv.sum(dim=(0, 2))
    score = la_ego * s_ego + la_adv * s_adv
    _, top_k_indexes = score.topk(top_k)

    # step 2: pick the dominant timestep across the chosen targets
    a_ego = a_ego.index_select(dim=1, index=top_k_indexes)
    a_adv = a_adv.index_select(dim=1, index=top_k_indexes)
    beta_ego = a_ego.sum(dim=(1, 2))
    beta_adv = a_adv.sum(dim=(1, 2))
    beta = la_ego * beta_ego + la_adv * beta_adv

    return torch.argmax(beta).item()
