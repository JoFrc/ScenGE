"""Trajectory I/O helpers shared between the segment selector and PGD."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch


def align_trajectories(traj_data: Dict[str, List[Dict[str, Any]]]) -> pd.DataFrame:
    """Project per-actor JSON trajectories onto their common timestamps."""
    timestamp_sets = [{step["t"] for step in traj_data[actor]} for actor in traj_data]
    common_ts = sorted(set.intersection(*timestamp_sets))

    aligned: Dict[str, List[Any]] = {"time": common_ts}
    for actor, steps in traj_data.items():
        ts_map = {step["t"]: step for step in steps}
        aligned[f"{actor}_x"] = [ts_map[ts]["x"] for ts in common_ts]
        aligned[f"{actor}_y"] = [ts_map[ts]["y"] for ts in common_ts]

    df = pd.DataFrame(aligned).dropna()
    return df[df.columns.sort_values()]


def process_trajectories(
    traj_data: Dict[str, List[Dict[str, Any]]],
) -> Tuple[torch.Tensor, torch.Tensor, int, int, int]:
    """Build ``agent_tensor=(2,T,D)`` (ego + adv) and ``env_tensor=(N,T,D)``."""
    env_ids = [k for k in traj_data if k not in ("ego", "adv")]
    N = len(env_ids)

    aligned_df = align_trajectories(
        {
            "ego": traj_data["ego"],
            "adv": traj_data["adv"],
            **{k: traj_data[k] for k in env_ids},
        }
    )
    T = aligned_df.shape[0]
    dim = 2

    ego_tensor = torch.tensor(
        aligned_df[["ego_x", "ego_y"]].values, dtype=torch.float32
    ).unsqueeze(0)
    adv_tensor = torch.tensor(
        aligned_df[["adv_x", "adv_y"]].values, dtype=torch.float32
    ).unsqueeze(0)
    agent_tensor = torch.cat((ego_tensor, adv_tensor), dim=0)
    assert agent_tensor.shape == (2, T, dim)

    env_tensor = torch.empty((0, T, dim), dtype=torch.float32)
    for env in env_ids:
        env_xy = torch.tensor(
            aligned_df[[f"{env}_x", f"{env}_y"]].values, dtype=torch.float32
        ).unsqueeze(0)
        env_tensor = torch.cat((env_tensor, env_xy), dim=0)
    assert env_tensor.shape == (N, T, dim)

    return agent_tensor, env_tensor, N, T, dim


def write_perturbed_csv(
    save_path: str,
    final_traj_xy: np.ndarray,
    z_values: List[float],
    speeds: List[float],
    yaw_values: np.ndarray,
) -> None:
    """Save a perturbed trajectory back as a CSV next to its sibling actors."""
    df = pd.DataFrame(
        {
            "x": final_traj_xy[:, 0],
            "y": final_traj_xy[:, 1],
            "z": z_values,
            "v": speeds,
            "yaw": yaw_values,
        }
    )
    df.to_csv(save_path, index=False)
