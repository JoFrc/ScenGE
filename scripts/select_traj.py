"""Thin CLI around :mod:`safebench.scenge.threat`.

For each ``traj_dir/<scene>.json`` under ``--folder``:

1. Align the ego / adv / env actor trajectories.
2. Score actors with masked self-attention + temporal decay.
3. Pick the dominant timestep ``t`` and a ``T // 4`` sized window.
4. Run PGD on each env actor's segment with :class:`ThreatLoss`.
5. Write per-actor CSVs to
   ``{folder}/perturbed_traj/perturbed_<scene>/<actor>.csv``.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from safebench.scenge.threat.attention import SelfAttention, select_T
from safebench.scenge.threat.io import process_trajectories, write_perturbed_csv
from safebench.scenge.threat.losses import DEFAULT_SELECT_WEIGHTS, ThreatLoss
from safebench.scenge.threat.pgd import PGDConfig, perturb_segment


def _build_window(t: int, T: int) -> tuple[int, int]:
    window = max(T // 4, 1)
    start = max(0, min(T - window, t - window // 2))
    return start, start + window


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True)
    args = parser.parse_args()

    perturbed_root = os.path.join(args.folder, "perturbed_traj")
    os.makedirs(perturbed_root, exist_ok=True)

    traj_folder = os.path.join(args.folder, "traj_dir")
    loss_fn = ThreatLoss(**DEFAULT_SELECT_WEIGHTS)
    pgd_cfg = PGDConfig()
    total_time = 0.0

    for json_file in os.listdir(traj_folder):
        start_time = time.time()
        with open(os.path.join(traj_folder, json_file), "r") as fh:
            traj_data = json.load(fh)

        agent_tensor, env_tensor, N, T, dim = process_trajectories(traj_data)

        att_model = SelfAttention(N, T, dim, gamma=0.8)
        att = att_model(agent_tensor, env_tensor)

        top_k = min(5, N)
        t_idx = select_T(att, N, T, la_ego=0.8, la_adv=0.2, top_k=top_k)
        start_t, end_t = _build_window(t_idx, T)

        ego_traj = agent_tensor[0]
        adv_traj = agent_tensor[1]
        env_ids = [k for k in traj_data if k not in ("ego", "adv")]

        save_folder = os.path.join(perturbed_root, f"perturbed_{json_file[:-5]}")
        os.makedirs(save_folder, exist_ok=True)

        for env_idx, env_id in enumerate(env_ids):
            env_traj_full = env_tensor[env_idx].cpu()
            orig_yaw = torch.tensor(
                [step["yaw"] for step in traj_data[env_id][:T]], dtype=torch.float32
            )

            def _on_step(step: int, loss_value: float) -> None:
                print(f"Step {step}, Loss: {loss_value:.4f}")

            final_traj = perturb_segment(
                env_traj_full,
                start_t,
                end_t,
                ego_traj,
                adv_traj,
                orig_yaw,
                loss_fn,
                cfg=pgd_cfg,
                on_step=_on_step,
            )

            orig_speeds = [step["v"] for step in traj_data[env_id][:T]]
            z_values = [step["z"] for step in traj_data[env_id][:T]]
            write_perturbed_csv(
                save_path=os.path.join(save_folder, f"{env_id}.csv"),
                final_traj_xy=final_traj.numpy(),
                z_values=z_values,
                speeds=orig_speeds,
                yaw_values=orig_yaw.numpy(),
            )
            print(f"Saved perturbed trajectory for {env_id} in {json_file}")

        elapsed = time.time() - start_time
        total_time += elapsed
        print(f"Processed {json_file} in {elapsed:.2f} seconds")

    print(f"Total processing time: {total_time:.2f} seconds")


if __name__ == "__main__":
    main()
