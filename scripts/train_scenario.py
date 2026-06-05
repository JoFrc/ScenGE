"""Scenic-side training entry point.

Run a Scenic-driven scenario suite with ``mode=train_scenario`` so the
runner's ``select`` path picks the top-N most threatening scenes per route
and writes them back to ``scenic_data/scenario_{id}/scenario_{id}.json``.

Usage:

    python scripts/train_scenario.py \\
        --agent_cfg sac.yaml \\
        --scenario_cfg train_scenario_scenic.yaml \\
        --scenario_id 1 --port 2002 --tm_port 8002
"""

from __future__ import annotations

import argparse

import torch

from safebench.util.launch import REPO_ROOT, LaunchArgs, run_one


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent_cfg", type=str, default="sac.yaml")
    parser.add_argument(
        "--scenario_cfg", type=str, default="train_scenario_scenic.yaml"
    )
    parser.add_argument("--scenario_id", type=int, required=True)

    parser.add_argument("--ROOT_DIR", type=str, default=REPO_ROOT)
    parser.add_argument("--seed", "-s", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu"
    )

    parser.add_argument("--num_scenario", "-ns", type=int, default=1)
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--render", type=bool, default=True)
    parser.add_argument("--frame_skip", "-fs", type=int, default=1)
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--tm_port", type=int, default=8000)
    parser.add_argument("--fixed_delta_seconds", type=float, default=0.1)
    parser.add_argument("--max_episode_step", type=int, default=300)
    parser.add_argument("--auto_ego", action="store_true")
    parser.add_argument("--traj_root", type=str, default=None)
    parser.add_argument("--record_traj", action="store_true")

    return parser


def main() -> None:
    args = _build_parser().parse_args()
    launch = LaunchArgs(
        agent_cfg=args.agent_cfg,
        scenario_cfg=args.scenario_cfg,
        scenario_id=args.scenario_id,
        mode="train_scenario",
        output_dir="log",
        exp_name="exp",
        seed=args.seed,
        threads=args.threads,
        device=args.device,
        num_scenario=args.num_scenario,
        render=args.render,
        save_video=args.save_video,
        frame_skip=args.frame_skip,
        port=args.port,
        tm_port=args.tm_port,
        fixed_delta_seconds=args.fixed_delta_seconds,
        auto_ego=args.auto_ego,
        max_episode_step=args.max_episode_step,
        ROOT_DIR=args.ROOT_DIR,
        traj_root=args.traj_root,
        record_traj=args.record_traj,
    )
    run_one(launch)


if __name__ == "__main__":
    main()
