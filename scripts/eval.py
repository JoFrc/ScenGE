"""Evaluation entry point for both Scenic-style and Carla-style scenarios.

Selects ``CarlaRunner`` vs. ``ScenicRunner`` automatically from the
``scenario_category`` of the loaded scenario yaml.

Usage:

    # ChatScene / Scenic eval:
    python scripts/eval.py --agent_cfg sac.yaml --scenario_cfg eval_scenic.yaml \\
        --scenario_id 1 --port 2002 --tm_port 8002

    # CARLA baseline eval (LC / CS / AdvSim / AdvTraj):
    python scripts/eval.py --agent_cfg sac.yaml --scenario_cfg lc.yaml \\
        --scenario_id 1 --route_id 0 --port 2002 --tm_port 8002
"""

from __future__ import annotations

import argparse

import torch

from safebench.util.launch import REPO_ROOT, LaunchArgs, run_one


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent_cfg", type=str, default="sac.yaml")
    parser.add_argument("--scenario_cfg", type=str, default="eval_scenic.yaml")
    parser.add_argument("--scenario_id", type=int, required=True)
    parser.add_argument("--test_epoch", type=int, default=None)

    parser.add_argument("--ROOT_DIR", type=str, default=REPO_ROOT)
    parser.add_argument("--seed", "-s", type=int, default=19980321)
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

    parser.add_argument("--ori_eval", action="store_true")
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--traj_root", type=str, default=None)
    parser.add_argument("--record_traj", action="store_true")

    return parser


def main() -> None:
    args = _build_parser().parse_args()
    launch = LaunchArgs(
        agent_cfg=args.agent_cfg,
        scenario_cfg=args.scenario_cfg,
        scenario_id=args.scenario_id,
        mode="eval",
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
        ori_eval=args.ori_eval,
        replay=args.replay,
        traj_root=args.traj_root,
        record_traj=args.record_traj,
    )
    runner_run_kwargs = {}
    if args.test_epoch is not None:
        runner_run_kwargs["test_epoch"] = args.test_epoch
    run_one(launch, runner_run_kwargs=runner_run_kwargs)


if __name__ == "__main__":
    main()
