"""Shared launch plumbing for ``scripts/train_scenario.py`` and ``scripts/eval.py``.

Both entry scripts do the same setup: parse argparse, load the per-yaml
agent/scenario configs from ``safebench/{agent,scenario}/config``, fold
command-line overrides in, and instantiate the appropriate runner.

This module centralizes that into :func:`prepare_run_kwargs` so the entry
scripts stay tiny.
"""

from __future__ import annotations

import os
import os.path as osp
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from safebench.util.carla_setup import ensure_carla_pythonpath
from safebench.util.run_util import load_config
from safebench.util.torch_util import set_seed, set_torch_variable


REPO_ROOT = osp.abspath(osp.dirname(osp.dirname(osp.dirname(osp.realpath(__file__)))))
ensure_carla_pythonpath(REPO_ROOT)


@dataclass
class LaunchArgs:
    """Resolved launch arguments. Mirrors the argparse namespace fields."""

    agent_cfg: str
    scenario_cfg: str
    scenario_id: int
    mode: str  # "train_scenario" or "eval"
    output_dir: str
    exp_name: str
    seed: int
    threads: int
    device: str
    num_scenario: int
    render: bool
    save_video: bool
    frame_skip: int
    port: int
    tm_port: int
    fixed_delta_seconds: float
    auto_ego: bool
    max_episode_step: int
    ROOT_DIR: str
    # eval-only flags (ignored in train mode)
    ori_eval: bool = False
    replay: bool = False
    traj_root: Optional[str] = None
    record_traj: bool = False


def _resolve_output_dir(launch: LaunchArgs, scenario_id: int) -> tuple[str, str]:
    """Return ``(output_dir, exp_name)`` per existing logger conventions."""
    agent_stem = launch.agent_cfg.split(".")[0]
    if launch.mode == "train_scenario":
        return (
            osp.join("log", "train_scenario", agent_stem, f"scenario_{scenario_id}"),
            "exp",
        )

    # eval / replay / ori_eval -> different leaf folder for clarity
    if launch.ori_eval:
        leaf = "ori_eval"
    elif launch.replay:
        leaf = "replay"
    else:
        leaf = "eval"
    return osp.join("log", leaf, agent_stem), f"scenario_{scenario_id}"


def prepare_run_kwargs(launch: LaunchArgs) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Build ``(agent_config, scenario_config)`` ready to feed to a runner."""
    set_torch_variable(launch.device)
    torch.set_num_threads(launch.threads)
    set_seed(launch.seed)

    agent_path = osp.join(launch.ROOT_DIR, "safebench/agent/config", launch.agent_cfg)
    scenario_path = osp.join(
        launch.ROOT_DIR, "safebench/scenario/config", launch.scenario_cfg
    )
    agent_config = load_config(agent_path)
    scenario_config = load_config(scenario_path)
    scenario_config["scenario_id"] = launch.scenario_id

    output_dir, exp_name = _resolve_output_dir(launch, scenario_config["scenario_id"])

    overlay = dict(vars(launch))
    overlay["output_dir"] = output_dir
    overlay["exp_name"] = exp_name
    agent_config.update(overlay)
    scenario_config.update(overlay)
    if scenario_config["scenario_category"] == "scenic":
        scenario_config["num_scenario"] = 1
    return agent_config, scenario_config


def select_runner(scenario_config: Dict[str, Any]):
    """Pick CarlaRunner vs ScenicRunner based on ``scenario_category``."""
    from safebench.runner import CarlaRunner, ScenicRunner

    if scenario_config["scenario_category"] == "scenic":
        return ScenicRunner
    return CarlaRunner


def run_one(launch: LaunchArgs, *, runner_run_kwargs: Optional[Dict[str, Any]] = None) -> None:
    """End-to-end: build configs, instantiate the right runner, run, close."""
    agent_config, scenario_config = prepare_run_kwargs(launch)
    Runner = select_runner(scenario_config)
    runner = Runner(agent_config, scenario_config)
    try:
        runner.run(**(runner_run_kwargs or {}))
    finally:
        runner.close()
