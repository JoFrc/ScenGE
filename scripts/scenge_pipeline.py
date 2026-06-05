"""Single-command orchestrator for the full ScenGE pipeline.

Steps (each can be skipped via flags):

1. ``msgen``      -- LLM + RAG generation of Scenic scripts (offline; no
   CARLA needed). Writes to ``scenic_data/scenario_{id}/...``.
2. ``train``      -- Run ``scripts/train_scenario.py`` to pick the
   top-N most threatening scenes per route.
3. ``select``     -- Run ``scripts/select_traj.py`` to attention-pick
   key segments and perturb env trajectories. Expects
   ``--folder`` to contain ``traj_dir/*.json``.
4. ``eval``       -- Run ``scripts/eval.py`` (with ``--replay``) on
   the perturbed scenes for each ego policy.

Each step shells out via ``subprocess.run``; nothing is imported in-
process, so a failure in step 2 does not leak CARLA / pygame state into
step 3, etc.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import Iterable, Sequence


REPO_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")


def _python_argv(*parts: str) -> list[str]:
    return [sys.executable, *parts]


def _run(cmd: Sequence[str]) -> None:
    print(">>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def step_msgen(args: argparse.Namespace) -> None:
    cmd = _python_argv(
        os.path.join(REPO_ROOT, "safebench/scenge/msgen/meta_scenario_generation.py"),
        "--scenario_ids",
        str(args.scenario_id),
        "--llm-backend",
        args.llm_backend,
        "--llm_name",
        args.llm_name,
        "--scenario-route-id",
        str(args.scenario_route_id),
        "--max-repair-rounds",
        str(args.max_repair_rounds),
    )
    if args.embed_backend:
        cmd.extend(["--embed-backend", args.embed_backend])
    if args.embed_name:
        cmd.extend(["--embed_name", args.embed_name])
    if args.openai_api_key:
        cmd.extend(["--openai-api-key", args.openai_api_key])
    if args.openai_base_url:
        cmd.extend(["--openai-base-url", args.openai_base_url])
    if args.openai_temperature is not None:
        cmd.extend(["--openai-temperature", str(args.openai_temperature)])
    if args.openai_max_tokens is not None:
        cmd.extend(["--openai-max-tokens", str(args.openai_max_tokens)])
    if args.route_pickle:
        cmd.extend(["--route-pickle", args.route_pickle])
    if args.no_scenic_sample:
        cmd.append("--no-scenic-sample")
    _run(cmd)


def step_train(args: argparse.Namespace) -> None:
    _run(
        _python_argv(
            os.path.join(SCRIPTS_DIR, "train_scenario.py"),
            "--agent_cfg",
            f"{args.surrogate_policy}.yaml",
            "--scenario_cfg",
            "train_scenario_scenic.yaml",
            "--scenario_id",
            str(args.scenario_id),
            "--port",
            str(args.port),
            "--tm_port",
            str(args.tm_port),
            "--traj_root",
            args.traj_root,
            "--record_traj",
        )
    )


def step_select(args: argparse.Namespace) -> None:
    _run(
        _python_argv(
            os.path.join(SCRIPTS_DIR, "select_traj.py"),
            "--folder",
            args.traj_root,
        )
    )


def step_eval(args: argparse.Namespace, ego_policies: Iterable[str]) -> None:
    for ego in ego_policies:
        cmd = _python_argv(
            os.path.join(SCRIPTS_DIR, "eval.py"),
            "--agent_cfg",
            f"{ego}.yaml",
            "--scenario_cfg",
            "eval_scenic.yaml",
            "--scenario_id",
            str(args.scenario_id),
            "--port",
            str(args.port),
            "--tm_port",
            str(args.tm_port),
            "--replay",
            "--traj_root",
            args.traj_root,
        )
        _run(cmd)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario_id", type=int, required=True)
    parser.add_argument("--port", type=int, default=2002)
    parser.add_argument("--tm_port", type=int, default=8002)

    parser.add_argument(
        "--ego_policies",
        nargs="*",
        default=["sac", "ppo", "td3"],
        help="Which ego policies to evaluate in the final replay step.",
    )
    parser.add_argument(
        "--surrogate_policy",
        type=str,
        default="sac",
        help="Surrogate policy used during the train_scenario step.",
    )

    # msgen knobs
    parser.add_argument(
        "--llm_backend",
        type=str,
        choices=("huggingface", "hf", "openai"),
        default="huggingface",
        help="msgen LLM provider: local HuggingFace or OpenAI-compatible API.",
    )
    parser.add_argument("--llm_name", type=str, default="gpt-4o-mini")
    parser.add_argument(
        "--embed_backend",
        type=str,
        choices=("huggingface", "hf", "openai"),
        default=None,
        help="RAG embedding provider. Defaults to openai when --llm_backend=openai.",
    )
    parser.add_argument(
        "--embed_name",
        type=str,
        default=None,
        help="Embedding model id. OpenAI default: text-embedding-3-small.",
    )
    parser.add_argument(
        "--openai_api_key",
        type=str,
        default=None,
        help="OpenAI API key (fallback: env OPENAI_API_KEY).",
    )
    parser.add_argument(
        "--openai_base_url",
        type=str,
        default=None,
        help="OpenAI-compatible base URL (fallback: env OPENAI_BASE_URL).",
    )
    parser.add_argument("--openai_temperature", type=float, default=0.1)
    parser.add_argument("--openai_max_tokens", type=int, default=None)
    parser.add_argument("--scenario_route_id", type=int, default=1)
    parser.add_argument("--route_pickle", type=str, default=None)
    parser.add_argument("--max_repair_rounds", type=int, default=4)
    parser.add_argument("--no_scenic_sample", action="store_true")

    parser.add_argument(
        "--traj_root",
        type=str,
        default=None,
        help="Root folder for select_traj step. Must contain traj_dir/*.json. "
        "Defaults to log/scenge/scenario_{id}.",
    )

    parser.add_argument("--skip_msgen", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_select", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.traj_root is None:
        args.traj_root = os.path.join(
            REPO_ROOT, "log", "scenge", f"scenario_{args.scenario_id}"
        )

    if not args.skip_msgen:
        step_msgen(args)
    if not args.skip_train:
        step_train(args)
    if not args.skip_select:
        step_select(args)
    if not args.skip_eval:
        step_eval(args, args.ego_policies)


if __name__ == "__main__":
    main()
