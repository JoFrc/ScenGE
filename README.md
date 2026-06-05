# ScenGE

> **[AAAI 2026]** Adversarial Generation and Collaborative Evolution of Safety-Critical Scenarios for Autonomous Vehicles

Official implementation of **ScenGE**, a framework for generating and evolving safety-critical driving scenarios in [CARLA](https://carla.org/) on top of the [SafeBench](https://github.com/trust-ai/SafeBench) benchmark. ScenGE combines LLM-driven scenario synthesis, Scenic-based adversarial scene selection, and gradient-based trajectory perturbation to produce challenging test cases for reinforcement-learning ego policies.

---

## Overview

Autonomous vehicles must be validated under rare but dangerous traffic situations. ScenGE addresses this with a four-stage pipeline:

1. **Meta scenario generation (msgen)** — LLM + RAG produces executable Scenic scripts from scenario semantics and code snippets.
2. **Adversarial scene selection** — Monte Carlo simulation ranks generated scenes and keeps the most threatening ones per route.
3. **Collaborative trajectory evolution** — Attention-guided segment selection and PGD refine environment-actor trajectories.
4. **Replay evaluation** — Perturbed trajectories are replayed against multiple ego policies (SAC, PPO, TD3) for standardized comparison.

This repository also re-implements the baselines used in our paper: four traditional SafeBench methods (LC, CS, AdvSim, AdvTraj) and the ChatScene / Scenic evaluation path.

| Family                | Methods                                 | Runner                              |
| --------------------- | --------------------------------------- | ----------------------------------- |
| Traditional baselines | LC, CS, AdvSim, AdvTraj                 | `CarlaRunner`                       |
| Scenic / ChatScene    | `train_scenario_scenic` + `eval_scenic` | `ScenicRunner`                      |
| **ScenGE (ours)**     | msgen → train → select → replay-eval    | `ScenicRunner` + `safebench.scenge` |

**Ego policies under test:** SAC, PPO, and TD3 — all dispatched through `RLAgent` and configured via per-policy YAML files (`policy_name`, `load_dir`).

---

## Installation

### Prerequisites

- Linux with an NVIDIA GPU (CUDA 12.1 recommended)
- [CARLA 0.9.13](https://drive.google.com/file/d/139vLRgXP90Zk6Q_du9cRdOLx7GJIw_0v/view?usp=sharing) — download and extract locally; this repo handles only the Python side

### Environment setup

```bash
conda env create -f environment.yaml
conda activate scenge-env

# Core install (baselines + Scenic / ChatScene eval):
pip install -e .

# LLM-driven msgen step (transformers, llama-index, …):
pip install -e .[msgen]

# Scenic (bundled submodule):
cd Scenic && pip install . && cd -
```

### Start CARLA

Launch the simulator on the ports you will pass to `--port` / `--tm_port`:

```bash
./CarlaUE4.sh -prefernvidia -RenderOffScreen -carla-port=2000
```

For msgen-only runs (step 1 of the pipeline), CARLA is **not** required.

---

## Quick start

### ScenGE end-to-end

```bash
python scripts/scenge_pipeline.py --scenario_id 1 --port 2002 --tm_port 8002
```

Each stage runs in its own subprocess so a CARLA or pygame failure in one step does not corrupt the next:

| Step | Script                                               | Description                                                  |
| ---- | ---------------------------------------------------- | ------------------------------------------------------------ |
| 1    | `safebench/scenge/msgen/meta_scenario_generation.py` | RAG-based generation of Scenic scripts → `scenic_data/scenario_{id}/` |
| 2    | `scripts/train_scenario.py`                          | Select top-N most threatening scenes per route               |
| 3    | `scripts/train_scenario.py --record_traj`            | Dump per-scene actor trajectories to `{traj_root}/traj_dir/*.json` |
| 4    | `scripts/select_traj.py`                             | Attention + PGD on trajectory segments → `{traj_root}/perturbed_traj/` |
| 5    | `scripts/eval.py --replay`                           | Replay perturbed trajectories against SAC / PPO / TD3        |

Useful flags:

```bash
# Skip msgen if Scenic scripts already exist
python scripts/scenge_pipeline.py --scenario_id 1 --skip_msgen --port 2002 --tm_port 8002

# msgen with OpenAI-compatible API
export OPENAI_API_KEY=sk-...
python scripts/scenge_pipeline.py --scenario_id 1 \
  --llm_backend openai --llm_name gpt-4o-mini \
  --skip_train --skip_select --skip_eval
```

See [`safebench/scenge/msgen/README.md`](safebench/scenge/msgen/README.md) for msgen-specific options.

### Traditional baselines

`scripts/eval.py` selects `CarlaRunner` or `ScenicRunner` automatically from the scenario YAML's `scenario_category`.

```bash
# Learning-based (LC)
python scripts/eval.py --agent_cfg sac.yaml --scenario_cfg lc.yaml \
    --scenario_id 1 --port 2002 --tm_port 8002

# Critical-scenario search (CS)
python scripts/eval.py --agent_cfg sac.yaml --scenario_cfg cs.yaml \
    --scenario_id 1 --port 2002 --tm_port 8002

# AdvSim
python scripts/eval.py --agent_cfg sac.yaml --scenario_cfg advsim.yaml \
    --scenario_id 1 --port 2002 --tm_port 8002

# AdvTraj
python scripts/eval.py --agent_cfg sac.yaml --scenario_cfg advtraj.yaml \
    --scenario_id 1 --port 2002 --tm_port 8002
```

Swap `--agent_cfg` to `ppo.yaml` or `td3.yaml` to evaluate other ego policies.

### ChatScene / Scenic path

```bash
# Train: 50 simulations → top-2 scenes per route (saved under scenic_data/scenario_1/)
python scripts/train_scenario.py --agent_cfg sac.yaml \
    --scenario_cfg train_scenario_scenic.yaml --scenario_id 1 \
    --port 2002 --tm_port 8002

# Evaluate persisted top scenes across all three ego policies
for ego in sac ppo td3; do
  python scripts/eval.py --agent_cfg ${ego}.yaml \
      --scenario_cfg eval_scenic.yaml --scenario_id 1 \
      --port 2002 --tm_port 8002
done
```

---

## Pre-trained ego policies

Default checkpoints (Safe RL agents) are shipped under:

```
safebench/agent/model_ckpt/safe_rl/sac_0/model_save/model.pt
safebench/agent/model_ckpt/safe_rl/ppo_0/model_save/model.pt
safebench/agent/model_ckpt/safe_rl/td3_0/model_save/model.pt
```

Each ego YAML (`safebench/agent/config/{sac,ppo,td3}.yaml`) sets `load_dir` and `pretrain_dir` to these paths. Point `load_dir` at your own weights or at a directory containing `model.<policy>.<epoch>.torch` files to load a specific `--test_epoch`.

---

## Metrics

Aggregate Scenic evaluation outputs (`.pkl` files under `log/eval/`) with:

```bash
python scripts/calc_metrics.py --folder log/eval/sac/scenario_1 \
    --behaviors 1-5 --routes 0,1,8,9
```

Defaults cover behaviors 1–5 and routes 0–9; narrow with `--behaviors` and `--routes`.

To inspect a result file interactively:

```python
import joblib
records = joblib.load(
    "log/eval/sac/scenario_1/scenario_1_rl_lc_seed_19980321/eval_results/ROUTE-0_results.pkl"
)
records.keys()
# dict_keys(['collision_rate', 'avg_red_light_freq', ..., 'final_score'])
```

---

## Project structure

```
safebench/
  agent/
    safe_rl/                       # RL ego implementation (SAC / PPO / TD3)
    config/{sac,ppo,td3}.yaml
    model_ckpt/safe_rl/{sac_0,ppo_0,td3_0}/
  scenario/
    scenario_definition/           # LC, CS, AdvSim, AdvTraj, Scenic, …
    scenario_policy/
    config/{lc,cs,advsim,advtraj,train_scenario_scenic,eval_scenic}.yaml
    scenario_data/scenic_data/     # generated / selected Scenic scripts
  gym_carla/                       # CarlaEnv + VectorWrapper
  runner/                          # BaseRunner, CarlaRunner, ScenicRunner
  scenge/                          # ScenGE-specific logic
    msgen/                         # LLM + RAG → Scenic scripts
      knowledge_docs/              # phase-1 RAG: scenario semantics
      scenic_docs/                 # phase-2 RAG: Scenic snippets
      prompts/
    threat/                        # attention, PGD, threat losses
  util/                            # logger, launch, metrics, …
scripts/
  train_scenario.py                # Scenic adversarial scene selection
  eval.py                          # unified eval entry point
  scenge_pipeline.py               # end-to-end orchestration
  select_traj.py                   # trajectory selection + perturbation
  calc_metrics.py                  # post-eval metric aggregation
Scenic/                            # Scenic language (vendored)
```

Key entry points for developers:

- `safebench/runner/base_runner.py` — shared evaluation loop
- `safebench/runner/{carla,scenic}_runner.py` — runner-specific logic
- `safebench/scenge/threat/losses.py` — configurable `ThreatLoss`
- `safebench/util/launch.py` — CLI → `LaunchArgs` → runner hand-off

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{liu2026scenge,
  title={Adversarial generation and collaborative evolution of safety-critical scenarios for autonomous vehicles},
  author={Liu, Jiangfan and Guo, Yongkang and Zhong, Fangzhi and Zhang, Tianyuan and Jing, Zonglei and Liang, Siyuan and Wang, Jiakai and Zhang, Mingchuan and Liu, Aishan and Liu, Xianglong},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={45},
  pages={38926--38934},
  year={2026}
}
```

---

## Acknowledgments

This project builds on and extends several open-source efforts:

- **[SafeBench](https://github.com/trust-ai/SafeBench)** — benchmarking platform, scenario definitions, RL agents, and evaluation infrastructure. We thank the SafeBench team for their foundational work on safety-critical scenario evaluation in CARLA.
- **[ChatScene](https://github.com/javyduck/ChatScene)** — LLM + RAG pipeline for Scenic scenario generation. Our msgen module and Scenic evaluation path are inspired by and adapted from ChatScene; we are grateful to the authors for releasing their code and knowledge bases.
- **[Scenic](https://github.com/BerkeleyLearnVerify/Scenic)** — probabilistic scenario programming language used throughout this repository.

---

## License

This project is released under the [MIT License](LICENSE).