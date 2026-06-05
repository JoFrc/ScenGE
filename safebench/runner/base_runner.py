"""Shared runner scaffolding for CARLA + SafeBench.

``BaseRunner`` collects the parts that ``CarlaRunner`` and ``ScenicRunner``
duplicate verbatim:

- CARLA client + env_params construction
- agent / scenario policy wiring through the registries
- pygame birdeye renderer setup
- the inner per-step eval/train loop helpers
- checkpoint / video / cleanup plumbing

Subclasses provide world initialization, data loading, and any loop layout
that is unique to their pipeline.
"""

from __future__ import annotations

import copy
import glob
import os

import carla
import numpy as np
import pygame

from safebench.agent import AGENT_POLICY_LIST
from safebench.scenario import SCENARIO_POLICY_LIST
from safebench.scenario.scenario_manager.carla_data_provider import CarlaDataProvider
from safebench.gym_carla.envs.render import BirdeyeRender
from safebench.util.logger import Logger, setup_logger_kwargs
from safebench.util.metric_util import get_route_scores


DEFAULT_ENV_PARAMS = {
    "warm_up_steps": 9,
    "display_size": 128,
    "obs_range": 32,
    "d_behind": 12,
    "max_past_step": 1,
    "discrete": False,
    "discrete_acc": [-3.0, 0.0, 3.0],
    "discrete_steer": [-0.2, 0.0, 0.2],
    "continuous_accel_range": [-3.0, 3.0],
    "continuous_steer_range": [-0.3, 0.3],
    "max_waypt": 12,
    # spatial resolution (meters per pixel) of birdeye / camera observation
    "pixel_bin": 0.125,
    "out_lane_thres": 4,
    "desired_speed": 8,
}


class BaseRunner:
    """Common functionality between Carla- and Scenic-driven runners."""

    def __init__(self, agent_config, scenario_config):
        self.scenario_config = scenario_config
        self.agent_config = agent_config

        # high-level flags
        self.seed = scenario_config["seed"]
        self.exp_name = scenario_config["exp_name"]
        self.output_dir = scenario_config["output_dir"]
        self.mode = scenario_config["mode"]
        self.save_video = scenario_config["save_video"]
        self.render = scenario_config["render"]
        self.num_scenario = scenario_config["num_scenario"]
        self.fixed_delta_seconds = scenario_config["fixed_delta_seconds"]
        self.scenario_category = scenario_config["scenario_category"]

        # CARLA client
        self.client = carla.Client("localhost", scenario_config["port"])
        self.client.set_timeout(10.0)
        self.world = None
        self.env = None
        self.display = None
        self.birdeye_render = None

        self.env_params = self._build_env_params()
        self._bind_agent_config()

        self.logger = self._build_logger()
        self._prepare_mode()
        self._announce()

        self.agent_policy = AGENT_POLICY_LIST[agent_config["policy_type"]](
            agent_config, logger=self.logger
        )
        self.scenario_policy = SCENARIO_POLICY_LIST[scenario_config["policy_type"]](
            scenario_config, logger=self.logger
        )

        if self.save_video:
            self._enable_video_recorder()

    # ---- init helpers --------------------------------------------------

    def _build_env_params(self):
        params = dict(DEFAULT_ENV_PARAMS)
        params.update(
            {
                "auto_ego": self.scenario_config["auto_ego"],
                "obs_type": self.agent_config["obs_type"],
                "scenario_category": self.scenario_category,
                "ROOT_DIR": self.scenario_config["ROOT_DIR"],
                "max_episode_step": self.scenario_config["max_episode_step"],
            }
        )
        return params

    def _bind_agent_config(self):
        self.agent_config["mode"] = self.scenario_config["mode"]
        self.agent_config["ego_action_dim"] = self.scenario_config["ego_action_dim"]
        self.agent_config["ego_state_dim"] = self.scenario_config["ego_state_dim"]
        self.agent_config["ego_action_limit"] = self.scenario_config["ego_action_limit"]

    def _build_logger(self):
        logger_kwargs = setup_logger_kwargs(
            self.exp_name,
            self.output_dir,
            self.seed,
            agent=self.agent_config["policy_type"],
            scenario=self.scenario_config["policy_type"],
            scenario_category=self.scenario_category,
        )
        return Logger(**logger_kwargs)

    def _prepare_mode(self):
        """Subclasses override to set up mode-specific bookkeeping."""
        if self.mode not in ("train_scenario", "eval"):
            raise NotImplementedError(f"Unsupported mode: {self.mode}.")
        self.save_freq = self._resolve_save_freq()

    def _resolve_save_freq(self):
        # Scenic configs put save_freq on agent_config; Carla configs on
        # scenario_config. Try both, preferring scenario_config.
        if "save_freq" in self.scenario_config:
            return self.scenario_config["save_freq"]
        return self.agent_config["save_freq"]

    def _announce(self):
        self.logger.log(">> Agent Policy: " + self.agent_config["policy_type"])
        self.logger.log(">> Scenario Policy: " + self.scenario_config["policy_type"])
        if self.scenario_config["auto_ego"]:
            self.logger.log(
                ">> Using auto-pilot for ego vehicle, action of policy will be ignored",
                "yellow",
            )
        self.logger.log(">> " + "-" * 40)

    def _enable_video_recorder(self):
        self.logger.init_video_recorder()

    # ---- world / renderer ----------------------------------------------

    def _apply_world_settings(self, world):
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.fixed_delta_seconds
        world.apply_settings(settings)
        CarlaDataProvider.set_client(self.client)
        CarlaDataProvider.set_world(world)
        CarlaDataProvider.set_traffic_manager_port(self.scenario_config["tm_port"])

    def _init_renderer(self, num_panels=2):
        """birdeye + camera = 2 panels (lidar removed)."""
        self.logger.log(">> Initializing pygame birdeye renderer")
        pygame.init()
        flag = pygame.HWSURFACE | pygame.DOUBLEBUF
        if not self.render:
            flag = flag | pygame.HIDDEN

        display_size = self.env_params["display_size"]
        window_size = (
            display_size * num_panels,
            display_size * self.num_scenario,
        )
        self.display = pygame.display.set_mode(window_size, flag)

        pixels_per_meter = display_size / self.env_params["obs_range"]
        pixels_ahead_vehicle = (
            self.env_params["obs_range"] / 2 - self.env_params["d_behind"]
        ) * pixels_per_meter
        birdeye_params = {
            "screen_size": [display_size, display_size],
            "pixels_per_meter": pixels_per_meter,
            "pixels_ahead_vehicle": pixels_ahead_vehicle,
        }
        self.birdeye_render = BirdeyeRender(
            self.world, birdeye_params, logger=self.logger
        )

    # ---- step loop helpers ---------------------------------------------

    def _grab_video_frame(self):
        if self.save_video:
            self.logger.add_frame(
                pygame.surfarray.array3d(self.display).transpose(1, 0, 2)
            )

    def _score_for(self, rewards, infos, idx):
        """Choose reward source per scenario_category."""
        if self.scenario_category in ("planning", "scenic"):
            return rewards[idx]
        return 1 - infos[idx]["iou_loss"]

    def _run_step_loop(
        self,
        obs,
        infos,
        deterministic,
        replay_buffer=None,
        score_list=None,
        episode_reward=None,
    ):
        """Drive one scenario from reset until ``all_scenario_done``."""
        while not self.env.all_scenario_done():
            ego_actions = self.agent_policy.get_action(
                obs, infos, deterministic=deterministic
            )
            scenario_actions = self.scenario_policy.get_action(
                obs, infos, deterministic=deterministic
            )
            next_obs, rewards, dones, infos = self.env.step(
                ego_actions=ego_actions, scenario_actions=scenario_actions
            )

            if replay_buffer is not None:
                replay_buffer.store(
                    [ego_actions, scenario_actions, obs, next_obs, rewards, dones],
                    additional_dict=infos,
                )

            if score_list is not None:
                for idx, info in enumerate(infos):
                    score_list[info["scenario_id"]].append(
                        self._score_for(rewards, infos, idx)
                    )

            if episode_reward is not None:
                episode_reward.append(np.mean(rewards))

            self._grab_video_frame()
            obs = copy.deepcopy(next_obs)
        return obs, infos

    # ---- ckpt / cleanup ------------------------------------------------

    def check_continue_training(self, policy, replay_buffer=None):
        policy.load_model(replay_buffer=replay_buffer)
        if policy.continue_episode == 0:
            self.logger.log(">> Previous checkpoint not found. Training from scratch.")
            return 0
        self.logger.log(
            f">> Continue training from previous checkpoint, epoch: {policy.continue_episode}."
        )
        return policy.continue_episode

    def clean_cache(self, path):
        all_files = glob.glob(os.path.join(path, "*"))
        file_to_keep = os.path.join(path, "model.sac.-001.torch")
        for file in all_files:
            if file != file_to_keep:
                os.remove(file)

    def close(self):
        pygame.quit()
        if self.env is not None:
            try:
                self.env.clean_up()
            except Exception:
                pass

    # ---- to be implemented by subclasses -------------------------------

    def run(self, *args, **kwargs):
        raise NotImplementedError

    def eval(self, data_loader, *args, **kwargs):
        raise NotImplementedError

    # NOTE: ``get_route_scores`` is re-exported here so subclasses don't
    # need to import it independently when assembling logger entries.
    @staticmethod
    def _route_scores(records):
        return get_route_scores(records)
