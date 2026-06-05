"""CarlaRunner: drives the four traditional baselines (LC / CS / AdvSim /
AdvTraj). Worlds are loaded per-map via ``client.load_world(town)``.
"""

from __future__ import annotations

import carla
import numpy as np
from tqdm import tqdm

from safebench.gym_carla.env_wrapper import VectorWrapper
from safebench.runner.base_runner import BaseRunner
from safebench.scenario.scenario_data_loader import ScenarioDataLoader
from safebench.scenario.tools.scenario_utils import scenario_parse


class CarlaRunner(BaseRunner):
    """Standard CARLA-driven runner."""

    def _prepare_mode(self):
        if self.mode == "train_scenario":
            self.buffer_capacity = self.scenario_config["buffer_capacity"]
            self.eval_in_train_freq = self.scenario_config["eval_in_train_freq"]
            self.save_freq = self.scenario_config["save_freq"]
            self.train_episode = self.scenario_config["train_episode"]
            self.logger.save_config(self.scenario_config)
            self.logger.create_training_dir()
        elif self.mode == "eval":
            self.save_freq = self.scenario_config["save_freq"]
            self.logger.log(">> Evaluation Mode, skip config saving", "yellow")
            self.logger.create_eval_dir(load_existing_results=True)
        else:
            raise NotImplementedError(f"Unsupported mode: {self.mode}.")

    def _enable_video_recorder(self):
        assert self.mode == "eval", "only allow video saving in eval mode"
        super()._enable_video_recorder()

    # ---- world ---------------------------------------------------------

    def _init_world(self, town):
        self.logger.log(f">> Initializing carla world: {town}")
        self.world = self.client.load_world(town)
        self._apply_world_settings(self.world)
        self.world.set_weather(carla.WeatherParameters.ClearNoon)

    # ---- train ---------------------------------------------------------

    def train(self, data_loader, start_episode=0, replay_buffer=None):
        for _ in tqdm(range(len(data_loader))):
            self.current_episode += 1
            if self.current_episode >= self.train_episode:
                return
            if self.current_episode < start_episode:
                continue

            sampled_scenario_configs, _ = data_loader.sampler()

            static_obs = self.env.get_static_obs(sampled_scenario_configs)
            self.scenario_policy.load_model(sampled_scenario_configs)
            scenario_init_action, additional_dict = (
                self.scenario_policy.get_init_action(static_obs)
            )
            try:
                obs, infos = self.env.reset(
                    sampled_scenario_configs, scenario_init_action
                )
            except Exception:
                continue
            replay_buffer.store_init(
                [static_obs, scenario_init_action], additional_dict=additional_dict
            )

            self.agent_policy.set_ego_and_route(self.env.get_ego_vehicles(), infos)

            episode_reward = []
            self._run_step_loop(
                obs,
                infos,
                deterministic=False,
                replay_buffer=replay_buffer,
                episode_reward=episode_reward,
            )

            if (
                self.mode == "train_scenario"
                and self.scenario_policy.type == "offpolicy"
            ):
                self.scenario_policy.train(replay_buffer)

            all_scores = self._route_scores(self.env.running_results)

            self.env.clean_up()
            replay_buffer.finish_one_episode()
            self.logger.add_training_results("episode", self.current_episode)
            self.logger.add_training_results("episode_reward", np.sum(episode_reward))
            for key, value in all_scores.items():
                self.logger.add_training_results(key, value)
            critic_loss, actor_loss = 0, 0
            self.logger.add_training_results("critic_loss", critic_loss)
            self.logger.add_training_results("actor_loss", actor_loss)
            self.logger.log(
                f">> Episode: {self.current_episode}, "
                f"#buffer_len: {replay_buffer.buffer_len}, "
                f"critic: {critic_loss:.3f}, actor: {actor_loss:.3f}"
            )
            self.logger.save_training_results()

            if self.mode == "train_scenario" and self.scenario_policy.type in (
                "init_state",
                "onpolicy",
            ):
                self.scenario_policy.train(replay_buffer)

            if (self.current_episode + 1) % self.save_freq == 0:
                if self.mode == "train_scenario":
                    self.scenario_policy.save_model(self.current_episode)

    # ---- eval ----------------------------------------------------------

    def eval(self, data_loader):
        num_finished_scenario = 0
        data_loader.reset_idx_counter()

        log_name = ""
        while len(data_loader) > 0:
            sampled_scenario_configs, num_sampled_scenario = data_loader.sampler()
            log_name = f"ROUTE-{sampled_scenario_configs[0].route_id - 4}"
            num_finished_scenario += num_sampled_scenario

            data_ids = [config.data_id for config in sampled_scenario_configs]

            static_obs = self.env.get_static_obs(sampled_scenario_configs)
            self.scenario_policy.load_model(sampled_scenario_configs)
            scenario_init_action, _ = self.scenario_policy.get_init_action(
                static_obs, deterministic=True
            )
            try:
                obs, infos = self.env.reset(
                    sampled_scenario_configs, scenario_init_action
                )
            except Exception:
                continue

            self.agent_policy.set_ego_and_route(self.env.get_ego_vehicles(), infos)

            score_list = {s_i: [] for s_i in range(num_sampled_scenario)}
            self._run_step_loop(
                obs,
                infos,
                deterministic=True,
                score_list=score_list,
            )

            if self.save_video:
                self.logger.save_video(data_ids=data_ids, log_name=log_name)

            self.logger.log(">> All scenarios are completed. Cleaning up all actors")
            self.env.clean_up()

            self.logger.log(
                f"[{num_finished_scenario}/{data_loader.num_total_scenario}] "
                "Ranking scores for batch scenario:",
                "yellow",
            )
            for s_i in score_list:
                self.logger.log(
                    "\t Env id " + str(s_i) + ": " + str(np.mean(score_list[s_i])),
                    "yellow",
                )

            all_running_results = self.logger.add_eval_results(
                records=self.env.running_results
            )
            all_scores = self._route_scores(all_running_results)
            self.logger.add_eval_results(scores=all_scores)
            self.logger.print_eval_results()
            if len(self.env.running_results) % self.save_freq == 0:
                self.logger.save_eval_results(log_name)
        if log_name:
            self.logger.save_eval_results(log_name)

    # ---- top-level orchestration ---------------------------------------

    def run(self, test_epoch=None):
        config_by_map = scenario_parse(self.scenario_config, self.logger)
        map_keys = list(config_by_map.keys())

        self.agent_policy.load_model(episode=test_epoch)

        for m_i in map_keys:
            if self.mode == "eval":
                log_name = f"ROUTE-{config_by_map[m_i][0].route_id - 4}"
                if self.logger.check_eval_dir(log_name) == len(config_by_map[m_i]):
                    self.logger.log(">> This scenario and route have been done.")
                    continue

            try:
                self._init_world(m_i)
                self._init_renderer(num_panels=2)
            except Exception:
                continue

            self.env = VectorWrapper(
                self.env_params,
                self.scenario_config,
                self.world,
                self.birdeye_render,
                self.display,
                self.logger,
            )

            data_loader = ScenarioDataLoader(
                config_by_map[m_i], self.num_scenario, m_i, self.world
            )

            if self.mode == "eval":
                self.agent_policy.set_mode("eval")
                self.scenario_policy.set_mode("eval")
                self.eval(data_loader)
            elif self.mode == "train_scenario":
                start_episode = self.check_continue_training(self.scenario_policy)
                self.agent_policy.load_model()
                self.agent_policy.set_mode("eval")
                self.scenario_policy.set_mode("train")
                self.current_episode = start_episode
                self.train(data_loader, start_episode)
            else:
                raise NotImplementedError(f"Unsupported mode: {self.mode}.")
