import json
import os
import time
from typing import Any, List, Tuple

import gym
import numpy as np
import torch
from common import ENV_SEEDS, EXPERIMENT_SEEDS, MEASURES, load_lunar_lander_model
from mdpfuzz.executor import Executor
from mdpfuzz.mdpfuzz import Fuzzer

from framework import Framework

DEFAULT_MIN = -1000
DEFAULT_MAX = 1000


class LLFramework(Framework):
    def __init__(self, rand_seed, cell_granularity, features, descriptors, **kwargs):
        super().__init__(rand_seed, cell_granularity, features, descriptors, **kwargs)

        self.action_range = [0, 4]  # type: Tuple[int, int]
        self.action_bins = 4  # type: int
        self.path_to_measures_extrema = "grid/ll/measures.csv"  # type: str
        self.use_case = "Lunar Lander"

    def generate_input(self, **kwargs):
        return self.rng.uniform(low=DEFAULT_MIN, high=DEFAULT_MAX, size=2)

    def generate_inputs(self, n, **kwargs):
        return self.rng.uniform(low=DEFAULT_MIN, high=DEFAULT_MAX, size=(n, 2))

    def mutate(self, input, **kwargs):
        return np.clip(
            self.rng.normal(input, 5.0),
            [DEFAULT_MIN, DEFAULT_MIN],
            [DEFAULT_MAX, DEFAULT_MAX],
        )

    def execute_policy(self, input, model, env_seed, deterministic=True, render=False):
        t0 = time.time()
        env: gym.Env = gym.make("LunarLander-v3")
        env.seed(env_seed)
        obs = env.reset(input)
        state = None
        acc_reward = 0.0

        actions = []

        impact_x_pos = None
        impact_y_vel = None
        all_y_vels = []
        frames = []
        if render:
            frames.append(env.render("rgb_array"))

        for _ in range(1000):
            action, state = model.predict(obs, state=state, deterministic=deterministic)
            obs, reward, done, info = env.step(action)
            if render:
                frames.append(env.render("rgb_array"))
            acc_reward += reward

            actions.append(action)
            x_pos = obs[0]
            y_vel = obs[3]
            leg0_touch = bool(obs[6])
            leg1_touch = bool(obs[7])
            all_y_vels.append(y_vel)
            if impact_x_pos is None and (leg0_touch or leg1_touch):
                impact_x_pos = x_pos
                impact_y_vel = y_vel

            if done:
                break

        if impact_x_pos is None:
            impact_x_pos = x_pos
            impact_y_vel = min(all_y_vels)
        behavior = np.array([impact_x_pos, impact_y_vel])
        env.close()
        exec_time = time.time() - t0
        return acc_reward, (reward == -100), behavior, obs, exec_time, np.expand_dims(actions, axis=1), frames


class LLExecutor(Executor):

    def __init__(self, rand_seed: int, env_seeds: List[int], log_path: str) -> None:
        super().__init__(sim_steps=0, env_seed=0)
        self.executor = LLFramework(
            rand_seed=rand_seed,
            cell_granularity=50,
            features=MEASURES,
            descriptors=[],
            name="MDPFuzz"
        )

        self.env_seeds = env_seeds

        self.creation_time = time.time()
        if os.path.isdir(log_path):
            filepath = (
                f"{log_path}{self.creation_time}"
                if log_path.endswith("/")
                else f"{log_path}/{self.creation_time}"
            )
        else:
            filepath = log_path

        self.fp = filepath
        self.behaviors_buffer = open(f"{self.fp}_behaviors.txt", "w", buffering=1)
        self.inputs_buffer = open(f"{self.fp}_inputs.txt", "w", buffering=1)
        self.logs_buffer = open(f"{self.fp}_logs.txt", "w", buffering=1)
        self.final_states_buffers = [
            open(f"{self.fp}_final_states_{seed}.txt", "w", buffering=1)
            for seed in self.env_seeds
        ]
        self.expert_behaviors_buffers = [
            open(f"{self.fp}_expert_behaviors_{seed}.txt", "w", buffering=1)
            for seed in self.env_seeds
        ]

        self.features = MEASURES
        self.executor.config["env_seeds"] = self.env_seeds

    def generate_input(self, rng: np.random.Generator) -> np.ndarray:
        return self.executor.generate_input()

    def generate_inputs(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return self.executor.generate_inputs(n)

    def mutate(
        self, input: np.ndarray, rng: np.random.Generator, **kwargs
    ) -> np.ndarray:
        return self.executor.mutate(input)

    def load_policy(self, **kwargs):
        return None

    def log_execution(
        self,
        input: np.ndarray,
        mean_acc_reward: float,
        failure_prob: float,
        final_obs_list: List[np.ndarray],
        expert_behaviors_list: List[np.ndarray],
        behavior: np.ndarray,
        exec_time: float,
    ):
        np.savetxt(self.inputs_buffer, input.reshape(1, -1), delimiter=",")
        np.savetxt(self.behaviors_buffer, behavior.reshape(1, -1), delimiter=",")
        for buffer, fs in zip(self.final_states_buffers, final_obs_list):
            np.savetxt(buffer, fs.reshape(1, -1), delimiter=",")
        for eb_buffer, eb in zip(self.expert_behaviors_buffers, expert_behaviors_list):
            np.savetxt(eb_buffer, eb.reshape(1, -1), delimiter=",")
        print(
            f"episode_reward: {mean_acc_reward}, failure_prob: {failure_prob}, execution_time: {exec_time}",
            file=self.logs_buffer,
        )

    def execute_policy(
        self, input: np.ndarray, policy: Any
    ) -> Tuple[float, bool, np.ndarray, float]:
        t0 = time.time()
        mean_acc_reward, failure_prob, final_obs_list, behaviors_list, measures = (
            self.executor.execute_stochastic_policy(input, policy, self.env_seeds)
        )

        exec_time = time.time() - t0
        behavior = np.array(list(measures.values()))
        self.log_execution(
            input,
            mean_acc_reward,
            failure_prob,
            final_obs_list,
            behaviors_list,
            behavior,
            exec_time,
        )

        return (
            mean_acc_reward,
            bool(failure_prob),
            [],
            exec_time,
        )

    def clean(self):
        """Closes the file buffers and saves the executor."""
        self.behaviors_buffer.close()
        self.inputs_buffer.close()
        self.logs_buffer.close()
        for buffer in self.final_states_buffers:
            buffer.close()
        self.executor.save_state(self.fp)
        # creates also this empty file 9not done by the executor) for result data structure consistency...
        with open(f"{self.fp}_cells.txt", "w") as f:
            f.write("")


if __name__ == "__main__":
    from pathlib import Path

    torch.set_num_threads(1)
    main_seed = 2021
    model = load_lunar_lander_model()

    # expert and generic measures
    features = MEASURES

    # experimental parameters
    test_budget = 100
    init_budget = 10
    cell_granularity = 50

    # population_size, nb_iterations = 100, 50
    population_size, nb_iterations = 50, 2
    k = 3
    novelty_threshold = 0.005

    descriptors = ["action_entropy", "length_spread"]

    results_fp = Path("results_test/ll")
    results_fp.mkdir(parents=True, exist_ok=True)
    (results_fp / "qd").mkdir(parents=True, exist_ok=True)
    (results_fp / "ns").mkdir(parents=True, exist_ok=True)
    (results_fp / "rt").mkdir(parents=True, exist_ok=True)
    (results_fp / "mdpfuzz").mkdir(parents=True, exist_ok=True)

    for seed in EXPERIMENT_SEEDS[:1]:
        print(f"Seed {seed} starts.")

        f = LLFramework(
            seed,
            cell_granularity,
            features=features,
            descriptors=descriptors,
        )
        f.random_testing(model, ENV_SEEDS, test_budget, str(results_fp / "rt"))

        f = LLFramework(
            seed,
            cell_granularity,
            features=features,
            descriptors=descriptors,
        )
        f.test_policy(
            model, ENV_SEEDS, test_budget, init_budget, str(results_fp / "qd")
        )

        f = LLFramework(
            seed,
            cell_granularity,
            features=features,
            descriptors=descriptors,
        )
        f.novelty_search(
            model,
            ENV_SEEDS,
            population_size,
            nb_iterations,
            k,
            novelty_threshold,
            str(results_fp / "ns"),
        )

        executor = LLExecutor(seed, ENV_SEEDS, log_path=str(results_fp / "mdpfuzz"))
        fuzzer_logs_path = executor.fp + "_fuzzer"
        fuzzer = Fuzzer(random_seed=seed, executor=executor, k=4, tau=0.1, gamma=0.01)
        fuzzer.fuzzing_no_coverage(
            n=init_budget,
            test_budget=test_budget,  # 2*n will be removed since we assume that test_budget is the TOTAL budget
            policy=model,
            saving_path=fuzzer_logs_path,
            local_sensitivity=True,  # don"t re-run for computing the sensitivity
            exp_name="Lunar Lander",
            light_pool=True,  # don"t log the inputs
            save_logs_only=True,  # don"t save evaluated inputs
        )
        executor.clean()
