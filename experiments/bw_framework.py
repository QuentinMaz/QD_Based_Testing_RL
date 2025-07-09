import json
import os
import time
from typing import Any, List, Tuple

import gym
import numpy as np
import torch
from common import ENV_SEEDS, EXPERIMENT_SEEDS, MEASURES, load_bipedal_walker_model
from mdpfuzz.executor import Executor
from mdpfuzz.mdpfuzz import Fuzzer

from framework import Framework

FEATURES = [
    "meanDistance",
    "meanHeadStability",
    "meanTorquePerStep",
    "meanJump",
    "meanLeg0HipAngle",
    "meanLeg0HipSpeed",
    "meanLeg0KneeAngle",
    "meanLeg0KneeSpeed",
    "meanLeg1HipAngle",
    "meanLeg1HipSpeed",
    "meanLeg1KneeAngle",
    "meanLeg1KneeSpeed",
]
MIN_INPUT = np.array([1 for _ in range(15)])
MAX_INPUT = np.array([3 for _ in range(15)])
MAX_DIST_INPUT: np.ndarray = np.linalg.norm(MAX_INPUT - MIN_INPUT)
AVG_SIZE = 30
EXPERT_INDICES = [[0, 1], [2, 3], [4, 8], [5, 11]]
EXPERT_PLOT_ARGS = [
    {
        "xlabel": "distance to the goal",
        "ylabel": "hull angle",
        "title": "Distance vs Hull angle",
    },
    {"xlabel": "torque (actions)", "ylabel": "jump rate", "title": "Torque vs Jump"},
    {"xlabel": "1st leg", "ylabel": "2nd leg", "title": "Hip angles"},
    {"xlabel": "1st leg", "ylabel": "2nd leg", "title": "Hip speeds"},
]


class BWFramework(Framework):
    def __init__(self, rand_seed, cell_granularity, features, descriptors, **kwargs):
        super().__init__(rand_seed, cell_granularity, features, descriptors, **kwargs)

        self.action_range = [-1, 1]  # type: Tuple[int, int]
        self.action_bins = 8  # type: int
        self.path_to_measures_extrema = "grid/bw/measures.csv"  # type: str
        self.use_case = "Bipedal Walker"
        self.input_fmt = "%1.0f"

    def generate_input(self, **kwargs):
        return self.rng.integers(low=1, high=4, size=15)

    def generate_inputs(self, n, **kwargs):
        return self.rng.integers(low=1, high=4, size=(n, 15))

    def mutate(self, input, **kwargs):
        mutation = self.rng.choice(2, 15, p=[0.9, 0.1])
        if np.sum(mutation) == 0:
            mutation[0] = 1
        mutated_input = input + mutation
        mutated_input = np.remainder(mutated_input, 4)
        mutated_input = np.clip(mutated_input, 1, 3)
        return mutated_input

    def execute_policy(self, input, model, env_seed, deterministic=True, render=False):
        env = gym.make("BipedalWalkerHardcore-v4", rand_seed=env_seed)

        acc_reward = 0.0
        features = np.zeros(12)

        obs = env.reset(input)
        state = None
        t0 = time.time()

        action_seq = []
        frames = []
        if render:
            frames.append(env.render("rgb_array"))
        for t in range(300):
            action, state = model.predict(obs, state=state, deterministic=deterministic)
            obs, reward, done, info = env.step(action)
            if render:
                frames.append(env.render("rgb_array"))
            action_seq.append(action)
            features += info["features"]  # numpy array
            acc_reward += reward

            if done:
                break

        env.close()
        features /= t
        exec_time = time.time() - t0

        return (
            acc_reward,
            (reward == -100),
            features,
            obs,
            exec_time,
            np.array(action_seq),
            frames
        )


class BWExecutor(Executor):

    def __init__(self, rand_seed: int, env_seeds: List[int], log_path: str) -> None:
        super().__init__(sim_steps=0, env_seed=0)
        self.executor = BWFramework(
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
        np.savetxt(self.inputs_buffer, input.reshape(1, -1), fmt="%1.0f", delimiter=",")
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
        behavior = np.array([measures[k] for k in self.features])
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
    model = load_bipedal_walker_model()

    # expert and generic measures
    features = MEASURES

    # experimental parameters
    test_budget = 5000
    init_budget = 1000
    cell_granularity = 50

    population_size, nb_iterations = 100, 50
    k = 3
    novelty_threshold = 0.005

    descriptor_sets = [
        ["action_entropy_mean", "length_mean"],
        ["action_entropy_mean", "length_spread"],
    ]
    descriptors = descriptor_sets[0]

    results_fp = Path("results_test/bw")
    results_fp.mkdir(parents=True, exist_ok=True)
    # (results_fp / "qd").mkdir(parents=True, exist_ok=True)
    # (results_fp / "ns").mkdir(parents=True, exist_ok=True)
    # (results_fp / "rt").mkdir(parents=True, exist_ok=True)
    (results_fp / "mdpfuzz").mkdir(parents=True, exist_ok=True)

    for seed in EXPERIMENT_SEEDS[:1]:
        print(f"Seed {seed} starts.")

        # f = BWFramework(
        #     seed,
        #     cell_granularity,
        #     features=features,
        #     descriptors=descriptors,
        # )
        # f.random_testing(model, ENV_SEEDS, test_budget, str(results_fp / "rt"))

        # f = BWFramework(
        #     seed,
        #     cell_granularity,
        #     features=features,
        #     descriptors=descriptors,
        # )
        # f.test_policy(
        #     model, ENV_SEEDS, test_budget, init_budget, str(results_fp / "qd")
        # )

        # f = BWFramework(
        #     seed,
        #     cell_granularity,
        #     features=features,
        #     descriptors=descriptors,
        # )
        # f.novelty_search(
        #     model,
        #     ENV_SEEDS,
        #     population_size,
        #     nb_iterations,
        #     k,
        #     novelty_threshold,
        #     str(results_fp / "ns"),
        # )

        executor = BWExecutor(seed, ENV_SEEDS, log_path=str(results_fp / "mdpfuzz"))
        fuzzer_logs_path = executor.fp + "_fuzzer"
        fuzzer = Fuzzer(random_seed=seed, executor=executor, k=4, tau=0.1, gamma=0.01)
        fuzzer.fuzzing_no_coverage(
            n=init_budget,
            test_budget=test_budget,  # 2*n will be removed since we assume that test_budget is the TOTAL budget
            policy=model,
            saving_path=fuzzer_logs_path,
            local_sensitivity=True,  # don"t re-run for computing the sensitivity
            exp_name="Bipedal Walker",
            light_pool=True,  # don"t log the inputs
            save_logs_only=True,  # don"t save evaluated inputs
        )
        executor.clean()
