import json
import os
import time
from typing import Any, List, Tuple

from map_builder import MapBuilder
import gym
import numpy as np
import torch
from common import ENV_SEEDS, EXPERIMENT_SEEDS, MEASURES, get_expert_bin_edges, load_taxi_model
from mdpfuzz.executor import Executor
from mdpfuzz.mdpfuzz import Fuzzer

from framework import Framework

MAP_FILEPATH = "map_large.txt"
INPUT_LOWS = [0, 0, 0, 0]
INPUT_UPS = [18, 13, 11, 11]
PASS_IN_TAXI_IDX = 11


class BehaviorSpace():
    """
    Implementation of a descriptor for the Taxi use-case.
    It computes 2d behaviors as the sum of the first and second half of the feature values, respectively.
    The container is a regular grid which evenly ranges from the minima and the maxima of the behaviors.
    """

    def __init__(self, lower_bounds: np.ndarray, upper_bounds: np.ndarray) -> None:
        n = len(lower_bounds)
        assert len(upper_bounds) == n
        tmp = int(n / 2)
        self.mins: np.ndarray = np.array([sum(lower_bounds[:tmp]), sum(lower_bounds[tmp:])], dtype=int)
        self.maxs: np.ndarray = np.array([sum(upper_bounds[:tmp]), sum(upper_bounds[tmp:])], dtype=int)
        self.x: List[int] = np.arange(self.mins[0], self.maxs[0] + 1).tolist()
        self.y: List[int] = np.arange(self.mins[1], self.maxs[1] + 1).tolist()


    def compute_behavior(self, feature: np.ndarray) -> np.ndarray:
        tmp = int(len(feature) / 2)
        return np.array([sum(feature[:tmp]), sum(feature[tmp:])], dtype=int)


    def compute_cell(self, behavior: np.ndarray) -> List[int]:
        assert len(behavior) == 2
        return [self.x.index(behavior[0]), self.y.index(behavior[1])]


    def describe(self, feature: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        """Convenient function that returns the behavior and its cell of @feature."""
        behavior = self.compute_behavior(feature)
        return behavior, self.compute_cell(behavior)


    def get_container(self) -> List[List[int]]:
        return [[i, j] for i in self.x for j in self.y]


class TTFramework(Framework):
    def __init__(self, rand_seed, cell_granularity, features, descriptors, **kwargs):
        super().__init__(rand_seed, cell_granularity, features, descriptors, **kwargs)

        self.action_range = [0, 6]  # type: Tuple[int, int]
        self.action_bins = 6  # type: int
        self.path_to_measures_extrema = "grid/tt/measures.csv"  # type: str
        self.use_case = "Taxi"
        self.input_fmt = "%1.0f"

        self._env = self.get_taxi_env()

    def process_env_seeds(self, env_seeds: List[int]):
        self.xedges, self.yedges = get_expert_bin_edges(self.use_case, descriptors=self.expert_indices)
        self.bs = BehaviorSpace(self.xedges, self.yedges)
        get_behavior = lambda ebs_list, meas: self.bs.compute_behavior(ebs_list[0])
        get_cell = lambda behavior: self.bs.compute_cell(behavior)

        # it should be this but it does not matter...
        # self.config["xedges"] = self.bs.x.copy()
        # self.config["yedges"] = self.bs.y.copy()
        self.config["xedges"] = list(self.xedges)
        self.config["yedges"] = list(self.yedges)
        return get_behavior, get_cell

    def get_taxi_env(self, map_fp: str = MAP_FILEPATH):
        map = MapBuilder(map_fp)
        return gym.make("Taxi-v3", map=map.map)


    def generate_input(self, **kwargs):
        input = self.rng.integers(low=INPUT_LOWS, high=INPUT_UPS, size=4)
        # checks if the passenger is already at the destination
        if  input[2] == input[3]:
            return self.generate_input()
        else:
            return input

    def generate_inputs(self, n, **kwargs):
        inputs = []
        while len(inputs) < n:
            inputs.append(self.generate_input())
        return np.array(inputs, dtype=int)


    def mutate(self, input, **kwargs):
        mutant = input.copy()
        idx = self.rng.integers(0, 4)
        tmp = np.arange(INPUT_UPS[idx])
        value = mutant[idx]

        weights = np.abs(tmp - value)
        inversed_weights = np.max(weights) - weights
        inversed_weights[value] = 0.0
        probs = inversed_weights / sum(inversed_weights)
        mutant[idx] = self.rng.choice(tmp, p=probs)

        # the passenger location and its destination must be different
        if (idx == 2) and (mutant[idx] == mutant[idx + 1]):
            return self.mutate(input)
        elif (idx == 3) and (mutant[idx - 1] == mutant[idx]):
            return self.mutate(input)
        else:
            return mutant


    def execute_policy(self, input, model, env_seed, deterministic=True, render=False):
        t0 = time.time()
        # no use of seed
        obs = self._env.reset(input)
        acc_reward = 0.0
        behavior = np.zeros(8)

        actions = []

        frames = []
        if render:
            frames.append(self._env.render("rgb_array"))

        done = False
        oracle = False

        while not done:
            action = model.step(obs)
            obs, reward, done, info = self._env.step(action)
            if render:
                frames.append(self._env.render("rgb_array"))
            acc_reward += reward
            actions.append(action)

            # checks whether the passenger is in the taxi
            pass_in_taxi = list(self._env.decode(obs))[2] == PASS_IN_TAXI_IDX

            if action < 4:
                behavior[action + 4 * int(pass_in_taxi)] += 1

            if not oracle:
                oracle = (reward == -10) or (info.get("crash", False))

            if done or oracle:
                break

        exec_time = time.time() - t0
        return acc_reward, oracle, behavior, np.array(list(self._env.decode(obs))), exec_time, np.expand_dims(actions, axis=1), frames


class TTExecutor(Executor):

    def __init__(self, rand_seed: int, env_seeds: List[int], log_path: str) -> None:
        super().__init__(sim_steps=0, env_seed=0)
        self.executor = TTFramework(
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
    model = load_taxi_model()

    # expert and generic measures
    features = MEASURES

    # experimental parameters
    test_budget = 5000
    init_budget = 1000
    cell_granularity = 50

    population_size, nb_iterations = 100, 50
    k = 3
    novelty_threshold = 0.9

    descriptor_sets = [
        ["action_entropy_mean", "length_mean"],
        ["action_entropy_mean", "length_spread"],
    ]
    descriptors = descriptor_sets[0]

    results_fp = Path("results_test/tt")
    results_fp.mkdir(parents=True, exist_ok=True)
    (results_fp / "qd").mkdir(parents=True, exist_ok=True)
    (results_fp / "ns").mkdir(parents=True, exist_ok=True)
    (results_fp / "rt").mkdir(parents=True, exist_ok=True)
    (results_fp / "mdpfuzz").mkdir(parents=True, exist_ok=True)

    for seed in EXPERIMENT_SEEDS:
        print(f"Seed {seed} starts.")

        f = TTFramework(
            seed,
            cell_granularity,
            features=features,
            descriptors=descriptors,
        )
        f.random_testing(model, ENV_SEEDS[0:1], test_budget, str(results_fp / "rt"))

        f = TTFramework(
            seed,
            cell_granularity,
            features=features,
            descriptors=descriptors,
        )
        f.test_policy(
            model, ENV_SEEDS[0:1], test_budget, init_budget, str(results_fp / "qd")
        )

        f = TTFramework(
            seed,
            cell_granularity,
            features=features,
            descriptors=descriptors,
        )
        f.novelty_search(
            model,
            ENV_SEEDS[0:1],
            population_size,
            nb_iterations,
            k,
            novelty_threshold,
            str(results_fp / "ns"),
        )

        executor = TTExecutor(seed, ENV_SEEDS[0:1], log_path=str(results_fp / "mdpfuzz"))
        fuzzer_logs_path = executor.fp + "_fuzzer"
        fuzzer = Fuzzer(random_seed=seed, executor=executor, k=4, tau=0.1, gamma=0.01)
        fuzzer.fuzzing_no_coverage(
            n=init_budget,
            test_budget=test_budget,  # 2*n will be removed since we assume that test_budget is the TOTAL budget
            policy=model,
            saving_path=fuzzer_logs_path,
            local_sensitivity=True,  # don"t re-run for computing the sensitivity
            exp_name="Taxi",
            light_pool=True,  # don"t log the inputs
            save_logs_only=True,  # don"t save evaluated inputs
        )
        executor.clean()
