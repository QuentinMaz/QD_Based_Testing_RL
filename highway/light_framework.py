import json
import os
import time
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import tqdm
from agents import AgentWrapper
from my_utils import compute_cell, get_bin_edges
from stable_baselines3.common.base_class import BaseAlgorithm

from executor import HighwayTestManager

EXPERIMENT_SEEDS = [2021, 42, 2023, 20, 0, 10, 4, 2006, 512, 1453]
ENV_SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
POP_SIZES = [100, 250, 500]
ITERATIONS = [50, 20, 10]
FEATURES = [
    "length_mean",
    "length_std",
    "length_spread",
    "action_std",
    "action_entropy_mean",
    "action_entropy_argmax",
    "action_divergence",
]


class Framework:
    def __init__(
        self,
        rand_seed: int,
        cell_granularity: int,
        features: List[str],
        descriptors: Tuple[str, str],
        **kwargs,
    ) -> None:
        """Init.

        Parameters
        ----------
        features : List[str]
            Names of the generic measures. At least two.
        descriptors : Tuple[str, str]
            Names of the two measures in `features` (x and y when plotting) for the generic behavior space and grid.
        """
        self.rand_seed = rand_seed
        self.rng: np.random.Generator = np.random.default_rng(rand_seed)
        self.creation_time = time.time()

        self.loaded = False
        self.has_init = False
        self.test_budget = None
        self.init_budget = None

        self.granularity = cell_granularity
        self.features = features
        self.descriptors = descriptors

        if not all(isinstance(v, str) for v in self.descriptors):
            raise ValueError("The descriptors must be string.")

        if not all(v in self.features for v in self.descriptors):
            raise ValueError("The descriptors must be in the feature list.")

        self.descriptor_indices = [self.features.index(d) for d in self.descriptors]

        # as indices
        self.last_cell_selected = None
        self.last_cell_updated = None

        # data structure consists of a list of cells (list of integers) and a list of test results
        self.cells: list[list[int]] = []
        # the test case results for each cell explored (input, performance, failure prob result, behavior)
        self.cells_data: list[tuple[np.ndarray, float, float, np.ndarray]] = []

        self.config = {
            "rand_seed": self.rand_seed,
            "cell_granularity": self.granularity,
            "features": self.features,
            "descriptors": self.descriptors,
            "descriptor_indices": self.descriptor_indices,
            "use_case": "Highway",
        }

        self.name = None
        self.executor = HighwayTestManager()

    def save_configuration(self, filepath: str):
        """
        Saves the configuration of the object.
        This lets us know what BS has been used, which can be handy for organizing the results and to compare to MDPFuzz.
        """
        if self.name is not None:
            self.config["name"] = self.name
        if not filepath.endswith("config"):
            filepath += "_config"
        f = open(f"{filepath}.json", "w")
        f.write(json.dumps(self.config))
        f.close()

    def save_random_state(self, filepath: str):
        """Saves the state of the BitGenerator instance (of the Generator)."""
        f = open(f"{filepath}_state.json", "w")
        f.write(json.dumps(self.rng.bit_generator.state))
        f.close()
        return self.rng.bit_generator.state

    def save_state(self, filepath: str):
        """
        Saves the current state of the framework to possibly resume execution.
        The resulting data is a .csv file export of a DataFrame and a .npy file of the inputs.
        Both data shares the same order, which is not temporal (logs are though) but results from iterating over the results for each cell.
        """
        with open(f"{filepath}_data.csv", "w") as f:
            f.write("")
        with open(f"{filepath}_cells.txt", "w") as f:
            f.write("")
        # saves the random state
        self.save_random_state(filepath)
        # saves the configuration
        self.save_configuration(filepath)

    def select_input(self, index: int):
        """Returns the current solution in the cell."""
        return self.cells_data[index][0]

    def local_competition(self, index: int, failure_prob: float, acc_reward : float):
        """
        2-step local competition: on failure probability maximization if they are not 0.0;
        then on accumulated reward minimization.

        Returns True if the input solution is better (i.e., the cell must be updated).
        """
        curr_fp, curr_acc_r = self.cells_data[index][1:3]
        if curr_fp != 0:
            # higher failure probability
            return failure_prob >= curr_fp
        else:
            # none zero failure probability or lower acc. reward
            return (failure_prob != 0) or (acc_reward < curr_acc_r)


    def select_cell(self):
        """Selects the cell for the next search iteration."""
        return int(self.rng.integers(0, len(self.cells)))


    def update_cell(
        self,
        cell: List[int],
        input: np.ndarray,
        performance: float,
        failure_prob: float,
        behavior: np.ndarray,
    ):
        """
        Records the execution result to the corresponding cell.
        It returns the index of the cell updated.
        It performs the local competition of MAP-Elites.
        """
        index = None
        try:
            # index of the cell to update
            index = self.cells.index(cell)
            if self.local_competition(index, failure_prob, performance):
                self.cells_data[index] = (input, performance, failure_prob, behavior)
        except ValueError:
            self.cells.append(cell)
            self.cells_data.append((input, performance, failure_prob, behavior))
            # print(f"[DATA UPDATE LOG] NEW CELL CREATED. CURRENT SCORE: {performance}.")
        finally:
            # sanity checks
            assert len(self.cells) == len(
                self.cells_data
            ), "inconsistent cells and cells_data lists!"
            self.last_cell_updated = (
                index if index is not None else (len(self.cells) - 1)
            )
        return self.last_cell_updated

    def mutate(self, input: np.ndarray) -> np.ndarray:
        return self.executor.mutate_input(input, self.rng)

    def test_policy(
        self,
        model: BaseAlgorithm,
        env_seeds: List[int],
        test_budget: int,
        init_budget: int,
        results_fp: str,
        disable_pbar: bool = False,
    ):
        """
        Parameters
        ----------
        env_seeds : List[int]
            Seeds for the executions of the test cases.
        """

        assert test_budget > init_budget
        self.test_budget = test_budget
        self.config["test_budget"] = self.test_budget
        self.init_budget = init_budget
        self.config["init_budget"] = self.init_budget
        self.name = "MAP-Elites"

        self.config["env_seeds"] = env_seeds
        n = len(env_seeds)
        self.executor.seeds = env_seeds

        if os.path.isdir(results_fp):
            filepath = (
                f"{results_fp}{self.creation_time}"
                if results_fp.endswith("/")
                else f"{results_fp}/{self.creation_time}"
            )
        else:
            filepath = results_fp

        if os.getenv("SLURM_ARRAY_TASK_ID"):
            filepath += os.getenv("SLURM_ARRAY_TASK_ID")

        behaviors_buffer = open(f"{filepath}_behaviors.txt", "w", buffering=1)
        inputs_buffer = open(f"{filepath}_inputs.txt", "w", buffering=1)
        cells_buffer = open(f"{filepath}_cells.txt", "w", buffering=1)
        logs_buffer = open(f"{filepath}_logs.txt", "w", buffering=1)
        # saves the N final states and expert behaviors separately
        final_states_buffers = [
            open(f"{filepath}_final_states_{seed}.txt", "w", buffering=1)
            for seed in env_seeds
        ]
        expert_behaviors_buffers = [
            open(f"{filepath}_expert_behaviors_{seed}.txt", "w", buffering=1)
            for seed in env_seeds
        ]

        executions_budget = test_budget - init_budget
        print(
            f"Total testing budget of {test_budget}, with {init_budget} iterations for the initialization."
        )

        inputs: List[np.ndarray] = []
        behaviors = []
        expert_behaviors = []
        final_states: List[List[np.ndarray]] = []
        acc_rewards: List[float] = []
        failure_probs: List[float] = []
        testing_start_time = time.time()
        execution_times = []

        if len(env_seeds) == 1:
            self.xedges, self.yedges = np.load("../experiments/grid/hw/edges.npy")
            get_behavior = lambda ebs_list, meas: ebs_list[0]
            get_cell = lambda behavior: compute_cell(
                behavior[[0, 1]], self.xedges, self.yedges
            ).tolist()  # type: List[int]
        else:
            df = pd.read_csv("measures.csv")
            model_name = "DQN" if not isinstance(model, AgentWrapper) else model.model_name
            df = df.loc[df.model_name == model_name]
            self.xedges, self.yedges = get_bin_edges(
                df, measures=self.descriptors, num_bins=self.granularity
            )
            get_behavior = lambda ebs_list, meas: np.array([meas[k] for k in self.features]) # type: np.ndarray
            get_cell = lambda behavior: compute_cell(
                behavior[self.descriptor_indices], self.xedges, self.yedges
            ).tolist()  # type: List[int]

        self.config["xedges"] = list(self.xedges)
        self.config["yedges"] = list(self.xedges)

        for _ in tqdm.tqdm(range(init_budget), disable=disable_pbar):
            input: np.ndarray = self.executor.generate_input(self.rng)

            t0 = time.time()
            episode_reward, failure_prob, final_obs_list, behaviors_list, measures = (
                self.executor.execute_stochastic_policy(
                    input, model, n=n, deterministic=True
                )
            )
            t1 = time.time()
            execution_times.append(t1 - t0)

            behavior = get_behavior(behaviors_list, measures)

            inputs.append(input)
            behaviors.append(behavior)
            final_states.append(final_obs_list)
            expert_behaviors.append(behaviors_list)
            acc_rewards.append(episode_reward)
            failure_probs.append(failure_prob)

        behaviors = np.array(behaviors)

        for i in range(init_budget):
            behavior = behaviors[i]
            cell = get_cell(behavior)
            mutated_input_index = self.update_cell(
                cell, inputs[i], acc_rewards[i], failure_probs[i], behavior
            )
            print(
                f"episode_reward: {acc_rewards[i]}, failure_prob: {failure_probs[i]}, cell_selected_index: -1, cell_updated_index: {mutated_input_index}, nb_cells: {len(self.cells)}, execution_time: {t1 - t0}",
                file=logs_buffer,
            )
            np.savetxt(
                inputs_buffer, inputs[i].reshape(1, -1), fmt="%1.0f", delimiter=","
            )
            np.savetxt(behaviors_buffer, behavior.reshape(1, -1), delimiter=",")
            np.savetxt(
                cells_buffer, np.array(cell).reshape(1, -1), fmt="%1.0f", delimiter=","
            )
            for fs_buffer, fs in zip(final_states_buffers, final_states[i]):
                np.savetxt(fs_buffer, fs.reshape(1, -1), delimiter=",")
            for eb_buffer, eb in zip(expert_behaviors_buffers, expert_behaviors[i]):
                np.savetxt(eb_buffer, eb.reshape(1, -1), delimiter=",")

        nb_executions = 0
        pbar = tqdm.tqdm(total=executions_budget, disable=disable_pbar)

        while (nb_executions < executions_budget):
            cell_index = self.select_cell()
            self.last_cell_selected = cell_index
            input = self.select_input(cell_index)

            mutated_input = self.mutate(input)
            t0 = time.time()
            episode_reward, failure_prob, final_obs_list, behaviors_list, measures = (
                self.executor.execute_stochastic_policy(
                    mutated_input, model, n=n, deterministic=True
                )
            )
            t1 = time.time()
            execution_times.append(t1 - t0)

            behavior = get_behavior(behaviors_list, measures)

            cell = get_cell(behavior)

            mutated_input_index = self.update_cell(
                cell, mutated_input, episode_reward, failure_prob, behavior
            )
            print(
                f"episode_reward: {episode_reward}, failure_prob: {failure_prob}, cell_selected_index: {cell_index}, cell_updated_index: {mutated_input_index}, nb_cells: {len(self.cells)}, execution_time: {t1 - t0}",
                file=logs_buffer,
            )
            np.savetxt(
                inputs_buffer, mutated_input.reshape(1, -1), fmt="%1.0f", delimiter=","
            )
            np.savetxt(behaviors_buffer, behavior.reshape(1, -1), delimiter=",")
            np.savetxt(
                cells_buffer, np.array(cell).reshape(1, -1), fmt="%1.0f", delimiter=","
            )

            for fs_buffer, fs in zip(final_states_buffers, final_obs_list):
                np.savetxt(fs_buffer, fs.reshape(1, -1), delimiter=",")
            for eb_buffer, eb in zip(expert_behaviors_buffers, behaviors_list):
                np.savetxt(eb_buffer, eb.reshape(1, -1), delimiter=",")

            nb_executions += 1
            pbar.update(1)

        testing_end_time = time.time()
        self.config["testing_start_time"] = testing_start_time
        self.config["testing_end_time"] = testing_end_time
        self.config["testing_time"] = testing_end_time - testing_start_time
        self.config["total_execution_time"] = sum(execution_times)

        pbar.close()
        behaviors_buffer.close()
        inputs_buffer.close()
        cells_buffer.close()
        logs_buffer.close()
        for buffer in final_states_buffers + expert_behaviors_buffers:
            buffer.close()
        self.save_state(filepath)

    def random_testing(
        self,
        model: BaseAlgorithm,
        env_seeds: List[int],
        test_budget: int,
        results_fp: str,
        disable_pbar: bool = False,
    ):
        """Random testing loop baseline."""
        self.test_budget = test_budget
        self.config["test_budget"] = self.test_budget
        self.config["env_seeds"] = env_seeds
        n = len(env_seeds)
        self.executor.seeds = env_seeds
        self.name = "Random Testing"

        if os.path.isdir(results_fp):
            filepath = (
                f"{results_fp}{self.creation_time}"
                if results_fp.endswith("/")
                else f"{results_fp}/{self.creation_time}"
            )
        else:
            filepath = results_fp

        if os.getenv("SLURM_ARRAY_TASK_ID"):
            filepath += os.getenv("SLURM_ARRAY_TASK_ID")

        behaviors_buffer = open(f"{filepath}_behaviors.txt", "w", buffering=1)
        inputs_buffer = open(f"{filepath}_inputs.txt", "w", buffering=1)
        logs_buffer = open(f"{filepath}_logs.txt", "w", buffering=1)
        # saves the N final states and expert behaviors separately
        final_states_buffers = [
            open(f"{filepath}_final_states_{seed}.txt", "w", buffering=1)
            for seed in env_seeds
        ]
        expert_behaviors_buffers = [
            open(f"{filepath}_expert_behaviors_{seed}.txt", "w", buffering=1)
            for seed in env_seeds
        ]

        print(f"Testing budget of {test_budget}.")

        if len(env_seeds) == 1:
            self.xedges, self.yedges = np.load("../experiments/grid/hw/edges.npy")
            get_behavior = lambda ebs_list, meas: ebs_list[0]
        else:
            df = pd.read_csv("measures.csv")
            model_name = "DQN" if not isinstance(model, AgentWrapper) else model.model_name
            df = df.loc[df.model_name == model_name]
            self.xedges, self.yedges = get_bin_edges(
                df, measures=self.descriptors, num_bins=self.granularity
            )
            get_behavior = lambda ebs_list, meas: np.array([meas[k] for k in self.features]) # type: np.ndarray

        self.config["xedges"] = list(self.xedges)
        self.config["yedges"] = list(self.xedges)

        start_time = time.time()
        nb_executions = 0
        pbar = tqdm.tqdm(total=test_budget, disable=disable_pbar)

        while (nb_executions < test_budget):
            input: np.ndarray = self.executor.generate_input(self.rng)
            t0 = time.time()
            episode_reward, failure_prob, final_obs_list, behaviors_list, measures = (
                self.executor.execute_stochastic_policy(
                    input, model, n=n, deterministic=True
                )
            )
            t1 = time.time()
            behavior = get_behavior(behaviors_list, measures)
            print(
                f"episode_reward: {episode_reward}, failure_prob: {failure_prob}, cell_selected_index: -1, cell_updated_index: -1, nb_cells: -1, execution_time: {t1 - t0}",
                file=logs_buffer,
            )
            np.savetxt(inputs_buffer, input.reshape(1, -1), fmt="%1.0f", delimiter=",")
            np.savetxt(behaviors_buffer, behavior.reshape(1, -1), delimiter=",")

            for fs_buffer, fs in zip(final_states_buffers, final_obs_list):
                np.savetxt(fs_buffer, fs.reshape(1, -1), delimiter=",")
            for eb_buffer, eb in zip(expert_behaviors_buffers, behaviors_list):
                np.savetxt(eb_buffer, eb.reshape(1, -1), delimiter=",")

            nb_executions += 1
            pbar.update(1)

        testing_end_time = time.time()
        self.config["testing_start_time"] = start_time
        self.config["testing_end_time"] = testing_end_time
        self.config["testing_time"] = testing_end_time - start_time

        pbar.close()
        behaviors_buffer.close()
        inputs_buffer.close()
        logs_buffer.close()
        for buffer in final_states_buffers + expert_behaviors_buffers:
            buffer.close()
        self.save_state(filepath)


if __name__ == "__main__":
    from pathlib import Path

    from executor import HighwayTestManager

    torch.set_num_threads(1)
    main_seed = 2021
    dqnagent_path = "saved_models/dqnagent/checkpoint-35000.tar"
    model = HighwayTestManager.load_policy(dqnagent_path)

    # experimental parameters
    test_budget = 15#00
    init_budget = 5#00
    cell_granularity = 50

    descriptors = ["action_entropy_mean", "length_spread"]

    results_fp = Path("test_env_seeds/hw")
    results_fp.mkdir(parents=True, exist_ok=True)
    (results_fp / "qd").mkdir(parents=True, exist_ok=True)
    # (results_fp / "rt").mkdir(parents=True, exist_ok=True)

    for seed in EXPERIMENT_SEEDS[:1]:
        print(f"Seed {seed} starts.")

        # f = Framework(
        #     seed,
        #     cell_granularity,
        #     features=FEATURES,
        #     descriptors=descriptors,
        #     name="Random Testing",
        # )
        # f.random_testing(model, ENV_SEEDS[:5], test_budget, str(results_fp / "rt"))

        f = Framework(
            seed,
            cell_granularity,
            features=FEATURES,
            descriptors=descriptors,
            name="MAP-Elites",
        )
        f.test_policy(
            model, ENV_SEEDS[:5], test_budget, init_budget, str(results_fp / "qd")
        )
