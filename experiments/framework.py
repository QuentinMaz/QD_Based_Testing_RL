import json
import os
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tqdm
from metrics import compute_action_distributions, compute_action_std, compute_entropy
from stable_baselines3.common.base_class import BaseAlgorithm

from common import compute_cell, get_bin_edges


class Framework(ABC):
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

        # data structure consists of a list of cells (list of integers) and a list of list of test results
        self.cells: list[list[int]] = []
        # the test case results for each cell explored (input, performance, failure prob, behavior)
        self.cells_data: list[list[tuple[np.ndarray, float, float, np.ndarray]]] = []

        self.config = {
            "rand_seed": self.rand_seed,
            "cell_granularity": self.granularity,
            "features": self.features,
            "descriptors": self.descriptors,
            "descriptor_indices": self.descriptor_indices,
        }

        # attribute set when a testing method is used
        self.name = None

        # attributes to implement
        self.action_range = None  # type: Tuple[int, int]
        self.action_bins = None  # type: int
        self.path_to_measures_extrema = None  # type: str
        self.use_case = None  # type: str

    def save_configuration(self, filepath: str):
        """
        Saves the configuration of the object.
        This lets us know what BS has been used, which can be handy for organizing the results and to compare to MDPFuzz.
        """
        self.config["use_case"] = self.use_case
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
        cell_dfs = []
        for i, cell_data in enumerate(self.cells_data):
            # a record consist of a score, the oracle result, the cell index, the cell and behavior point
            cell_dfs.append(
                pd.DataFrame.from_records(
                    data=[
                        [mean_acc_reward, failure_prob, i]
                        + self.cells[i]
                        + behavior.tolist()
                        for (
                            _input,
                            mean_acc_reward,
                            failure_prob,
                            behavior,
                        ) in cell_data
                    ],
                    columns=["mean_acc_reward", "failure_prob", "cell_index"]
                    + [f"cell{i}" for i in range(2)]
                    + self.features,
                )
            )
        if len(cell_dfs) != 0:
            df = pd.concat(cell_dfs, ignore_index=True)
        else:
            df = pd.DataFrame()
        df.to_csv(f"{filepath}_data.csv", index=0)

        if len(self.cells_data) != 0:
            np.save(
                f"{filepath}_inputs.npy",
                np.concatenate(
                    [
                        np.array(list(map(lambda x: x[0], cell_data)))
                        for cell_data in self.cells_data
                    ]
                ),
            )
        # saves the random state
        self.save_random_state(filepath)
        # saves the configuration
        self.save_configuration(filepath)

    def load_configuration(self, filepath: str):
        """Loads and sets the configuration attribute of the instance."""
        if not filepath.endswith("config"):
            filepath += "_config"
        f = open(f"{filepath}.json", "r")
        self.config = json.load(f)
        f.close()

    def load_random_state(self, filepath: str):
        """Loads and sets the state of BitGenerator instance (of the Generator)."""
        if not filepath.endswith("state"):
            filepath += "_state"
        f = open(f"{filepath}.json", "r")
        self.rng.bit_generator.state = json.load(f)
        f.close()

    def load_state(self, filepath: str):
        """Loads a state of an instance to resume testing and returns the number of test cases loaded."""
        inputs_fp, df_fp = f"{filepath}_inputs.npy", f"{filepath}_data.csv"

        assert os.path.exists(inputs_fp) and os.path.exists(df_fp), "files are missing."
        self.cells = []
        self.cells_data = []

        inputs = np.load(inputs_fp)
        df = pd.read_csv(df_fp)
        assert len(inputs) == len(df)

        # removes 1 because of cell_index column
        bs_dim = len([c for c in df.columns.to_list() if c.startswith("cell")]) - 1
        assert bs_dim > 0

        for i, row in df.iterrows():
            row_data = row.tolist()
            cell, input, performance, is_faulty, behavior = (
                row_data[3 : 3 + bs_dim],
                inputs[i],
                row_data[0],
                row_data[1],
                row_data[3 + bs_dim :],
            )
            self.update_cell(cell, input, performance, is_faulty, np.array(behavior))

        self.load_random_state(filepath)
        self.load_configuration(filepath)
        self.loaded = True
        return len(df)

    def select_input(self, index: int):
        """Selection based on the failure probability if they are not all equal to 0; worst accumulated reward otherwise."""
        failure_probs = list(map(lambda x: x[2], self.cells_data[index]))
        if max(failure_probs) > 0.0:
            # the best performing input is one whose score is the maximum, since it corresponds to the failure probability.
            best_performer_index = int(np.argmax(failure_probs))
        else:
            print("No failure triggering input found in cell index {}.".format(index))
            scores = list(map(lambda x: x[1], self.cells_data[index]))
            best_performer_index = int(np.argmin(scores))
        return self.cells_data[index][best_performer_index][0]

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
        """
        index = None
        try:
            # index of the cell to update
            index = self.cells.index(cell)
            self.cells_data[index].append((input, performance, failure_prob, behavior))
            # print(f"[DATA UPDATE LOG] CELL {index} UPDATED: {len(cells_data[index])} RECORDS; AVG SCORE: {np.mean(list(map(lambda x: x[1], cells_data[index]))):.2f}.")
        except ValueError:
            self.cells.append(cell)
            self.cells_data.append([(input, performance, failure_prob, behavior)])
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

    @abstractmethod
    def mutate(
        self,
        input: np.ndarray,
        **kwargs,
    ):
        raise NotImplementedError()

    @abstractmethod
    def generate_input(self, **kwargs) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def generate_inputs(self, n: int, **kwargs) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def execute_policy(
        self,
        input: np.ndarray,
        model: BaseAlgorithm,
        env_seed: int,
        deterministic: bool = True,
    ) -> Tuple[float, bool, np.ndarray, np.ndarray, float]:
        raise NotImplementedError()

    def execute_stochastic_policy(
        self,
        input: np.ndarray,
        model: BaseAlgorithm,
        env_seeds: List[int],
    ) -> Tuple[float, float, List[np.ndarray], List[np.ndarray], Dict[str, float]]:
        rewards, failures, actions = [], [], []
        final_obs_list = []
        behaviors_list = []
        for seed in env_seeds:
            acc_reward, failed, behavior, final_obs, exec_time, action_seq = (
                self.execute_policy(input, model, seed, deterministic=True)
            )
            rewards.append(acc_reward)
            failures.append(failed)
            actions.append(action_seq)
            final_obs_list.append(final_obs)
            behaviors_list.append(behavior)

        # metrics for possible generic behavior space
        ep_length = [len(l) for l in actions]
        action_dist = compute_action_distributions(
            actions, range=self.action_range, bins=self.action_bins
        )

        measures = dict(
            length_mean=np.mean(ep_length),
            length_std=np.std(ep_length),
            length_spread=max(ep_length) - min(ep_length),
            action_std=compute_action_std(actions),
            action_entropy=compute_entropy(action_dist),
        )

        return (
            np.mean(rewards),
            np.mean(failures),
            final_obs_list,
            behaviors_list,
            measures,
        )

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
        self.name = "MAP-Elites"
        assert test_budget > init_budget
        self.test_budget = test_budget
        self.config["test_budget"] = self.test_budget
        self.init_budget = init_budget
        self.config["init_budget"] = self.init_budget

        self.config["env_seeds"] = env_seeds

        if os.path.isdir(results_fp):
            filepath = (
                f"{results_fp}{self.creation_time}"
                if results_fp.endswith("/")
                else f"{results_fp}/{self.creation_time}"
            )
        else:
            filepath = results_fp

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

        time_budget = min(12, test_budget) * 3600
        executions_budget = test_budget - init_budget if test_budget > 12 else 10000
        print(
            f"Time budget of {(time_budget / 60):.2f} minutes; bound to {executions_budget} executions."
        )

        inputs: List[np.ndarray] = []
        behaviors = []
        expert_behaviors = []
        final_states: List[List[np.ndarray]] = []
        acc_rewards: List[float] = []
        failure_probs: List[float] = []
        testing_start_time = time.time()
        execution_times = []

        for _ in tqdm.tqdm(range(init_budget), disable=disable_pbar):
            input: np.ndarray = self.generate_input()

            t0 = time.time()
            episode_reward, failure_prob, final_obs_list, behaviors_list, measures = (
                self.execute_stochastic_policy(input, model, env_seeds)
            )
            t1 = time.time()
            execution_times.append(t1 - t0)

            behavior = np.array([measures[k] for k in self.features])

            inputs.append(input)
            behaviors.append(behavior)
            final_states.append(final_obs_list)
            expert_behaviors.append(behaviors_list)
            acc_rewards.append(episode_reward)
            failure_probs.append(failure_prob)

        behaviors = np.array(behaviors)

        df = pd.read_csv(self.path_to_measures_extrema)
        self.xedges, self.yedges = get_bin_edges(
            df, measures=self.descriptors, num_bins=self.granularity
        )

        self.config["xedges"] = list(self.xedges)
        self.config["yedges"] = list(self.xedges)

        for i in range(init_budget):
            behavior = behaviors[i]
            cell = compute_cell(
                behavior[self.descriptor_indices], self.xedges, self.yedges
            ).tolist()
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

        start_time = time.time()
        current_time = time.time()
        nb_executions = 0
        pbar = tqdm.tqdm(total=executions_budget, disable=disable_pbar)

        while (current_time - start_time < time_budget) and (
            nb_executions < executions_budget
        ):
            cell_index = self.select_cell()
            self.last_cell_selected = cell_index
            input = self.select_input(cell_index)

            mutated_input = self.mutate(input)
            t0 = time.time()
            episode_reward, failure_prob, final_obs_list, behaviors_list, measures = (
                self.execute_stochastic_policy(mutated_input, model, env_seeds)
            )
            t1 = time.time()
            execution_times.append(t1 - t0)

            behavior = np.array([measures[k] for k in self.features])

            cell = compute_cell(
                behavior[self.descriptor_indices], self.xedges, self.yedges
            ).tolist()

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

            current_time = time.time()
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
        self.name = "Random Testing"
        self.test_budget = test_budget
        self.config["test_budget"] = self.test_budget
        self.config["env_seeds"] = env_seeds

        if os.path.isdir(results_fp):
            filepath = (
                f"{results_fp}{self.creation_time}"
                if results_fp.endswith("/")
                else f"{results_fp}/{self.creation_time}"
            )
        else:
            filepath = results_fp

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

        time_budget = min(12, test_budget) * 3600
        executions_budget = test_budget if test_budget > 12 else 10000
        print(
            f"Time budget of {(time_budget / 60):.2f} minutes; bound to {executions_budget} executions."
        )

        df = pd.read_csv(self.path_to_measures_extrema)
        self.xedges, self.yedges = get_bin_edges(
            df, measures=self.descriptors, num_bins=self.granularity
        )

        self.config["xedges"] = list(self.xedges)
        self.config["yedges"] = list(self.xedges)

        execution_times = []

        start_time = time.time()
        current_time = time.time()
        nb_executions = 0
        pbar = tqdm.tqdm(total=executions_budget, disable=disable_pbar)

        while (current_time - start_time < time_budget) and (
            nb_executions < executions_budget
        ):
            input: np.ndarray = self.generate_input()
            t0 = time.time()
            episode_reward, failure_prob, final_obs_list, behaviors_list, measures = (
                self.execute_stochastic_policy(input, model, env_seeds)
            )
            t1 = time.time()
            execution_times.append(t1 - t0)
            behavior = np.array([measures[k] for k in self.features])
            cell = compute_cell(
                behavior[self.descriptor_indices], self.xedges, self.yedges
            ).tolist()

            input_index = self.update_cell(
                cell, input, episode_reward, failure_prob, behavior
            )
            print(
                f"episode_reward: {episode_reward}, failure_prob: {failure_prob}, cell_selected_index: -1, cell_updated_index: {input_index}, nb_cells: {len(self.cells)}, execution_time: {t1 - t0}",
                file=logs_buffer,
            )
            np.savetxt(inputs_buffer, input.reshape(1, -1), fmt="%1.0f", delimiter=",")
            np.savetxt(behaviors_buffer, behavior.reshape(1, -1), delimiter=",")
            np.savetxt(
                cells_buffer, np.array(cell).reshape(1, -1), fmt="%1.0f", delimiter=","
            )

            for fs_buffer, fs in zip(final_states_buffers, final_obs_list):
                np.savetxt(fs_buffer, fs.reshape(1, -1), delimiter=",")
            for eb_buffer, eb in zip(expert_behaviors_buffers, behaviors_list):
                np.savetxt(eb_buffer, eb.reshape(1, -1), delimiter=",")

            current_time = time.time()
            nb_executions += 1
            pbar.update(1)

        testing_end_time = time.time()
        self.config["testing_start_time"] = start_time
        self.config["testing_end_time"] = testing_end_time
        self.config["testing_time"] = testing_end_time - start_time
        self.config["total_execution_time"] = sum(execution_times)

        pbar.close()
        behaviors_buffer.close()
        inputs_buffer.close()
        cells_buffer.close()
        logs_buffer.close()
        for buffer in final_states_buffers + expert_behaviors_buffers:
            buffer.close()
        self.save_state(filepath)

    def novelty_search(
        self,
        model: BaseAlgorithm,
        env_seeds: List[int],
        pop_size: int,
        nb_iterations: int,
        k: int,
        nov_threshold: float,
        results_fp: str,
        disable_pbar: bool = False,
    ):
        """Does not use cached data anymore."""
        self.name = "Novelty Search"
        self.config["pop_size"] = pop_size
        self.config["nb_iterations"] = nb_iterations
        self.config["test_budget"] = pop_size * nb_iterations
        self.config["env_seeds"] = env_seeds
        self.config["nov_threshold"] = nov_threshold
        self.config["k"] = k

        if os.path.isdir(results_fp):
            filepath = (
                f"{results_fp}{self.creation_time}"
                if results_fp.endswith("/")
                else f"{results_fp}/{self.creation_time}"
            )
        else:
            filepath = results_fp

        # to collect the data during the search, i.e., every model execution
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

        df = pd.read_csv(self.path_to_measures_extrema)
        self.xedges, self.yedges = get_bin_edges(
            df, measures=self.descriptors, num_bins=self.granularity
        )

        self.config["xedges"] = list(self.xedges)
        self.config["yedges"] = list(self.xedges)

        # helpers 1: recording the executions during each iteration
        def record(
            input: np.ndarray,
            reward: float,
            failure_prob: float,
            behavior: np.ndarray,
            final_states_list: List[np.ndarray],
            expert_behaviors_list: List[np.ndarray],
        ) -> None:
            cell = compute_cell(
                behavior[self.descriptor_indices], self.xedges, self.yedges
            ).tolist()
            updated_cell_index = self.update_cell(
                cell, input, reward, failure_prob, behavior
            )
            # parent"s cell is not logged
            print(
                f"episode_reward: {reward}, failure_prob: {failure_prob}, cell_updated_index: {updated_cell_index}, nb_cells: {len(self.cells)}",
                file=logs_buffer,
            )
            np.savetxt(inputs_buffer, input.reshape(1, -1), fmt="%1.0f", delimiter=",")
            np.savetxt(behaviors_buffer, behavior.reshape(1, -1), delimiter=",")
            np.savetxt(
                cells_buffer, np.array(cell).reshape(1, -1), fmt="%1.0f", delimiter=","
            )
            for fs_buffer, fs in zip(final_states_buffers, final_states_list):
                np.savetxt(fs_buffer, fs.reshape(1, -1), delimiter=",")
            for eb_buffer, eb in zip(expert_behaviors_buffers, expert_behaviors_list):
                np.savetxt(eb_buffer, eb.reshape(1, -1), delimiter=",")

        # helpers 2: evaluates a batch of individuals
        def evaluate(individuals: np.ndarray) -> np.ndarray:
            behaviors = []
            for ind in individuals:
                r, fp, final_obs_list, behaviors_list, measures = (
                    self.execute_stochastic_policy(ind, model, env_seeds)
                )
                b = np.array([measures[k] for k in self.features])
                record(ind, r, fp, b, final_obs_list, behaviors_list)
                behaviors.append(b)
            return np.array(behaviors)

        # helper 3: mutates a batch of individuals
        def mutate(inputs: np.ndarray):
            mutants = [self.mutate(input) for input in inputs]
            return np.array(mutants)

        # ns logs
        ns_logs_buffer = open(f"{filepath}_ns_logs.txt", "w", buffering=1)
        nov_scores_buffer = open(f"{filepath}_nov_scores.txt", "w", buffering=1)
        # initial population and novelty archive
        from novelty_search import NoveltyArchive

        pop = self.generate_inputs(pop_size)
        pop_behaviors = evaluate(pop)
        nov_archive = NoveltyArchive(pop_behaviors, k, nov_threshold)
        pop_nov_scores = nov_archive.score(pop_behaviors)
        [
            np.savetxt(nov_scores_buffer, s.reshape(1, -1), delimiter=",")
            for s in pop_nov_scores
        ]
        # novelty search loop
        print(
            f"iteration: 0, archive_size: {nov_archive.size()}, archive_sparseness: {nov_archive.archive_sparseness():0.5f}",
            file=ns_logs_buffer,
        )
        for i in tqdm.tqdm(range(1, nb_iterations), disable=disable_pbar):
            # 1. generates offspring
            offspring = mutate(pop)
            # 1. evaluates the offspring
            offspring_behaviors = evaluate(offspring)
            # 1. novelty scores of the offspring w.r.t the archive and the population
            offspring_nov_scores = nov_archive.score(offspring_behaviors, pop_behaviors)

            # 2. selects the most novel individuals to form the new population
            joined_pop = np.vstack([pop, offspring])
            joined_scores = np.hstack([pop_nov_scores, offspring_nov_scores])
            median_score = np.median(joined_scores)

            # 3. updates the archive
            _updated, _offspring_indices = nov_archive.update3(offspring_behaviors)

            # 4. updates the population and their data
            mask = joined_scores >= median_score

            pop = joined_pop[mask].copy()
            pop_behaviors = np.vstack([pop_behaviors, offspring_behaviors])[mask]
            pop_nov_scores = nov_archive.score(pop_behaviors)
            if len(pop) > pop_size:
                pop = pop[:pop_size]
                pop_behaviors = pop_behaviors[:pop_size]
                pop_nov_scores = pop_nov_scores[:pop_size]

            # assert len(pop) == pop_size, (len(pop), pop.shape)
            # assert len(pop_behaviors) == pop_size, (len(pop), pop.shape)
            # assert len(pop_nov_scores) == pop_size, (len(pop), pop.shape)
            [
                np.savetxt(nov_scores_buffer, s.reshape(1, -1), delimiter=",")
                for s in pop_nov_scores
            ]
            print(
                f"iteration: {i}, archive_size: {nov_archive.size()}, archive_sparseness: {nov_archive.archive_sparseness():0.5f}",
                file=ns_logs_buffer,
            )

        behaviors_buffer.close()
        inputs_buffer.close()
        cells_buffer.close()
        logs_buffer.close()
        for buffer in final_states_buffers + expert_behaviors_buffers:
            buffer.close()
        self.save_state(filepath)
