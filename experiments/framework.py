import io
import json
import os
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tqdm
from metrics import (
    compute_action_distributions,
    compute_action_std,
    compute_divergence_time,
    compute_entropy,
)
from stable_baselines3.common.base_class import BaseAlgorithm

from common import compute_cell, get_bin_edges, get_expert_bin_edges


def concatenate_frames(frames_list: List[List[np.ndarray]]) -> np.ndarray:
    frames_list = [np.array(f) for f in frames_list]
    white_frame = np.ones_like(frames_list[0][0])
    max_length = np.max([len(l) for l in frames_list])
    frames_list = [
        (
            np.append(f, white_frame[None].repeat(max_length - len(f), axis=0), axis=0)
            if len(f) != max_length
            else f
        )
        for f in frames_list
    ]
    return np.concatenate(frames_list, axis=1)


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

        # additional attributes
        self.input_fmt = kwargs.get("input_fmt", "%.18e")  # type: str
        self.expert_indices = kwargs.get("expert_indices", [0, 1])  # type: Tuple[int, int]
        self.config["expert_indices"] = self.expert_indices

        # attribute set when a testing method is used
        self.name = kwargs.get("name", None)  # type: str
        self.behaviors_buffer = None  # type: io.TextIOWrapper
        self.inputs_buffer = None  # type: io.TextIOWrapper
        self.cells_buffer = None  # type: io.TextIOWrapper
        self.logs_buffer = None  # type: io.TextIOWrapper
        self.final_states_buffers = None  # type: List[io.TextIOWrapper]
        self.expert_behaviors_buffers = None  # type: List[io.TextIOWrapper]

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
            bs_size = len(cell_data[0][-1])
            columns = (
                ["mean_acc_reward", "failure_prob", "cell_index"]
                + [f"cell{i}" for i in range(2)]
                + self.features[:bs_size]
            )
            if len(columns) < (bs_size + 5):
                columns.extend([f"feature_{i}" for i in range((bs_size + 5) - len(columns))])
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
                    columns=columns,
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

    def prepare_logging(self, filepath: str, env_seeds: List[int]):
        self.behaviors_buffer = open(f"{filepath}_behaviors.txt", "w", buffering=1)
        self.inputs_buffer = open(f"{filepath}_inputs.txt", "w", buffering=1)
        self.cells_buffer = open(f"{filepath}_cells.txt", "w", buffering=1)
        self.logs_buffer = open(f"{filepath}_logs.txt", "w", buffering=1)
        # saves the N final states and expert behaviors separately
        self.final_states_buffers = [
            open(f"{filepath}_final_states_{seed}.txt", "w", buffering=1)
            for seed in env_seeds
        ]
        self.expert_behaviors_buffers = [
            open(f"{filepath}_expert_behaviors_{seed}.txt", "w", buffering=1)
            for seed in env_seeds
        ]

    def conclude_logging(self):
        buffers = [
            self.behaviors_buffer,
            self.inputs_buffer,
            self.cells_buffer,
            self.logs_buffer,
        ]
        if self.final_states_buffers is not None:
            buffers += self.final_states_buffers

        if self.expert_behaviors_buffers is not None:
            buffers += self.expert_behaviors_buffers

        for buffer in buffers:
            if buffer is not None:
                buffer.close()

    def log_execution(
        self,
        acc_reward: float,
        failure_prob: float,
        cell_selected_index: int,
        mutated_input_index: int,
        exec_time: float = None,
    ):
        log = f"episode_reward: {acc_reward}, failure_prob: {failure_prob}, cell_selected_index: {cell_selected_index}, cell_updated_index: {mutated_input_index}, nb_cells: {len(self.cells)}"
        if exec_time is not None:
            log += f", execution_time: {exec_time}"
        print(
            log,
            file=self.logs_buffer,
        )

    def log_data(
        self,
        input: np.ndarray,
        behavior: np.ndarray,
        cell: np.ndarray,
        final_states: List[np.ndarray],
        expert_behaviors: List[np.ndarray],
    ):

        np.savetxt(
            self.inputs_buffer, input.reshape(1, -1), fmt=self.input_fmt, delimiter=","
        )
        np.savetxt(self.behaviors_buffer, behavior.reshape(1, -1), delimiter=",")
        np.savetxt(
            self.cells_buffer, np.array(cell).reshape(1, -1), fmt="%1.0f", delimiter=","
        )
        for fs_buffer, fs in zip(self.final_states_buffers, final_states):
            np.savetxt(fs_buffer, fs.reshape(1, -1), delimiter=",")
        for eb_buffer, eb in zip(self.expert_behaviors_buffers, expert_behaviors):
            np.savetxt(eb_buffer, eb.reshape(1, -1), delimiter=",")

    @abstractmethod
    def execute_policy(
        self,
        input: np.ndarray,
        model: BaseAlgorithm,
        env_seed: int,
        deterministic: bool = True,
        render: bool = False,
    ) -> Tuple[
        float, bool, np.ndarray, np.ndarray, float, np.ndarray, List[np.ndarray]
    ]:
        """
        Parameters
        ----------
        input : np.ndarray
            Encoded values for setting the initial situation depicted by the test case.
        model : BaseAlgorithm
            Model under test.
        env_seed : int
            Seed to use when resetting the environment.
        deterministic : bool, optional
            If False, `model` is sampled. Default to True.
        render : bool, optional
            Whether to render the execution. Default to False.

        Returns
        -------
        acc_reward : float
            Accumulated reward.
        failed : bool
            Failure flag.
        behavior : np.ndarray
            Expert behavior.
        final_obs : np.ndarray
            Final observation.
        exec_time : float
            Execution time.
        action_seq : np.ndarray
            Actions
        frames : List[np.ndarray]
            List of RGB frames if `render` is True; empty list otherwise.
        """
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
            acc_reward, failed, behavior, final_obs, exec_time, action_seq, _frames = (
                self.execute_policy(
                    input, model, seed, deterministic=True, render=False
                )
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
        entropies = compute_entropy(action_dist)
        divergence_time = compute_divergence_time(
            actions, range=self.action_range, bins=self.action_bins
        )

        measures = dict(
            length_mean=np.mean(ep_length),
            length_std=np.std(ep_length),
            length_spread=max(ep_length) - min(ep_length),
            action_std=compute_action_std(actions),
            action_entropy_mean=entropies.mean(),
            action_entropy_argmax=np.argmax(entropies, axis=0).mean(),
            action_divergence=divergence_time,
            action_dist=action_dist,
        )

        return (
            np.mean(rewards),
            np.mean(failures),
            final_obs_list,
            behaviors_list,
            measures,
        )

    def render_stochastic_policy(
        self,
        input: np.ndarray,
        model: BaseAlgorithm,
        env_seeds: List[int],
    ) -> Tuple[
        float, float, List[np.ndarray], List[np.ndarray], np.ndarray, Dict[str, float]
    ]:
        """Same as `execute_stochastic_policy` but the results include the concatenated frames of the executions."""
        rewards, failures, actions = [], [], []
        final_obs_list = []
        behaviors_list = []
        frames_list = []
        for seed in env_seeds:
            acc_reward, failed, behavior, final_obs, exec_time, action_seq, frames = (
                self.execute_policy(input, model, seed, deterministic=True, render=True)
            )
            rewards.append(acc_reward)
            frames_list.append(frames)
            failures.append(failed)
            actions.append(action_seq)
            final_obs_list.append(final_obs)
            behaviors_list.append(behavior)

        # metrics for possible generic behavior space
        ep_length = [len(l) for l in actions]
        action_dist = compute_action_distributions(
            actions, range=self.action_range, bins=self.action_bins
        )
        entropies = compute_entropy(action_dist)
        divergence_time = compute_divergence_time(
            actions, range=self.action_range, bins=self.action_bins
        )

        measures = dict(
            length_mean=np.mean(ep_length),
            length_std=np.std(ep_length),
            length_spread=max(ep_length) - min(ep_length),
            action_std=compute_action_std(actions),
            action_entropy_mean=entropies.mean(),
            action_entropy_argmax=np.argmax(entropies, axis=0).mean(),
            action_divergence=divergence_time,
            action_dist=action_dist,
        )

        return (
            np.mean(rewards),
            np.mean(failures),
            final_obs_list,
            behaviors_list,
            concatenate_frames(frames_list),
            measures,
        )


    def process_env_seeds(self, env_seeds: List[int]):
        if len(env_seeds) == 1:
            self.xedges, self.yedges = get_expert_bin_edges(self.use_case, descriptors=self.expert_indices) # or [4, 8]
            get_behavior = lambda ebs_list, meas: ebs_list[0]
            get_cell = lambda behavior: compute_cell(
                behavior[self.expert_indices], self.xedges, self.yedges
            ).tolist()  # type: List[int]
        else:
            df = pd.read_csv(self.path_to_measures_extrema)
            self.xedges, self.yedges = get_bin_edges(
                df, measures=self.descriptors, num_bins=self.granularity
            )
            get_behavior = lambda ebs_list, meas: np.array([meas[k] for k in self.features]) # type: np.ndarray
            get_cell = lambda behavior: compute_cell(
                behavior[self.descriptor_indices], self.xedges, self.yedges
            ).tolist()  # type: List[int]

        self.config["xedges"] = list(self.xedges)
        self.config["yedges"] = list(self.yedges)
        return get_behavior, get_cell


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

        self.prepare_logging(filepath, env_seeds)

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

        get_behavior, get_cell = self.process_env_seeds(env_seeds)

        for _ in tqdm.tqdm(range(init_budget), disable=disable_pbar):
            input: np.ndarray = self.generate_input()

            t0 = time.time()
            episode_reward, failure_prob, final_obs_list, behaviors_list, measures = (
                self.execute_stochastic_policy(input, model, env_seeds)
            )
            t1 = time.time()

            behavior = get_behavior(behaviors_list, measures)

            inputs.append(input)
            behaviors.append(behavior)
            final_states.append(final_obs_list)
            expert_behaviors.append(behaviors_list)
            acc_rewards.append(episode_reward)
            failure_probs.append(failure_prob)
            execution_times.append(t1 - t0)

        behaviors = np.array(behaviors)

        for i in range(init_budget):
            behavior = behaviors[i]
            cell = get_cell(behavior)
            mutated_input_index = self.update_cell(
                cell, inputs[i], acc_rewards[i], failure_probs[i], behavior
            )
            self.log_execution(
                acc_rewards[i],
                failure_probs[i],
                -1,
                mutated_input_index,
                execution_times[i],
            )
            self.log_data(
                inputs[i], behaviors[i], cell, final_states[i], expert_behaviors[i]
            )

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

            behavior = get_behavior(behaviors_list, measures)

            cell = get_cell(behavior)

            mutated_input_index = self.update_cell(
                cell, mutated_input, episode_reward, failure_prob, behavior
            )
            self.log_execution(
                episode_reward, failure_prob, cell_index, mutated_input_index, t1 - t0
            )
            self.log_data(mutated_input, behavior, cell, final_obs_list, behaviors_list)

            current_time = time.time()
            nb_executions += 1
            pbar.update(1)

        testing_end_time = time.time()
        self.config["testing_start_time"] = testing_start_time
        self.config["testing_end_time"] = testing_end_time
        self.config["testing_time"] = testing_end_time - testing_start_time
        self.config["total_execution_time"] = sum(execution_times)

        pbar.close()
        self.conclude_logging()
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

        self.prepare_logging(filepath, env_seeds)

        time_budget = min(12, test_budget) * 3600
        executions_budget = test_budget if test_budget > 12 else 10000
        print(
            f"Time budget of {(time_budget / 60):.2f} minutes; bound to {executions_budget} executions."
        )

        get_behavior, get_cell = self.process_env_seeds(env_seeds)

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
            behavior = get_behavior(behaviors_list, measures)
            cell = get_cell(behavior)

            input_index = self.update_cell(
                cell, input, episode_reward, failure_prob, behavior
            )
            self.log_execution(episode_reward, failure_prob, -1, input_index, t1 - t0)
            self.log_data(input, behavior, cell, final_obs_list, behaviors_list)

            current_time = time.time()
            nb_executions += 1
            pbar.update(1)

        testing_end_time = time.time()
        self.config["testing_start_time"] = start_time
        self.config["testing_end_time"] = testing_end_time
        self.config["testing_time"] = testing_end_time - start_time
        self.config["total_execution_time"] = sum(execution_times)

        pbar.close()
        self.conclude_logging()
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

        self.prepare_logging(filepath, env_seeds)

        get_behavior, get_cell = self.process_env_seeds(env_seeds)

        # helpers 1: recording the executions during each iteration
        def record(
            input: np.ndarray,
            reward: float,
            failure_prob: float,
            behavior: np.ndarray,
            final_states_list: List[np.ndarray],
            expert_behaviors_list: List[np.ndarray],
        ) -> None:
            cell = get_cell(behavior)
            updated_cell_index = self.update_cell(
                cell, input, reward, failure_prob, behavior
            )
            self.log_execution(
                reward,
                failure_prob,
                -1,  # parent's cell is not logged
                updated_cell_index,
            )
            self.log_data(
                input, behavior, cell, final_states_list, expert_behaviors_list
            )

        # helpers 2: evaluates a batch of individuals
        def evaluate(individuals: np.ndarray) -> np.ndarray:
            behaviors = []
            for ind in individuals:
                r, fp, final_obs_list, behaviors_list, measures = (
                    self.execute_stochastic_policy(ind, model, env_seeds)
                )
                b = get_behavior(behaviors_list, measures)
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

        self.conclude_logging()
        self.save_state(filepath)
