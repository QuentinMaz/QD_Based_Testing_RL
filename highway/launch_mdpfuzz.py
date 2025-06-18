import json
import os
import time
from typing import Any, Dict, List, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from mdpfuzz.executor import Executor
from mdpfuzz.logger import FuzzerLogger
from mdpfuzz.mdpfuzz import Fuzzer
from executor import HighwayTestManager
from hw_framework import ENV_SEEDS


class MDPFuzzExecutor(Executor):

    def __init__(self, sim_steps, env_seeds: List[int], log_path: str) -> None:
        super().__init__(sim_steps, env_seed=0)
        self.executor = HighwayTestManager(seeds=env_seeds)
        self.env_seeds = env_seeds

        self.creation_time = time.time()
        if os.path.isdir(log_path):
            filepath = f"{log_path}{self.creation_time}" if log_path.endswith("/") else f"{log_path}/{self.creation_time}"
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

        self.config = {
            "use_case": "Highway",
            "name": "MDPFuzz",
            "env_seeds": self.env_seeds
        }


    def generate_input(self, rng: np.random.Generator) -> np.ndarray:
        return self.executor.generate_input(rng)


    def generate_inputs(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return self.executor.generate_inputs(rng, n)


    def mutate(
        self, input: np.ndarray, rng: np.random.Generator, **kwargs
    ) -> np.ndarray:
        return self.executor.mutate_input(input, rng)


    def load_policy(self, **kwargs):
        return self.executor.load_policy(kwargs.get("model_path", None))


    def log_execution(
            self,
            input: np.ndarray,
            mean_acc_reward: float,
            failure_prob: float,
            final_obs_list: List[np.ndarray],
            expert_behaviors_list: List[np.ndarray],
            behavior: np.ndarray,
            exec_time: float
        ):
        np.savetxt(self.inputs_buffer, input.reshape(1, -1), fmt="%1.0f", delimiter=",")
        np.savetxt(self.behaviors_buffer, behavior.reshape(1, -1), delimiter=",")
        for buffer, fs in zip(self.final_states_buffers, final_obs_list):
            np.savetxt(buffer, fs.reshape(1, -1), delimiter=",")
        for eb_buffer, eb in zip(self.expert_behaviors_buffers, expert_behaviors_list):
                np.savetxt(eb_buffer, eb.reshape(1, -1), delimiter=",")
        print(f"episode_reward: {mean_acc_reward}, failure_prob: {failure_prob}, execution_time: {exec_time}", file=self.logs_buffer)


    def execute_policy(
        self, input: np.ndarray, policy: Any
    ) -> Tuple[float, bool, np.ndarray, float]:
        t0 = time.time()

        # failure, obs_seq, action_seq, reward_seq, trajectory, frames = self.executor._record_execution(
        #     policy, input, record=False, deterministic=True
        # )
        mean_acc_reward, failure_prob, final_obs_list, behaviors_list, measures = self.executor.execute_stochastic_policy(
            input, policy, len(self.env_seeds), deterministic=True
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
            exec_time
        )

        return (
            mean_acc_reward,
            bool(failure_prob),
            [],
            exec_time,
        )


    def clean(self):
        """Closes the file buffers and saves the configuration."""
        self.behaviors_buffer.close()
        self.inputs_buffer.close()
        self.logs_buffer.close()
        for buffer in self.final_states_buffers:
            buffer.close()
        with open(f"{self.fp}_config.json", "w") as f:
            f.write(json.dumps(self.config))


def accumulate_failures(failures: np.ndarray) -> np.ndarray:
    if failures.dtype != int:
        failures = failures.astype(int)

    num_failures = 0
    acc_failures = []
    for f in failures:
        num_failures += f
        acc_failures.append(num_failures)
    return np.array(acc_failures, dtype=int)


if __name__ == "__main__":
    from pathlib import Path

    results_fp = Path("results/hw/mdpfuzz")
    results_fp.mkdir(parents=True, exist_ok=True)


    executor = MDPFuzzExecutor(
        sim_steps=800,
        env_seeds=ENV_SEEDS,
        log_path=str(results_fp)
    )
    fuzzer_logs_path = executor.fp + "_fuzzer"

    model = executor.load_policy(
        model_path="saved_models/dqnagent/checkpoint-35000.tar"
    )


    test_budget = 80
    init_budget = 20
    # GMM parameters won't be used
    fuzzer = Fuzzer(random_seed=0, executor=executor, k=4, tau=0.1, gamma=0.01)
    fuzzer.fuzzing_no_coverage(
        n=init_budget,
        test_budget=test_budget, # 2*n will be removed since we assume that test_budget is the TOTAL budget
        policy=model,
        saving_path=fuzzer_logs_path,
        local_sensitivity=True, # don"t re-run for computing the sensitivity
        exp_name="Highway",
        light_pool=True, # don"t log the inputs
        save_logs_only=True # don"t save evaluated inputs
    )
    executor.clean()

    mdpfuzz_failures = accumulate_failures(
        FuzzerLogger(
            fuzzer_logs_path + "_logs.txt"
        ).load_logs()["oracle"].astype(int).to_numpy()
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(mdpfuzz_failures))
    ax.plot(x, mdpfuzz_failures, label="MDPFuzz")
    ax.legend()
    ax.set_xlabel("# Iterations")
    ax.set_ylabel("# Failures")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title("Number of failures over test iterations")
    fig.tight_layout()
    fig.savefig(fuzzer_logs_path + "_failures.png")