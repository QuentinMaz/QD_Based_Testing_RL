import time
from typing import Any, List, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from mdpfuzz.executor import Executor
from mdpfuzz.logger import FuzzerLogger
from mdpfuzz.mdpfuzz import Fuzzer
from executor import HighwayTestManager


class MDPFuzzExecutor(Executor):

    def __init__(self, sim_steps, env_seed) -> None:
        super().__init__(sim_steps, env_seed)
        self.executor = HighwayTestManager(seeds=[env_seed])

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

    def execute_policy(
        self, input: np.ndarray, policy: Any
    ) -> Tuple[float, bool, np.ndarray, float]:
        t0 = time.time()

        failure, obs_seq, action_seq, reward_seq, trajectory, frames = self.executor._record_execution(
            policy, input, record=False, deterministic=True
        )

        return (
            sum(reward_seq),
            failure,
            obs_seq,
            time.time() - t0,
        )



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
    executor = MDPFuzzExecutor(sim_steps=800, env_seed=0)

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
        saving_path="fuzzer",
        local_sensitivity=True, # don't re-run for computing the sensitivity
        exp_name="Highway",
        light_pool=True, # don't log the inputs
        save_logs_only=True # don't save evaluated inputs
    )

    mdpfuzz_failures = accumulate_failures(
        FuzzerLogger("fuzzer_logs.txt").load_logs()["oracle"].astype(int).to_numpy()
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
    fig.savefig("fuzzer_failures.png")