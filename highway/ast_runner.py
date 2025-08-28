import torch

from hw_framework import (
    Framework,
    ENV_SEEDS,
    EXPERIMENT_SEEDS,
    FEATURES
)
from mdpfuzz.mdpfuzz import Fuzzer

import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(
        prog="AST Experiments' Runner",
        description="Evaluates a policy with one of the RL testing frameworks.",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["mdpfuzz", "ns", "qd", "rt"],
        required=True,
        help="RL testing framework.",
    )
    parser.add_argument(
        "--test_budget",
        default=5000,
        type=int,
        help="Total number of iterations."
    )
    parser.add_argument(
        "--seed_index",
        default=0,
        type=int,
        help="Seed index for the method (testing)."
    )
    parser.add_argument(
        "--log_folder",
        default="results",
        type=str,
        help="Name of the directory to log the results."
    )
    return parser.parse_args()


if __name__ == "__main__":
    torch.set_num_threads(1)
    from pathlib import Path

    args = parse_arguments()

    method = args.method  # type: str
    folder = args.log_folder  # type: str

    seed_index = args.seed_index  # type: int
    test_budget = args.test_budget  # type: int

    n = 1
    descriptors = ["action_entropy_mean", "length_spread"] # generic descriptors (unused)

    assert len(descriptors) == 2, len(descriptors)
    assert all([d in FEATURES for d in descriptors]), descriptors
    assert test_budget > 1000, "The test budget must be superior to the one for the initialization one (1000)."

    # experimental parameters
    init_budget = 1000
    cell_granularity = 50

    nb_iterations = 50
    k = 3
    novelty_threshold = 0.005

    # parameters / configurations from arguments
    assert (seed_index >= 0) and (seed_index < len(EXPERIMENT_SEEDS)), f"Seed index: {seed_index}..."
    seed = EXPERIMENT_SEEDS[seed_index]
    env_seeds = ENV_SEEDS[:n]

    print("=================================")
    print("method, seed inded (and thus seed), descriptors, env_seeds:")
    print(method, seed_index, seed, descriptors, env_seeds)
    print("=================================")

    results_fp = Path(f"{folder}/hw/{method}")
    results_fp.mkdir(parents=True, exist_ok=True)

    from executor import HighwayTestManager
    dqnagent_path = "saved_models/dqnagent/checkpoint-35000.tar"
    model = HighwayTestManager.load_policy(dqnagent_path)
    framework = Framework(
        seed,
        cell_granularity,
        features=FEATURES,
        descriptors=descriptors,
    )

    if method == "rt":
        framework.random_testing(
            model, env_seeds, test_budget, str(results_fp)
        )

    elif method == "qd":
        framework.test_policy(
            model, env_seeds, test_budget, init_budget, str(results_fp)
        )

    elif method == "ns":
        num_exec = test_budget * n
        population_size = test_budget // nb_iterations
        while (population_size * nb_iterations * n) < num_exec:
            print(f"Adjusting the population size to {population_size + 1} to at least reach the required total number of executions ({num_exec})...")
            population_size += 1

        print(f"NS LOG: pop_size: {population_size}, (actual) test_budget: {population_size * nb_iterations * n}")
        framework.novelty_search(
            model,
            env_seeds,
            population_size,
            nb_iterations,
            k,
            novelty_threshold,
            str(results_fp),
        )

    elif method == "mdpfuzz":
        from launch_mdpfuzz import MDPFuzzExecutor
        executor = MDPFuzzExecutor(
            sim_steps=800,
            env_seeds=env_seeds,
            log_path=str(results_fp)
        )
        fuzzer_logs_path = executor.fp + "_fuzzer"
        executor.config["rand_seed"] = seed
        fuzzer = Fuzzer(random_seed=seed, executor=executor, k=4, tau=0.1, gamma=0.01)
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

    else:
        print("Unknown method.")
