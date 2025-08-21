import warnings
import torch
from typing import Tuple

from bw_framework import BWFramework

from common import ENV_SEEDS, EXPERIMENT_SEEDS, MEASURES, load_bipedal_walker_model

import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(
        prog="Experiments' Runner for BW with Expert Behaviors.",
        description="Evaluates a policy with one of the RL testing frameworks.",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["ns", "qd"],
        required=True,
        help="RL testing framework.",
    )
    parser.add_argument(
        "--descriptors",
        type=int,
        required=True,
        nargs=2,
        help="Pair of indices of the expert descriptors.",
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

    # input parameters
    method = args.method  # type: str
    seed_index = args.seed_index  # type: int
    expert_indices = args.descriptors  # type: Tuple[int, int]
    folder = args.log_folder  # type: str

    # fixed parameters
    use_case = "bw"
    descriptors = ["action_entropy_mean", "length_spread"] # generic descriptors (unused)
    test_budget = 5000
    n = 1

    assert len(descriptors) == 2, len(descriptors)
    assert all([d in MEASURES for d in descriptors]), descriptors
    assert test_budget > 1000, "The test budget must be superior to the one for the initialization one (1000)."

    assert n > 0, "Number of seeds for the environments must be superior to 0."
    if n > 10:
        n = 10
        warnings.warn(
            f"The maximum number of seeds for the environment is 10 (received '{n}'). Set to 10.",
            UserWarning
        )

    assert len(expert_indices) == 2, len(expert_indices)
    assert all([(d >= 0) and (d < 12) for d in expert_indices]), f"Invalid expert indices: {expert_indices}."

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
    print("use case, method, seed inded (and thus seed), descriptors, env_seeds:")
    print(use_case, method, seed_index, seed, descriptors, env_seeds)
    print("=================================")
    print("expert indices:")
    print(expert_indices)
    print("=================================")

    results_fp = Path(f"{folder}/{use_case}/{method}")
    results_fp.mkdir(parents=True, exist_ok=True)

    framework = BWFramework(
        seed,
        cell_granularity,
        features=MEASURES,
        descriptors=descriptors,
        expert_indices=expert_indices
    )
    model = load_bipedal_walker_model()

    if method == "qd":
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

    else:
        print("Unknown method.")
