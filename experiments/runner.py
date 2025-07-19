import warnings
import torch
import json
import os
import sys
import time
from typing import Any, List, Tuple

from bw_framework import BWFramework, BWExecutor
from ll_framework import LLExecutor, LLFramework

from common import ENV_SEEDS, EXPERIMENT_SEEDS, MEASURES, load_lunar_lander_model, load_bipedal_walker_model
from mdpfuzz.mdpfuzz import Fuzzer

import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(
        prog="Experiments' Runner",
        description="Evaluates a policy with one of the RL testing frameworks.",
    )
    # testin task parameters
    parser.add_argument(
        "--use_case",
        type=str,
        choices=["bw", "ll"],
        help="Use case.",
        required=True,
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
        "--descriptors",
        type=str,
        required=True,
        nargs=2,
        help="Descriptor pair for the generic behavior space.",
    )
    parser.add_argument(
        "--seed_index",
        default=0,
        type=int,
        help="Seed index for the method (testing)."
    )
    parser.add_argument(
        "--env_seeds",
        default=3,
        type=int,
        help="Number of seeds for the environments. At least 1, and up to 10."
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

    use_case = args.use_case  # type: str
    method = args.method  # type: str

    seed_index = args.seed_index  # type: int
    n = args.env_seeds  # type: int
    test_budget = args.test_budget  # type: int

    descriptors = args.descriptors  # type: Tuple[str, str]
    folder = args.log_folder  # type: str

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

    results_fp = Path(f"{folder}/{use_case}/{method}")
    results_fp.mkdir(parents=True, exist_ok=True)

    if use_case == "bw":
        framework = BWFramework(
            seed,
            cell_granularity,
            features=MEASURES,
            descriptors=descriptors,
        )
        model = load_bipedal_walker_model()
    else:
        framework = LLFramework(
            seed,
            cell_granularity,
            features=MEASURES,
            descriptors=descriptors,
        )
        model = load_lunar_lander_model()

    if method == "rt":
        framework.random_testing(
            model, env_seeds, test_budget, str(results_fp)
        )

    elif method == "mdpfuzz":
        if use_case == "ll":
            executor = LLExecutor(seed, env_seeds, log_path=str(results_fp))
            exp_name = "Lunar Lander"
        else:
            executor = BWExecutor(seed, env_seeds, log_path=str(results_fp))
            exp_name = "Bipedal Walker"

        fuzzer_logs_path = executor.fp + "_fuzzer"
        fuzzer = Fuzzer(random_seed=seed, executor=executor, k=4, tau=0.1, gamma=0.01)
        fuzzer.fuzzing_no_coverage(
            n=init_budget,
            test_budget=test_budget,  # 2*n will be removed since we assume that test_budget is the TOTAL budget
            policy=model,
            saving_path=fuzzer_logs_path,
            local_sensitivity=True,  # don"t re-run for computing the sensitivity
            exp_name=exp_name,
            light_pool=True,  # don"t log the inputs
            save_logs_only=True,  # don"t save evaluated inputs
        )
        executor.clean()

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

    else:
        print("Unknown method.")
