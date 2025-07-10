import torch
import json
import os
import sys
import time
from typing import Any, List, Tuple

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
        prog="Experiments' Runner",
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
    return parser.parse_args()


if __name__ == "__main__":
    torch.set_num_threads(1)
    from pathlib import Path

    args = parse_arguments()

    method = args.method  # type: str

    seed_index = args.seed_index  # type: int
    descriptors = args.descriptors  # type: Tuple[str, str]

    assert len(descriptors) == 2, len(descriptors)
    assert all([d in FEATURES for d in descriptors]), descriptors

    # experimental parameters
    test_budget = 5000
    init_budget = 1000
    cell_granularity = 50

    population_size, nb_iterations = 10, 5
    k = 3
    novelty_threshold = 0.005

    # parameters / configurations from arguments
    assert (seed_index >= 0) and (seed_index < len(EXPERIMENT_SEEDS)), f"Seed index: {seed_index}..."
    seed = EXPERIMENT_SEEDS[seed_index]

    print("=================================")
    print("method, seed inded (and thus seed), descriptors:")
    print(method, seed_index, seed, descriptors)
    print("=================================")

    results_fp = Path(f"results_new/hw/{method}")
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
            model, ENV_SEEDS, test_budget, str(results_fp)
        )

    elif method == "qd":
        framework.test_policy(
            model, ENV_SEEDS, test_budget, init_budget, str(results_fp)
        )

    elif method == "ns":
        framework.novelty_search(
            model,
            ENV_SEEDS,
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
            env_seeds=ENV_SEEDS,
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
