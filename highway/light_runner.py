import warnings
import torch
import json
import os
import sys
import time
from typing import Any, List, Tuple

from light_framework import (
    Framework,
    ENV_SEEDS,
    EXPERIMENT_SEEDS,
    FEATURES
)

import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(
        prog="Experiments' Runner",
        description="Evaluates a policy with one of the RL testing frameworks.",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["qd", "rt"],
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

    method = args.method  # type: str

    seed_index = args.seed_index  # type: int
    n = args.env_seeds  # type: int
    test_budget = args.test_budget  # type: int

    descriptors = args.descriptors  # type: Tuple[str, str]
    folder = args.log_folder  # type: str

    assert len(descriptors) == 2, len(descriptors)
    assert all([d in FEATURES for d in descriptors]), descriptors
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

    else:
        print("Unknown method.")
