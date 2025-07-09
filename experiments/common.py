import json
import os
import sys
import time
import warnings
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from sb3_contrib import TQC
from stable_baselines3.ppo.ppo import PPO

EXPERIMENT_SEEDS = [2021, 42, 2023, 20, 0, 10, 4, 2006, 512, 1453]
POP_SIZES = [100, 250, 500]
ITERATIONS = [50, 20, 10]
ENV_SEEDS = [0, 1, 2]
MEASURES = [
    "length_mean",
    "length_std",
    "length_spread",
    "action_std",
    "action_entropy_mean",
    "action_entropy_argmax",
    "action_divergence",
]
# TODO: currently under test
MEAS_STR_INDICES = [
    ["action_entropy_mean", "length_mean"],
    ["action_entropy_mean", "length_spread"],
    ["action_entropy_argmax", "length_mean"],
    ["action_entropy_argmax", "length_spread"],
    ["action_divergence", "length_mean"],
    ["action_divergence", "length_spread"],
]
MEAS_INDICES = [
    [4, 0],
    [4, 2],
    [5, 0],
    [5, 2],
    [6, 0],
    [6, 2],
]


###############################################################################################
################################## CELL AND GRID HELPERS ######################################


def compute_cell(
    behavior: np.ndarray, xedges: np.ndarray, yedges: np.ndarray
) -> np.ndarray:
    cell = []
    for b, v in zip([xedges, yedges], behavior):
        if v < b[1]:
            cell.append(0)
        elif v >= b[-2]:
            cell.append(len(b) - 1)
        else:
            cell.append(np.argmax(v < b) - 1)
    return np.array(cell)


def compute_cells(
    behaviors: np.ndarray, xedges: np.ndarray, yedges: np.ndarray
) -> np.ndarray:
    cells = []
    for behavior in behaviors:
        cell = []
        for b, v in zip([xedges, yedges], behavior):
            if v < b[1]:
                cell.append(0)
            elif v >= b[-2]:
                cell.append(len(b) - 1)
            else:
                cell.append(np.argmax(v < b) - 1)
        cells.append(np.array(cell))
    return np.array(cells)


def compute_grid_edges(
    bins: int = 50,
    mins: np.ndarray = None,
    maxs: np.ndarray = None,
    behaviors: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns the cell edges structure used by the regular grid.
    The edges are bins + 1 numpy arrays where the first and last value are the minimum and maximum, respectively.
    Besides, the extrema can be provided instead of the behaviors.
    """
    # I don't know how to handle this... the function's signature is soo bad
    if behaviors is None:
        assert (mins is not None) and (
            maxs is not None
        ), "Either behaviors or extrema have to be provided."

    if mins is None:
        mins = np.min(behaviors, axis=0)
    if maxs is None:
        maxs = np.max(behaviors, axis=0)

    edges = np.array(
        [np.linspace(min, max, num=(bins + 1)) for min, max in zip(mins, maxs)]
    )
    return edges, mins, maxs


def compute_extrema(df: pd.DataFrame, columns: List[str]):
    return pd.DataFrame(
        data=[df[columns].min().to_numpy(), df[columns].max().to_numpy()],
        columns=columns,
    )


def get_bin_edges(df: pd.DataFrame, measures: List[str], num_bins: int = 50):
    """Returns num_bins + 1 edges."""
    return np.array(
        [
            np.linspace(*df[meas].to_list(), num=(num_bins + 1), endpoint=True)
            for meas in measures
        ]
    )


def get_histogram(
    behaviors: np.ndarray, xedges: np.ndarray, yedges: np.ndarray
) -> np.ndarray:
    """Returns the histogram of the behaviors (i.e., the behavior points distribution in the space)."""
    # the issue is that some behaviors found during the search might be outside of the edges.
    # this is handled by the archive, but not here.
    return np.histogram2d(behaviors[:, 0], behaviors[:, 1], bins=(xedges, yedges))[0]


def get_expert_bin_edges(use_case: str, descriptors: np.ndarray = None) -> np.ndarray:
    if use_case not in ["Bipedal Walker", "Lunar Lander"]:
        raise ValueError()

    if use_case == "Bipedal Walker":
        edges = np.load(f"grid/bw/0_300_edges.npy")
        return edges[descriptors]

    else:
        np.load(f"grid/ll/0_1000_xedges.npy"), np.load(f"grid/ll/0_1000_yedges.npy")


def get_measures_edges(
    use_case: str, num_bins: int = 50, model_name: str = "DQNAgent-35000"
) -> List[np.ndarray]:
    if use_case not in ["Bipedal Walker", "Highway", "Lunar Lander"]:
        raise ValueError()

    if use_case == "Bipedal Walker":
        df = pd.read_csv("../experiments/grid/bw/measures.csv")
    elif use_case == "Lunar Lander":
        df = pd.read_csv("../experiments/grid/ll/measures.csv")
    else:
        df = pd.read_csv("measures.csv")
        if model_name not in df["model_name"].tolist():
            warnings.warn(
                "Model name for the measures' extrema in Highway not found: using the default model name value instead...",
                UserWarning,
            )
            model_name = "DQNAgent-35000"
        df = df.loc[df.model_name == model_name]

    return get_bin_edges(df, measures=MEASURES, num_bins=num_bins)


def compute_cell_filling(
    behaviors: np.ndarray,
    descriptor_indices_list: List[Tuple[int, int]],
    edges: List[Tuple[np.ndarray, np.ndarray]],
):
    """Compute the (grid) cells of 2 behavior points in `behaviors` given pairs of the behavior indices."""

    indices_arr = np.array(descriptor_indices_list)
    assert len(indices_arr) == len(edges)

    return [
        np.apply_along_axis(
            func1d=lambda x: compute_cell(x[idx], xedges, yedges), axis=1, arr=behaviors
        )
        for idx, (xedges, yedges) in zip(indices_arr, edges)
    ]


#################################################################################################
################################## RESULTS READING HELPERS ######################################


def process_txt_log(filename: str):
    """Reads a log file (.txt, lines of "key:value,") of an execution and returns the results as a DataFrame (a row describes the log of an iteration)."""
    if not filename.endswith(".txt"):
        filename += ".txt"
    assert os.path.isfile(filename)

    t0 = time.time()
    dicts = []
    with open(filename, "r") as f:
        for line in f.readlines():
            # dirty...
            try:
                splits = line.split(",")
                str_dict = dict(s.strip().split(":") for s in splits)
                dicts.append({k: float(v) for k, v in str_dict.items()})
            except:
                print(f'ERROR_TXT_LOG_PROCESSING for "{line}".', file=sys.stderr)

    df = pd.DataFrame.from_records(dicts)
    if "failure_prob" in df.columns:
        df["oracle"] = df["failure_prob"].astype(bool)
    process_time = time.time() - t0
    return df, process_time


def retrieve_result(
    filepath: str, **kwargs
) -> Dict[str, Union[np.ndarray, pd.DataFrame]]:
    """
    Returns the results of a testing methodology at ``filepath`` as a dictionary of:
    - 3 numpy arrays (inputs, behaviors and cells).
    - DataFrame of the logs.
    - DataFrame of the internal data of the Framework class.
    - Dictionary of the (experimental) configuration.

    Raise an error if any of the expected file is missing.

    Kwargs:
    - "include_final_states": the dictionary has the latter at the key `final_states`.
    - "is_ns": the dictionaries have particular NS logs at the key `ns_logs`.
    - "include_expert_behaviors": the dictionaries have the latter at the key `expert_behaviors`.
    """
    filepaths = [f"{filepath}_{k}.txt" for k in ["inputs", "behaviors", "cells"]]
    filepaths.append(f"{filepath}_logs.txt")
    filepaths.append(f"{filepath}_data.csv")

    if not np.all([os.path.exists(fp) for fp in filepaths]):
        raise FileNotFoundError("One of the required result file is missing.")

    result = {
        k: np.loadtxt(f"{filepath}_{k}.txt", delimiter=",")
        for k in ["inputs", "behaviors", "cells"]
    }
    result["logs"] = process_txt_log(f"{filepath}_logs.txt")[0]
    try:
        result["data"] = pd.read_csv(f"{filepath}_data.csv")
    except pd.errors.EmptyDataError:
        result["data"] = pd.DataFrame()

    try:
        with open(f"{filepath}_config.json", "r") as f:
            result["config"] = json.load(f)
    except:
        result["config"] = {}
        warnings.warn(f"No configuration found at {filepath}.")

    include_final_states = kwargs.get("include_final_states", False)
    if include_final_states:
        try:
            seeds = result["config"]["env_seeds"]
            result["final_states"] = [
                np.loadtxt(f"{filepath}_final_states_{seed}.txt", delimiter=",")
                for seed in seeds
            ]
        except Exception as e:
            warnings.warn(f"Fetching of final states failed: {e}")

    include_expert_behaviors = kwargs.get("include_expert_behaviors", False)
    if include_expert_behaviors:
        try:
            seeds = result["config"]["env_seeds"]
            result["expert_behaviors"] = [
                np.loadtxt(f"{filepath}_expert_behaviors_{seed}.txt", delimiter=",")
                for seed in seeds
            ]
        except Exception as e:
            warnings.warn(f"Fetching of expert behaviors failed: {e}")

    is_ns = kwargs.get("is_ns", False)
    if is_ns:
        ns_logs_fp = f"{filepath}_ns_logs.txt"
        if os.path.exists(ns_logs_fp):
            result["ns_logs"] = process_txt_log(ns_logs_fp)[0]

    return result


def read_results_from_folder(results_folder: str, **kwargs) -> List[Dict]:
    """
    Returns all the results found in ``results_folder`` as a list of dictionaries.

    Kwargs:
        - "include_final_states": the dictionaries have the latter at the key `final_states`.
        - "is_ns": the dictionaries have particular NS logs at the key `ns_logs`.
        - "include_expert_behaviors": the dictionaries have the latter at the key `expert_behaviors`.
    """
    assert os.path.isdir(results_folder)
    if not results_folder.endswith("/"):
        results_folder += "/"

    results_filepaths = [
        results_folder + fp
        for fp in set(f.split("_")[0] for f in os.listdir(results_folder))
    ]
    config = kwargs.get("config", {})
    name_key = kwargs.get("name_key", None)

    dicts = []

    for fp in results_filepaths:
        config_fp = fp + "_config.json"
        if os.path.exists(config_fp):
            with open(config_fp, "r") as f:
                f_config: dict = json.load(f)

            if not all(f_config.get(k) == config[k] for k in config.keys()):
                continue
            if (name_key is not None) and (not name_key in f_config["name"]):
                continue
        try:
            d = retrieve_result(fp, **kwargs)
            dicts.append(d)
        except Exception as e:
            print(e)
    return dicts


#################################################################################################
####################################### MODELS LOADING ##########################################


def load_lunar_lander_model():
    """Loads the model under test."""
    custom_objects = {
        "learning_rate": 0.0,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0,
    }
    return PPO.load(
        "rl-trained-agents/ppo/LunarLander-v2_1/LunarLander-v2.zip",
        custom_objects=custom_objects,
        device="cpu",
    )


def load_bipedal_walker_model():
    return TQC.load(
        "rl-trained-agents/tqc/BipedalWalkerHardcore-v3_1/BipedalWalkerHardcore-v3.zip",
        custom_objects={},
        kwargs={"seed": 0, "buffer_size": 1},
        device="cpu",
    )
