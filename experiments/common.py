from collections.abc import Iterable
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
ENV_SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
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


def bin_observation(obs: np.ndarray, edges: np.ndarray) -> np.ndarray:
    obs = obs.flatten()
    indices = np.zeros(obs.shape, dtype=int)

    for i, e in enumerate(edges):
        indices[i] = np.digitize(obs[i], e) - 1
        indices[i] = np.clip(indices[i], 0, len(e) - 2)

    return indices


def compute_bins_edges(low: np.ndarray, high: np.ndarray, num_bins: int) -> np.ndarray:
    edges = [np.linspace(start=l, stop=h, num=(num_bins + 1)) for l, h in zip(low, high)]
    return np.array(edges)


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
    if use_case not in ["Bipedal Walker", "Highway", "Lunar Lander"]:
        raise ValueError()

    if use_case == "Bipedal Walker":
        edges = np.load(f"grid/bw/0_300_edges.npy")
        if descriptors is not None:
            return edges[descriptors]
        else:
            return edges

    if use_case == "Highway":
        return np.load(f"grid/hw/edges.npy")

    else:
        return np.load(f"grid/ll/0_1000_xedges.npy"), np.load(f"grid/ll/0_1000_yedges.npy")


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
                print("=======================================", file=sys.stderr)
                print(f'FILENAME: {filename}.', file=sys.stderr)
                print(f'ERROR_TXT_LOG_PROCESSING for "{line}".', file=sys.stderr)
                print("=======================================", file=sys.stderr)

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
    filepaths.append(f"{filepath}_data.csv")

    if not np.all([os.path.exists(fp) for fp in filepaths]):
        raise FileNotFoundError("One of the required result file is missing.")

    result = {
        k: np.loadtxt(f"{filepath}_{k}.txt", delimiter=",")
        for k in ["inputs", "behaviors", "cells"]
    }

    if os.path.exists(f"{filepath}_logs.txt"):
        logs = process_txt_log(f"{filepath}_logs.txt")[0]
    elif os.path.exists(f"{filepath}_logs.csv"):
        logs = pd.read_csv(f"{filepath}_logs.csv")
    else:
        warnings.warn(f"No logs file found for {filepath}.")
        logs = pd.DataFrame()
    result["logs"] = logs

    try:
        result["data"] = pd.read_csv(f"{filepath}_data.csv")
    except pd.errors.EmptyDataError:
        result["data"] = pd.DataFrame()

    try:
        with open(f"{filepath}_config.json", "r") as f:
            result["config"] = json.load(f)
    except:
        result["config"] = {}
        warnings.warn(f"No configuration found at {filepath}.", RuntimeWarning)

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
    if not os.path.isdir(results_folder):
        warnings.warn(f"Folder \"{results_folder}\" not found.", RuntimeWarning)
        return []

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


##############################################################################
########################## RESULTS STORAGE/LOADING ###########################


def dump_dictionary(d: Dict[str, Union[np.ndarray, List]], filename: str):
    dict_to_dump = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            dict_to_dump[k] = v.tolist()
        else:
            dict_to_dump[k] = v
    with open(f'{filename.split(".")[0]}.json', "w") as f:
        f.write(json.dumps(dict_to_dump))


def dump_dictionaries(
    dict_list: List[Dict[str, Union[np.ndarray, List]]], filenames: List[str]
):
    assert len(filenames) == len(dict_list)

    for d, name in zip(dict_list, filenames):
        dump_dictionary(d, name)


def dump_results(dict_list: List[Dict[str, Iterable]], filenames: List[str]):
    assert len(filenames) == len(dict_list)

    for d, name in zip(dict_list, filenames):
        dict_to_dump = {}
        for k, v in d.items():
            # assert len(v) == 3, f'Three statistics are expected (found {len(v)}).'
            # assert np.all([len(t) == v[0] for t in v]), 'Three statistics of a result is malformed.'
            if isinstance(v, np.ndarray):
                dict_to_dump[k] = v.tolist()
            else:
                # must be a list of numpy arrays
                assert np.all([isinstance(t, np.ndarray) for t in v])
                dict_to_dump[k] = [t.tolist() for t in v]
        with open(f'{name.split(".")[0]}.json', "w") as f:
            f.write(json.dumps(dict_to_dump))


def load_result(filepath: str):
    fp = f"{filepath}.json"
    if not os.path.exists(fp):
        warnings.warn(f"File {fp} not found.", RuntimeWarning)
        return {}
    try:
        with open(fp, "r") as f:
            d = json.load(f)
    except:
        d = {}
    finally:
        for k in d.keys():
            v = d[k]
            assert isinstance(v, List)
            if isinstance(v[0], List):
                new_v = [np.array(l) for l in v]
            else:
                new_v = np.array(v)
            d[k] = new_v
        return d


def load_results(
    data_folder: str = "data/",
    keys: List[str] = ["rq1", "bs_cov", "fbs_cov", "obs_cov", "fobs_cov"],
):
    """
    Returns all the results per use-case-folder as a list of dictionaries ordered by use-case.
    Therefore, every list has a length of |use-cases|, where the elements are dictionaries of the results (whose keys are the names of the framework).
    """
    if not data_folder.endswith("/"):
        data_folder += "/"
    use_cases = [d for d in os.listdir(data_folder) if os.path.isdir(f"{data_folder}{d}")]
    use_cases.sort()
    all_results = [[] for _ in range(len(keys))]
    for k in range(len(keys)):
        for case in use_cases:
            all_results[k].append(load_result(f"{data_folder}{case}/{keys[k]}"))
    return all_results


def assemble_n_results(
        data_folders: List[str],
        methods=["MAP-Elites", "MDPFuzz", "Novelty Search", "Random Testing"],
        use_cases=["Bipedal Walker", "Highway", "Lunar Lander"],
        suffix="MAE+TS",
        metric="bs_cov"
    ):
    assert metric in ["rq1", "bs_cov", "fbs_cov", "obs_cov", "fobs_cov"]

    # shape (env_seeds, use cases)
    all_results = [load_results(data_folder, [metric])[0] for data_folder in data_folders]
    # print(len(all_results), [len(r) for r in all_results])
    assembled_data = [] # {k: [] for k in methods} for _ in range(use_cases)]
    # so we have to re-organize the data per use case
    for i, case in enumerate(use_cases):
        case_results = [r[i] for r in all_results]
        # print(f"Found {len(case_results)} for use case {case}.")
        case_data = {k: [] for k in methods}
        for f, d in enumerate(case_results):
            # print(case, list(d.keys()))
            for k in methods:
                if k in d:
                    case_data[k].append(d[k])
                else:
                    # print(f"{k} not found in data folder \"{data_folders[f]}\" for case {case}...")
                    if k in ["MAP-Elites", "Novelty Search"]:
                        suffixed_key = f"{k} {suffix}"
                        if suffixed_key in d:
                            case_data[k].append(d[suffixed_key])
                            # print(f"Added suffixed data for {k} in {case}.")
                        else:
                            print(f"No data for {k} in {case} (folder \"{data_folders[f]}\").")
                    else:
                        print(f"No data for {k} in {case} (folder \"{data_folders[f]}\").")

        # print(f"============= SUMMARY OF DATA FOR CASE {case} ================")
        # for k, v in case_data.items():
        #     print(k, f"{len(v)} results.")

        # print("===============================================================")
        assembled_data.append(case_data)
    return assembled_data


def concatenate_results(d1: Dict, d2: Dict, size=5000):
    keys = list(d1.keys())
    for k in keys:
        v = d1[k]
        if isinstance(v, np.ndarray):
            d1[k] = np.concatenate([d1[k], d2[k]], axis=0)[:size]
            print(f"Key '{k}' adjusted to size {len(d1[k])}.")
        if isinstance(v, pd.DataFrame):
            d1[k] = pd.concat([d1[k], d2[k]])[:size]
            print(f"Key '{k}' adjusted to size {len(d1[k])}.")
        if isinstance(v, List):
            assert all([isinstance(arr, np.ndarray) for arr in v]), f"Not all data in the list (entry '{k}') is numpy arrays..."
            d1[k] = [
                np.concatenate([d1[k][i], d2[k][i]], axis=0)[:size] for i in range(len(d1[k]))
            ]
            print(f"Key '{k}' adjusted to a list of {len(d1[k])} elements.")
            print(f"Their respective lengths are '{[len(arr) for arr in d1[k]]}'.")
    return d1


def save_result(data: Dict, filepath: str, int_input: bool = False):
    """Save a typical results dictionary. As done by the frameworks, the current date is appended to `filepath`."""
    if int_input:
        input_fmt = "%1.0f"
    else:
        input_fmt = "%.18e"

    keys = list(data.keys())
    t = time.time()
    path = f"{filepath}{t}"
    for k in keys:
        v = data[k]

        if k == "inputs":
            np.savetxt(f"{path}_{k}.txt", v, fmt=input_fmt, delimiter=",")

        # behaviors and cells
        elif isinstance(v, np.ndarray):
            np.savetxt(f"{path}_{k}.txt", v, delimiter=",")

        elif isinstance(v, List):
            assert all([isinstance(arr, np.ndarray) for arr in v])
            [np.savetxt(f"{path}_{k}_{i}.txt", v[i], delimiter=",") for i in range(len(v))]

        elif isinstance(v, pd.DataFrame):
            v.to_csv(f"{path}_{k}.csv", index=(None if v.empty else 0))

        elif isinstance(v, Dict):
            with open(f"{path}_{k}.json", "w") as f:
                json.dump(v, f)


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


#################################################################################################
################################## PERFORMANCE AND METRICS ######################################


def isin(array: np.ndarray, element: np.ndarray) -> Union[bool, np.ndarray]:
    return (array[:, None] == element).all(axis=-1).any(axis=1)


def isin_index(array: np.ndarray, element: np.ndarray):
    return next(
        (
            i
            for i, j in enumerate((array[:, None] == element).all(axis=-1).any(axis=1))
            if j
        ),
        None,
    )

