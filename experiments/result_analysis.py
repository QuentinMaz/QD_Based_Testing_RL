import json
import os
from pathlib import Path
import warnings
from typing import Dict, Iterable, List, Tuple, Union

from matplotlib.legend import Legend
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import torch
from bw_framework import EXPERT_INDICES
import matplotlib
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors
from collections.abc import Iterable

from common import (
    MEAS_INDICES,
    MEAS_STR_INDICES,
    MEASURES,
    assemble_n_results,
    bin_observation,
    compute_cell_filling,
    dump_results,
    get_expert_bin_edges,
    get_measures_edges,
    read_results_from_folder,
)

FAULT_LABEL = "#Faults"
AXIS_LABEL_FONTSIZE = 17
TITLE_LABEL_FONTSIZE = 18

USE_CASES = ["Bipedal Walker", "Highway", "Lunar Lander"]

#################################################################################################################################
############################################################## HELPERS ##########################################################


def filter_data(
    boolean_list: List[np.ndarray], points_list: List[np.ndarray]
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Accumulates the number of points which are both unique and filtered by a given boolean mask.

    For instance, given the mask [0 0 1 1 0 0 1] and the points [1 2 3 3 4 5 5]:
        - The merged mask of the relevant points is [0 0 1 0 0 0 1].
        - Therefore, the result of their accumulation is [0 0 1 1 1 1 2].

    It returns a list of tuples as:
    1. Unique filtered points
    2. Their indices in the original points
    3. The corresponding mask and
    4. The accumulation of the latter

    Therefore, 3 and 4 have the same length of the points while the 1 and 2 share a length which is lower or equal to the length of the points.
    """
    n = len(boolean_list)
    if points_list is not None:
        assert len(points_list) == n
        for bl, pl in zip(boolean_list, points_list):
            assert len(bl) == len(pl)

    def is_in_list(list: List[np.ndarray], element: np.ndarray):
        for e in list:
            if np.array_equal(e, element):
                return True
        return False

    results = []
    for oracle_mask, points in zip(boolean_list, points_list):
        if len(points.shape) == 1:
            points = points[:, None]
        data = []
        data_indices = []
        data_mask = []
        accumulator = []
        for i in range(len(oracle_mask)):
            oracle, point = oracle_mask[i], points[i]
            # only updates the result lists if unseen faulty
            if oracle and (not is_in_list(data, point)):
                data.append(point)
                data_indices.append(i)
                data_mask.append(oracle)  # True
            else:
                data_mask.append(False)

        # if I handle i == 0 before, I can also accumulate during the previous for loop...
        counter = 0
        for b in data_mask:
            counter += int(b)
            accumulator.append(counter)

        data = np.array(data)
        data_indices = np.array(data_indices)
        data_mask = np.array(data_mask)
        accumulator = np.array(accumulator)

        # print('oracle mask:', oracle_mask)
        # print('mask to accumulate:', data_mask)
        # print('indices in the points:', data_indices)
        # print('-----------------------------------------------------------')
        results.append((data, data_indices, data_mask, accumulator))

    return results


def accumulate(
    data: List[np.ndarray],
) -> List[np.ndarray]:
    acc_list = []
    for arr in data:
        cpt = 0
        acc = []
        if arr.dtype != bool:
            raise ValueError("Input list expected to be of boolean type.")

        for x in arr:
            cpt += int(x)
            acc.append(cpt)
        acc_list.append(np.array(acc))

    return acc_list


def accumulate_uniques(
    data: List[np.ndarray],
) -> List[np.ndarray]:
    """
    Accumulates the number of unique points in each numpy array.
    """
    acc_list = []
    for arr in data:
        seen = set()
        acc = []
        if len(arr) == 0:
            raise ValueError("Empty list in `data`.")

        if isinstance(arr[0], Iterable):
           it =  map(tuple, arr)
        else:
            it = arr

        for point in it:
            seen.add(point)
            acc.append(len(seen))

        # Convert to numpy array if desired
        acc_list.append(np.array(acc))
    return acc_list


def knn_dists(data: np.ndarray, k: int) -> np.ndarray:
    """
    Distance to k nearest neighbours.
    This is the sparseness criterion of the original novelty search paper.
    Intuitively, if the average distance to a given point's nearest
    neighbors is large then it is in a sparse area; it is in a dense region if the average
    distance is small.
    It returns all the mean distances.
    """
    if not isinstance(data, np.ndarray):
        return np.nan
    u_data = np.unique(data, axis=0)
    # nb_totals = data.shape[0]
    # nb_uniques = u_data.shape[0]
    # print(f'{nb_uniques} out of {nb_totals} points are unique')
    if len(u_data.shape) == 1:
        u_data = u_data.reshape(-1, 1)
    neighbors = NearestNeighbors(n_neighbors=k).fit(u_data)
    distances, _ = neighbors.kneighbors()
    return np.mean(distances, axis=1)


#####################################################################################
################################### Novelty Search data analysis ####################


def ns_analysis(ns_res: List[Dict]):
    """Updates each dictionary in the input list, assuming the results come from NS runs.

    Parameters
    ----------
    ns_res : List[Dict]
        The dictionaries to update. They must contain the entries "config" and "ns_logs".

    Returns
    -------
    List[Dict]
        The input data, with the new entries "archive_sizes" and "archive_sparsenesses", and the already existing one "config" updated.
    """
    for res in ns_res:
        # adds the popsize and nov_threshold parameters to the name
        popsize = res["config"]["pop_size"]
        t = res["config"]["nov_threshold"]
        ns_logs_df: pd.DataFrame = res["ns_logs"]

        res["config"]["name"] += f" popsize/threshold = ({popsize}, {t})"
        res["archive_sizes"] = [
            s for s in ns_logs_df["archive_size"].to_numpy() for _ in range(popsize)
        ]
        res["archive_sparsenesses"] = [
            s
            for s in ns_logs_df["archive_sparseness"].to_numpy()
            for _ in range(popsize)
        ]
    return ns_res


def plot_ns_analysis(ns_res: List[Dict]):
    def compute_statistics(
        data,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Helper that computes statistical results from a set of results."""
        if not isinstance(data, np.ndarray):
            data: np.ndarray = np.array(data)
        x = np.arange(data.shape[1])
        y = np.median(data, axis=0)
        perc_25 = np.percentile(data, 25, axis=0)
        perc_75 = np.percentile(data, 75, axis=0)
        return y, perc_25, perc_75, x

    use_cases: List[str] = np.unique([d["config"]["use_case"] for d in ns_res]).tolist()
    nb_use_cases = len(use_cases)
    methods_names: List[str] = np.unique([d["config"]["name"] for d in ns_res]).tolist()
    cmap = plt.cm.jet  # plt.cm.jet is a LinearSegmentedColormap
    rgba_colors = [cmap(i) for i in np.linspace(0, 1, len(methods_names))]
    colors_dict = {n: c for n, c in zip(methods_names, rgba_colors)}

    fig, axs = plt.subplots(nrows=nb_use_cases, ncols=2, figsize=(15, 7 * nb_use_cases))
    if nb_use_cases == 1:
        axs = [axs]

    axs[0][0].set_title("Archive sizes")
    axs[0][1].set_title("(distinct) #Faults")
    for u in range(nb_use_cases):
        axs[u][0].set_ylabel(use_cases[u])
        axs[u][0].set_xlabel("#Iterations")
        axs[u][1].set_xlabel("#Iterations")

        for method_name in methods_names:
            sub_results = [
                d
                for d in ns_res
                if (d["config"]["use_case"] == use_cases[u])
                and (d["config"]["name"] == method_name)
            ]
            if len(sub_results) == 0:
                print(
                    f"No result found for use-case {use_cases[u]} and methodology {method_name}"
                )
                continue
            color = colors_dict[method_name]
            label = method_name

            sizes = [d["archive_sizes"] for d in sub_results]
            y, perc_25, perc_75, x = compute_statistics(sizes)
            axs[u][0].plot(x, y, color=color, label=label)
            axs[u][0].fill_between(
                x, perc_25, perc_75, alpha=0.15, linewidth=0, color=color
            )

            inputs = [d["inputs"] for d in sub_results]
            oracles = [d["logs"]["oracle"].to_numpy() for d in sub_results]
            nb_faults = compute_evolution_fault_triggering_inputs(inputs, oracles)
            y, perc_25, perc_75, x = compute_statistics(nb_faults)
            color = colors_dict[method_name]
            label = method_name
            axs[u][1].plot(x, y, color=color, label=label)
            axs[u][1].fill_between(
                x, perc_25, perc_75, alpha=0.15, linewidth=0, color=color
            )
    legend = axs[-1][0].legend(title="Testing Methodology")
    legend_frame = legend.get_frame()
    legend_frame.set_facecolor("0.9")
    legend_frame.set_edgecolor("0.9")

    fig.tight_layout()
    return (fig, axs)


##############################################################################################
#################################### Plotting parameters #####################################


def color_data(
    data: List[Dict],
) -> Tuple[List[str], List[str], Dict[str, Tuple[float]]]:
    try:
        use_cases: List[str] = np.unique([d["config"]["use_case"] for d in data]).tolist()
        methods_names: List[str] = np.unique([d["config"]["name"] for d in data]).tolist()
    except:
        use_cases = []
        methods_names = set(sum([list(d.keys()) for d in data], []))
    cmap = plt.cm.jet  # type: matplotlib.colors.LinearSegmentedColormap
    rgba_colors = [cmap(i) for i in np.linspace(0, 1, len(methods_names))]
    colors_dict = {n: c for n, c in zip(methods_names, rgba_colors)}
    return use_cases, methods_names, colors_dict


##############################################################################################
############ RQ1: How many (distinct) fault-triggering inputs do frameworks find? ############


def compute_evolution_fault_triggering_inputs(
    inputs: List[np.ndarray], oracles: List[np.ndarray]
):
    """
    Computes the evolution of the number of fault-triggering inputs found for each set of results.
    As such, the ``inputs`` and ``oracles`` lists are expected to be of the same length, every ith data belonging to a particular testing methodology.
    Redundant inputs are not considered.
    """
    return [res[-1] for res in filter_data(oracles, inputs)]


def compute_rq1_results(
    data: List[Dict], only_uniques: bool = True
) -> List[Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    use_cases: List[str] = np.unique([d["config"]["use_case"] for d in data]).tolist()
    methods_names: List[str] = np.unique([d["config"]["name"] for d in data]).tolist()

    def compute_statistics(data) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not isinstance(data, np.ndarray):
            data: np.ndarray = np.array(data)
        y = np.median(data, axis=0)
        perc_25 = np.percentile(data, 25, axis=0)
        perc_75 = np.percentile(data, 75, axis=0)
        return y, perc_25, perc_75

    results_list = []
    for case in use_cases:
        results = {}

        for method_name in methods_names:
            method_data = [
                d
                for d in data
                if (d["config"]["use_case"] == case)
                and (d["config"]["name"] == method_name)
            ]

            if len(method_data) == 0:
                print(
                    f"No result found for use-case {case} and methodology {method_name}"
                )
                continue

            inputs = [d["inputs"] for d in method_data]
            oracles = [d["logs"]["oracle"].to_numpy() for d in method_data]

            results[method_name] = compute_statistics(
                compute_evolution_fault_triggering_inputs(inputs, oracles)
                if only_uniques else accumulate(oracles)
            )

        results_list.append(results)

    return results_list


def plot_rq1_results(
    use_cases: List[str],
    colors_dict: Dict[str, Tuple],
    results: List[Dict[str, Union[Tuple, np.ndarray]]],
):
    """Plots the results for RQ1."""

    n = len(use_cases)
    fig, axs = plt.subplots(ncols=n, figsize=(7 * n, 6), sharex=True)
    if n == 1:
        axs = [axs]
        axs[0].grid(axis="y", color="0.9", linestyle="-", linewidth=1)
    else:
        [ax.grid(axis="y", color="0.9", linestyle="-", linewidth=1) for ax in axs.flat]

    axs[0].set_ylabel(FAULT_LABEL, fontsize=AXIS_LABEL_FONTSIZE)

    for u in range(n):
        data = results[u]
        ax = axs[u]
        ax.set_xlabel("#Iterations", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_title(use_cases[u], fontsize=TITLE_LABEL_FONTSIZE)
        for name, data in data.items():
            color = colors_dict[name]
            label = name
            if "MAE+LS" in label:
                linestyle = "dotted"
            elif "MAE+ML" in label:
                linestyle = "dashed"
            elif "AAE+LS" in label:
                linestyle = "dashdot"
            else:
                linestyle = "solid"
            if isinstance(data, np.ndarray):
                ax.plot(np.arange(len(data)), data, color=color, label=label, linestyle=linestyle, linewidth=2)
            else:
                y, perc_25, perc_75 = data
                x = np.arange(len(y))
                ax.plot(x, y, color=color, label=label, linestyle=linestyle, linewidth=2)
                ax.fill_between(
                    x, perc_25, perc_75, alpha=0.15, linewidth=0, color=color
                )

    # ax = axs[np.argmax([len(d.keys()) for d in results])]
    ax = axs[-1]
    legend = ax.legend(
        prop={"size": 12},
        labelspacing=1.1,
        handletextpad=1.05,
        borderpad=1.05,
        borderaxespad=1.0,
    )
    legend_frame = legend.get_frame()
    legend_frame.set_facecolor("0.9")
    legend_frame.set_edgecolor("0.9")
    fig.tight_layout()
    return (fig, axs)


#############################################################################################################
################## RQ2: How well QD-based testing methodologies improve test coverage/diversity? ############


def compute_evolution_cells(data: List[pd.DataFrame]):
    """Computes the number of cells discovered over search iterations."""
    results = [df["nb_cells"].to_numpy() for df in data]
    return results


def load_obs_space_edges(use_case: str, num_bins: int = 20) -> np.ndarray:
    if use_case.capitalize().startswith("B"):
        return np.load(f"grid/bw/obs_space_edges_{num_bins}.npy")

    elif use_case.capitalize().startswith("L"):
        return np.load(f"grid/ll/obs_space_edges_{num_bins}.npy")

    elif use_case.capitalize().startswith("H"):
        return np.load(f"grid/hw/obs_space_edges_{num_bins}.npy")

    else:
        raise ValueError(f"No edges available for use case {use_case}.")

def compute_obs_coverage(results: List[Dict]):
    use_cases: List[str] = np.unique(
        [d["config"]["use_case"] for d in results]
    ).tolist()
    methods_names: List[str] = np.unique(
        [d["config"]["name"] for d in results]
    ).tolist()

    def compute_statistics(data) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not isinstance(data, np.ndarray):
            data: np.ndarray = np.array(data)
        y = np.median(data, axis=0)
        perc_25 = np.percentile(data, 25, axis=0)
        perc_75 = np.percentile(data, 75, axis=0)
        return y, perc_25, perc_75

    # re-aranges the results per use_case
    use_cases_dict: Dict[str, Dict[str, List]] = {}
    for case in use_cases:
        use_cases_dict[case] = {}
        for name in methods_names:
            # case's results
            sub_results = [
                d
                for d in results
                if (d["config"]["use_case"].startswith(case))
                and (d["config"]["name"] == name)
            ]
            use_cases_dict[case].update({name: sub_results})

    # dict of (the means of) the medians and quantiles for all methodologies per use-case
    results_dicts: List[Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]] = []
    # similar data but with distinct fault-triggering final states
    fresults_dicts: List[Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]] = []

    for case in use_cases:
        # results of all methodologies
        method_result_dict = {}
        method_fresult_dict = {}
        # edges for the use case
        edges = load_obs_space_edges(case, num_bins=10)
        for method_name, sub_results in use_cases_dict[case].items():
            if len(sub_results) == 0:
                print(
                    f"No result found for use-case {case} and methodology {method_name}"
                )
                continue

            fs_list = []
            oracles_list = []
            for d in sub_results:
                fs_list.extend(d["final_states"])
                # copies the oracle flags for each list
                for _ in range(len(d["final_states"])):
                    oracles_list.append(d["logs"]["oracle"].to_numpy().copy())

            assert len(fs_list) == len(oracles_list)
            print(f"For ({case}, {method_name}), found a total of {len(fs_list)} final states lists.")

            # bin the final states
            cells_arr = np.apply_along_axis(
                lambda x: bin_observation(x, edges),
                axis=-1,
                arr=fs_list
            )
            obs_coverage = accumulate_uniques(cells_arr)
            median, q1, q3 = compute_statistics(obs_coverage)
            method_result_dict[method_name] = (median, q1, q3)

            # accumulates the number of unique cells flagged as faulty fault-triggering
            fobs_coverage = compute_evolution_fault_triggering_inputs(cells_arr, oracles_list)
            fmedian, fq1, fq3 = compute_statistics(fobs_coverage)
            method_fresult_dict[method_name] = (fmedian, fq1, fq3)

        results_dicts.append(method_result_dict)
        fresults_dicts.append(method_fresult_dict)
    return use_cases, results_dicts, fresults_dicts


def compute_relative_performance(
    results: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    name_ref: str = "Random Testing",
):
    """
    Returns the relative performance to the set of results labeled ``name_ref``.
    Even though it assumes statistical results (as medians and 1/3 quantiles), the relative performance is computed on the medians only.
    """
    relative_results: Dict[str, np.ndarray] = {}
    if results.get(name_ref, None) is None:
        print(f"No result found for the reference {name_ref}.")
        return results
    m_ref, q1_ref, q3_ref = results[name_ref]
    for name, (m, q1, q3) in results.items():
        relative_results[name] = ((m - m_ref) / m_ref) + 1
    return relative_results


def legend_axis(ax):
    legend = ax.legend(
        prop={"size": 11},
        ncol=2,
        labelspacing=1.05,
        handletextpad=1.025,
        borderpad=1.025,
        borderaxespad=1.0,
        loc="upper left"
    )
    legend_frame = legend.get_frame()
    legend_frame.set_facecolor("0.9")
    legend_frame.set_edgecolor("0.9")
    return ax


def plot_coverage_results(
    use_cases: List[str],
    colors_dict: Dict[str, Tuple],
    cov_results: List[Dict[str, Tuple]],
    faulty_cov_results: List[Dict[str, np.ndarray]],
):
    """Plots the coverage results for RQ2."""

    nb_use_cases = len(use_cases)

    fig, axs = plt.subplots(
        nrows=nb_use_cases, ncols=2, figsize=(10, 4.5 * nb_use_cases), sharex=True
    )
    if nb_use_cases == 1:
        axs = [axs]
        [
            ax.grid(axis="y", color="0.9", linestyle="-", linewidth=1)
            for ax in axs[0].flat
        ]
    else:
        [ax.grid(axis="y", color="0.9", linestyle="-", linewidth=1) for ax in axs.flat]

    axs[-1][0].set_xlabel("#Iterations", fontsize=AXIS_LABEL_FONTSIZE)
    axs[-1][1].set_xlabel("#Iterations", fontsize=AXIS_LABEL_FONTSIZE)

    for u in range(nb_use_cases):
        case = use_cases[u]
        axs[u][0].set_ylabel(case, fontsize=AXIS_LABEL_FONTSIZE)
        for name in cov_results[u].keys():
            color = colors_dict[name]
            label = name
            if "MAP-Elites" in label:
                label = label.replace("MAP-Elites", "ME")
            if "Novelty Search" in label:
                label = label.replace("Novelty Search", "NS")
            if "MAE+LS" in label:
                linestyle = "dotted"
            elif "MAE+ML" in label:
                linestyle = "dashed"
            elif "AAE+LS" in label:
                linestyle = "dashdot"
            else:
                linestyle = "solid"
            # BS coverage (statistical)
            ax = axs[u][0]
            y, perc_25, perc_75 = cov_results[u][name]
            x = np.arange(len(y))
            ax.plot(x, y, color=color, label=label, linestyle=linestyle, linewidth=2)
            ax.fill_between(x, perc_25, perc_75, alpha=0.15, linewidth=0, color=color)
            # Faulty BS coverage (statistical)
            ax = axs[u][1]
            y, perc_25, perc_75 = faulty_cov_results[u][name]
            x = np.arange(len(y))
            ax.plot(x, y, color=color, label=label, linestyle=linestyle, linewidth=2)
            ax.fill_between(x, perc_25, perc_75, alpha=0.15, linewidth=0, color=color)

        # once every methodology's results is displayed, adds the legend
        ax = axs[u][-1]
        ax = legend_axis(ax)
    fig.tight_layout()
    return (fig, axs)


def plot_rq2_ebs_results(
    use_cases: List[str],
    colors_dict: Dict[str, Tuple],
    ebs_results: List[Dict[str, Tuple]],
    faulty_ebs_results: List[Dict[str, np.ndarray]],
):
    fig, axs = plot_coverage_results(use_cases, colors_dict, ebs_results, faulty_ebs_results)
    axs[0][0].set_title("#Expert Behaviors", fontsize=TITLE_LABEL_FONTSIZE)
    axs[0][1].set_title("#Faulty Expert Behaviors", fontsize=AXIS_LABEL_FONTSIZE)
    fig.tight_layout()
    return fig, axs

def plot_rq2_fobs_results(
    use_cases: List[str],
    colors_dict: Dict[str, Tuple],
    fobs_results: List[Dict[str, Tuple]],
    faulty_fobs_results: List[Dict[str, np.ndarray]],
):
    fig, axs = plot_coverage_results(use_cases, colors_dict, fobs_results, faulty_fobs_results)
    axs[0][0].set_title("#Final States", fontsize=TITLE_LABEL_FONTSIZE)
    axs[0][1].set_title("#Faulty Final States", fontsize=AXIS_LABEL_FONTSIZE)
    fig.tight_layout()
    return fig, axs


def plot_n_results(
    use_cases: List[str],
    env_seeds: List[int],
    results: List[Dict[str, List[Tuple[np.ndarray, np.ndarray, np.ndarray]]]],
    colors_dict: Dict[str, Tuple],
    additional_results: List[Dict[str, List[Tuple[np.ndarray, np.ndarray, np.ndarray]]]] = None,
    x_axis: str = "iterations"
):
    """
    Plots any statistical results for each use case and n.
    `results` is assumed to be ordered w.r.t to `use_cases`.
    Use cases are shown horizontally (with the different values of n).

    Parameters
    ----------
    use_cases : List[str]
        Names of the use cases.
    env_seeds : List[int]
        Number of seeds (i.e. n) used for each result in `results`' values.
    results : List[Dict[str, List[Tuple[np.ndarray, np.ndarray, np.ndarray]]]]
        List of dictionaries whose keys are the names of the methods, and values are a list of statistical data (tuple of 3 numpy arrays of equal length); one for each n value.
        As such, the length of results must equal the one of `use_cases`, and all the values in all the dictionaries must be of length equal to one of `env_seeds`.
    additional_results : List[Dict[str, List[Tuple[np.ndarray, np.ndarray, np.ndarray]]]], optinal
        Additional results to show, cf. RQ2 and RQ3. Notably, the results are assumed to come from the same methods. They are plotted with dashed lines.
    x_axis : str, optional
        Mode for the x axis. Must be either:
        - "iterations": no manipulation of the data.
        - "executions": each value in the data is repeated w.r.t `env_seeds`.

        Default to "iterations".

    Returns
    -------
    (fig, axes)
    """
    assert x_axis in ["iterations", "executions"]
    assert len(use_cases) == len(results)
    if not all([l == len(env_seeds) for l in sum([[len(r) for r in d.values()] for d in results], [])]):
        warnings.warn("Not all the lists of results in `results`' dictionaries have |env_seeds| data...")

    fig_size = 5
    fig, axes = plt.subplots(
        nrows=len(use_cases),
        ncols=len(env_seeds),
        figsize=(fig_size * len(env_seeds), fig_size * len(use_cases)),
        sharey="row",
        # sharex=True
        sharex=True if x_axis == "iterations" else "col"
    )

    if len(use_cases) == 1:
        axes = [axes]
        axes[0].grid(axis="both", color="0.9", linestyle="-", linewidth=1)
    else:
        [ax.grid(axis="both", color="0.9", linestyle="-", linewidth=1) for ax in axes.flat]

    [ax.set_xlabel(f"#{x_axis.capitalize()}", fontsize=AXIS_LABEL_FONTSIZE) for ax in axes[-1]]
    [ax.set_title(f"N={n}", fontsize=AXIS_LABEL_FONTSIZE) for (ax, n) in zip(axes[0], env_seeds)]
    [ax.set_ylabel(case.capitalize(), fontsize=AXIS_LABEL_FONTSIZE) for (ax, case) in zip([a[0] for a in axes], use_cases)]

    def _repeat_data(arr: np.ndarray, n: int):
        return  np.array(
            sum(
                [[v for _ in range(n)] for v in arr],
                []
            )
        )

    for (data, axs) in zip(results, axes):
        for name, res_list in data.items():
            for i, (y, perc_25, perc_75) in enumerate(res_list):
                ax = axs[i]
                color = colors_dict[name]
                label = name
                if x_axis != "iterations":
                    n = env_seeds[i]
                    x = np.arange(len(y) * n)
                    y = _repeat_data(y, n)
                    perc_25 = _repeat_data(perc_25, n)
                    perc_75 = _repeat_data(perc_75, n)
                else:
                    x = np.arange(len(y))

                ax.plot(x, y, color=color, label=label, linewidth=2)
                ax.fill_between(
                    x, perc_25, perc_75, alpha=0.15, linewidth=0, color=color
                )

    if additional_results is not None:
        for (data, axs) in zip(additional_results, axes):
            for name, res_list in data.items():
                for i, (y, perc_25, perc_75) in enumerate(res_list):
                    ax = axs[i]
                    color = colors_dict[name]
                    label = name
                    if x_axis != "iterations":
                        n = env_seeds[i]
                        x = np.arange(len(y) * n)
                        y = _repeat_data(y, n)
                        perc_25 = _repeat_data(perc_25, n)
                        perc_75 = _repeat_data(perc_75, n)
                    else:
                        x = np.arange(len(y))

                    ax.plot(x, y, color=color, linewidth=2, linestyle="dashed")
                    ax.fill_between(
                        x, perc_25, perc_75, alpha=0.15, linewidth=0, color=color
                    )

    # xticks = np.arange(0, 1 + max([s for s in env_seeds]) * 5000, 5000)
    # ax.set_xticks(
    #     xticks
    # )
    # ax.set_xticklabels(
    #     ["0"] + [f"{i}K" for i in range(1, len(xticks))]
    #     )
    ax = axes[np.argmax([len(d.keys()) for d in results])][0]
    legend = ax.legend(
        prop={"size": 10},
        labelspacing=1.1,
        handletextpad=1.05,
        borderpad=1.05,
        borderaxespad=1.0,
    )
    legend_frame = legend.get_frame()
    legend_frame.set_facecolor("0.9")
    legend_frame.set_edgecolor("0.9")
    fig.tight_layout()
    return (fig, axes)


#############################################################################################################
##################### NEW RQ2: coverage of the different expert behavior spaces #############################
##################### UNUSED RQ2: coverage of the generic behavior spaces ###################################


def compute_expert_behaviors_coverage(data: List[Dict]):
    """Compute the evolution of the number of cells filled in the expert behavior space(s).
    The statistical computation is not detailed in case of multiple spaces for a use case (e.g. Bipedal Walker).

    Parameters
    ----------
    data : List[Dict]
        List of dictionaries of results. They must contain the entries ``config`` and ``expert_behaviors``.

    Returns
    -------
    ebs_coverage : List[Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]]
        List of dictionaries per use case. The results of each method consists of three numpy arrays, and are indexed by their name.
    febs_coverage : List[Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]]
        Same as `ebs_coverage`, except that only fault triggering data is accounted for.
    """
    use_cases: List[str] = np.unique([d["config"]["use_case"] for d in data]).tolist()
    methods_names: List[str] = np.unique([d["config"]["name"] for d in data]).tolist()

    def compute_statistics(data) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not isinstance(data, np.ndarray):
            data: np.ndarray = np.array(data)
        y = np.median(data, axis=0)
        perc_25 = np.percentile(data, 25, axis=0)
        perc_75 = np.percentile(data, 75, axis=0)
        return y, perc_25, perc_75

    ebs_coverages = []
    faulty_ebs_coverages = []
    for case in use_cases:
        ebs_cov = {}
        faulty_ebs_cov = {}

        for method_name in methods_names:
            method_data = [
                d
                for d in data
                if (d["config"]["use_case"] == case)
                and (d["config"]["name"] == method_name)
            ]

            if len(method_data) == 0:
                warnings.warn(
                    f"No result found for use-case {case} and methodology {method_name}",
                    RuntimeWarning,
                )
                continue

            oracles = []
            cells = []
            for d in method_data:
                c_list = []
                o_arr = d["logs"]["oracle"].to_numpy()  # type: np.ndarray
                expert_behaviors_list = d.get(
                    "expert_behaviors", []
                )  # type: List[np.ndarray]
                for eb in expert_behaviors_list:
                    if case == "Bipedal Walker":
                        desc_indices = EXPERT_INDICES
                        edges_list = [
                            get_expert_bin_edges(case, desc) for desc in desc_indices
                        ]

                    else:
                        edges_list = [get_expert_bin_edges(case)]
                        desc_indices = [[0, 1]]
                    c_list.extend(
                        compute_cell_filling(
                            behaviors=eb,
                            descriptor_indices_list=desc_indices,
                            edges=edges_list,
                        )
                    )
                # flags the faults for each array of cells
                # TODO: for BW, how to track the dimensions' names...
                for c in c_list:
                    oracles.append(o_arr.copy())
                    cells.append(c)
            print(case, method_name, len(cells))
            assert len(cells) == len(oracles)

            ebs_cov[method_name] = compute_statistics(accumulate_uniques(cells))
            faulty_ebs_cov[method_name] = compute_statistics(
                [res[-1] for res in filter_data(oracles, cells)]
            )

        ebs_coverages.append(ebs_cov)
        faulty_ebs_coverages.append(faulty_ebs_cov)

    return ebs_coverages, faulty_ebs_coverages


def compute_measures_coverage(
    data: List[Dict], meas_indices: List[Tuple[int, int]] = MEAS_INDICES
):
    """Similar to `compute_expert_behaviors_coverage` but the data used is the multi-dimensional generic behaviors.


    Parameters
    ----------
    data : List[Dict]
        List of dictionaries of results. They must contain the entries ``config`` and ``behaviors``.
    meas_indices : List[Tuple[int, int]], optional
        List of index pairs of the 2D generic behavior spaces. Default to MEAS_INDICES.

    Returns
    -------
    meas_coverage : List[Dict[str, List[Tuple[np.ndarray, np.ndarray, np.ndarray]]]]
        List of dictionaries per use case. The results of each method consists of three numpy arrays, and are indexed by their name.
    fmeas_coverage : List[Dict[str, List[Tuple[np.ndarray, np.ndarray, np.ndarray]]]]
        Same as `meas_coverage`, except that only fault triggering data is accounted for.
    """
    use_cases: List[str] = np.unique([d["config"]["use_case"] for d in data]).tolist()
    methods_names: List[str] = np.unique([d["config"]["name"] for d in data]).tolist()

    def compute_statistics(data) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not isinstance(data, np.ndarray):
            data: np.ndarray = np.array(data)
        y = np.median(data, axis=0)
        perc_25 = np.percentile(data, 25, axis=0)
        perc_75 = np.percentile(data, 75, axis=0)
        return y, perc_25, perc_75

    ebs_coverages = []
    faulty_ebs_coverages = []
    for case in use_cases:
        ebs_cov = {}
        faulty_ebs_cov = {}

        edges = get_measures_edges(case)

        for method_name in methods_names:
            method_data = [
                d
                for d in data
                if (d["config"]["use_case"] == case)
                and (d["config"]["name"] == method_name)
            ]

            if len(method_data) == 0:
                warnings.warn(
                    f"No result found for use-case {case} and methodology {method_name}",
                    RuntimeWarning,
                )
                continue

            # cells of each behavior space per run (i.e., list of list)
            meas_cells = [
                compute_cell_filling(
                    behaviors=d["behaviors"],
                    descriptor_indices_list=meas_indices,
                    edges=[edges[idx] for idx in meas_indices],
                )
                for d in method_data
            ]
            meas_cells = np.array(meas_cells)
            # oracles per run
            oracles = [
                d["logs"]["oracle"].to_numpy()  # type: np.ndarray
                for d in method_data
            ]

            ebs_cov[method_name] = [
                compute_statistics(accumulate_uniques(meas_cells[:, meas_index, :]))
                for meas_index in range(len(meas_indices))
            ]
            faulty_ebs_cov[method_name] = [
                compute_statistics(
                    [
                        res[-1]
                        for res in filter_data(oracles, meas_cells[:, meas_index, :])
                    ]
                )
                for meas_index in range(len(meas_indices))
            ]

        ebs_coverages.append(ebs_cov)
        faulty_ebs_coverages.append(faulty_ebs_cov)

    return ebs_coverages, faulty_ebs_coverages


def plot_rq2_meas_results(
    colors_dict: Dict[str, Tuple],
    meas_dict: Dict[str, List[Tuple[np.ndarray, np.ndarray, np.ndarray]]],
    measures: List[str] = MEASURES,
    meas_str_indices: List[Tuple[str, str]] = None,
    meas_indices: List[Tuple[int, int]] = None,
):
    """Plots the coverage of generic behavior spaces for a single use case.

    Parameters
    ----------
    measures : List[str], optional
        Names of the generic features. Default to MEASURES.
    meas_str_indices : List[Tuple[str, str]], optional
        List of dimension's name pairs. Inferred from `meas_indices` if not provided.
    meas_indices : List[Tuple[int, int]], optional
        List of index pairs of the 2D generic behavior spaces. Inferred from `meas_str_indices` if not provided.
    """

    if (meas_str_indices is None) and (meas_indices is None):
        if measures != MEASURES:
            raise ValueError(
                "Indices (either as str and int) must be specified in case of none default measures."
            )
        meas_str_indices = MEAS_STR_INDICES
        meas_indices = MEAS_INDICES

    # infers the str (int) indices from the int (str) ones, respectively
    if meas_str_indices is None:
        assert meas_indices is not None
        meas_str_indices = [[measures[i], measures[j]] for (i, j) in meas_indices]
    if meas_indices is None:
        assert meas_str_indices is not None
        meas_str_indices = [
            [measures[measures.index(i)], measures[measures.index(j)]]
            for (i, j) in meas_indices
        ]

    x_axes = list(set([meas_str_indices[i][0] for i in range(len(meas_str_indices))]))
    y_axes = list(set([meas_str_indices[i][1] for i in range(len(meas_str_indices))]))

    fig, axs = plt.subplots(
        nrows=len(y_axes),
        ncols=len(x_axes),
        figsize=(len(x_axes) * 5, len(y_axes) * 5),
        # sharex="col",
        # sharey="row"
    )
    [ax.grid(axis="y", color="0.9", linestyle="-", linewidth=1) for ax in axs.flat]

    for i in range(len(x_axes)):
        for j in range(len(y_axes)):

            data_index = meas_indices.index(
                [measures.index(x_axes[i]), measures.index(y_axes[j])]
            )

            for name in meas_dict.keys():
                color = colors_dict[name]
                label = name
                # BS coverage (statistical)
                ax = axs[j][i]
                y, perc_25, perc_75 = meas_dict[name][data_index]
                x = np.arange(len(y))
                ax.plot(x, y, color=color, label=label)
                ax.fill_between(
                    x, perc_25, perc_75, alpha=0.15, linewidth=0, color=color
                )
                ax.set_xlabel(x_axes[i])
                ax.set_ylabel(y_axes[j])
                # once every methodology's results is displayed, adds the legend
    ax = axs.flatten()[-1]
    legend = ax.legend(
        prop={"size": 10},
        labelspacing=1.1,
        handletextpad=1.05,
        borderpad=1.05,
        borderaxespad=1.0,
    )
    legend_frame = legend.get_frame()
    legend_frame.set_facecolor("0.9")
    legend_frame.set_edgecolor("0.9")
    fig.tight_layout()
    return (fig, axs)


def plot_rq3_results(
        data_lists: List[List[Dict[str, list]]],
        colors_dict: Dict[str, Tuple[float]],
        env_seeds: List[int],
        use_cases: List[str] = None,
        ylabels: List[str] = None,
        x_axis: str = "iterations"):

    if use_cases is None:
        use_cases = np.arange(np.max([len(l) for l in data_lists]))

    nrows = len(data_lists)
    ncols = len(use_cases)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6*ncols, 4.5*nrows), sharex="all", sharey="none")

    if ylabels is None:
        ylabels = ["" for _ in range(nrows)]
    for ax in axes.flat:
        ax.grid(axis="y", color="0.9", linestyle="-", linewidth=1)

    def _repeat_data(arr: np.ndarray, n: int):
        return  np.array(
            sum(
                [[v for _ in range(n)] for v in arr],
                []
            )
        )

    # distinct per-method N plotting with linestyling
    linestyles = ["dotted", "dashed", "dashdot", "solid"]
    assert len(env_seeds) <= len(linestyles), "Too many different env_seeds (n values) to plot."

    # per row (i.e. metric/result)
    for r, data in enumerate(data_lists):
        axes[r][0].set_ylabel(ylabels[r], fontsize=AXIS_LABEL_FONTSIZE)
        for c in range(ncols):
            ax = axes[r][c]
            case_dict = data[c]
            for name, res_list in case_dict.items():
                # list of stats results per method (for different n values)
                for i, (y, perc_25, perc_75) in enumerate(res_list):
                    color = colors_dict[name]
                    # label = f"{name} N={env_seeds[i]}"
                    label = name

                    if "MAP-Elites" in label:
                        label = label.replace("MAP-Elites", "ME")
                    if "Novelty Search" in label:
                        label = label.replace("Novelty Search", "NS")
                    if "Radom Testing" in label:
                        label = label.replace("Radom Testing", "RT")

                    if x_axis != "iterations":
                        n = env_seeds[i]
                        x = np.arange(len(y) * n)
                        y = _repeat_data(y, n)
                        # perc_25 = _repeat_data(perc_25, n)
                        # perc_75 = _repeat_data(perc_75, n)
                    else:
                        x = np.arange(len(y))

                    ax.plot(x, y, color=color, label=(label if i == len(env_seeds) - 1  else None), linewidth=2, linestyle=linestyles[i])
                    # ax.fill_between(
                    #     x, perc_25, perc_75, alpha=0.15, linewidth=0, color=color
                    # )

    def legend_axis(ax):
        legend = ax.legend(prop={"size": 10}, ncol=1, labelspacing=1.1, handletextpad=1.05, borderpad=1.05, borderaxespad=1.0, loc="upper left")
        legend_frame = legend.get_frame()
        legend_frame.set_facecolor("0.9")
        legend_frame.set_edgecolor("0.9")


        custom_lines = [
            Line2D([0], [0], color="black", linestyle=style, linewidth=2)
            for style in linestyles
        ]

        custom_labels = [r"$n = {}$".format(i) for i in env_seeds]
        custom_legend = Legend(
            ax,
            custom_lines,
            custom_labels,
            loc="lower right",
            labelspacing=1.1,
            handletextpad=1.0,
            handlelength=3, # default is 2
            borderpad=1.0,
            borderaxespad=1.0,
            prop={"size": 10}
        )
        legend_frame = custom_legend.get_frame()
        legend_frame.set_facecolor("0.9")
        legend_frame.set_edgecolor("0.9")

        ax.add_artist(custom_legend)
        return ax

    for c in range(ncols):
        axes[0][c].set_title(use_cases[c], fontsize=TITLE_LABEL_FONTSIZE)
    fig.tight_layout()

    return (fig, axes), legend_axis


##############################################################################
################################## MAIN ######################################


def load_data():
    # LL
    ll_results = read_results_from_folder(
        "results_new/ll/qd/",
        include_final_states=True,
        include_expert_behaviors=True
        )
    [
        ll_results.extend(
            read_results_from_folder(f"results_new/ll/{m}/", include_final_states=True, include_expert_behaviors=True)
        )
        for m in ["ns", "rt", "mdpfuzz"]
    ]
    [
        ll_results.extend(
            read_results_from_folder(f"results_new2/ll/{m}/", include_final_states=True, include_expert_behaviors=True)
        )
        for m in ["ns", "rt", "mdpfuzz", "qd"]
    ]

    # BW
    bw_results = read_results_from_folder(
        "results_new/bw/qd/",
        include_final_states=True,
        include_expert_behaviors=True
    )
    [
        bw_results.extend(
            read_results_from_folder(f"results_new/bw/{m}/", include_final_states=True, include_expert_behaviors=True)
        )
        for m in ["ns", "rt", "mdpfuzz"]
    ]
    [
        bw_results.extend(
            read_results_from_folder(f"results_new2/bw/{m}/", include_final_states=True, include_expert_behaviors=True)
        )
        for m in ["ns", "rt", "mdpfuzz", "qd"]
    ]

    # HW
    hw_results = read_results_from_folder(
        "../highway/results_new/hw/qd/",
        include_final_states=True,
        include_expert_behaviors=True
    )
    [
        hw_results.extend(
            read_results_from_folder(f"../highway/results_new/hw/{m}/", include_final_states=True, include_expert_behaviors=True)
        )
        for m in ["ns", "rt", "mdpfuzz"]
    ]
    [
        hw_results.extend(
            read_results_from_folder(f"../highway/results_new2/hw/{m}/", include_final_states=True, include_expert_behaviors=True)
        )
        for m in ["ns", "rt", "mdpfuzz", "qd"]
    ]
    # renames the QD-based methods w.r.t the descriptor pair used
    for d in ll_results + bw_results + hw_results:
        if "name" not in d["config"]:
            d["config"]["name"] = "MDPFuzz"
        if d["config"]["name"] in ["MAP-Elites", "Novelty Search"]:
            descriptors = d["config"]["descriptors"]
            prefix = "M" if "mean" in descriptors[0] else "A"
            suffix = "LS" if "spread" in descriptors[-1] else "ML"
            d["config"]["name"] += f" {prefix}AE+{suffix}"

    return bw_results + ll_results + hw_results

def fetch_result_data(folder_name: str):
    """
    Returns all the results found in the local folder `folder_name`.

    **It assumes that the same folder exists in ../highway/**.
    """
    use_cases = ["bw", "ll"]
    methods = ["ns", "rt", "mdpfuzz", "qd"]

    results = sum(
        [
            read_results_from_folder(
                f"{folder_name}/{u}/{m}/",
                include_final_states=True,
                include_expert_behaviors=True
            )
            for u in use_cases for m in methods],
        []
    )
    results.extend(
        sum(
            [
                read_results_from_folder(
                    f"../highway/{folder_name}/hw/{m}/",
                    include_final_states=True,
                    include_expert_behaviors=True
                )
                for m in methods
            ],
            []
        )
    )
    return results


# exec(open('result_analysis.py').read())
if __name__ == "__main__":
    torch.set_num_threads(1)

    ####################### Raw data loading #######################

    first_results = load_data()

    use_cases, method_names, colors_dict = color_data(first_results)
    print(method_names, use_cases, len(first_results))

    ####################### Analysis computation #######################

    # fault detection
    rq1_data = compute_rq1_results(first_results)

    # expert behavior coverage
    ebs_cov, efbs_cov = compute_expert_behaviors_coverage(first_results)
    # stores the results of the analysis
    folder = "data_new"
    for case in use_cases:
        sub_folder = f"{folder}/{case}"
        Path(sub_folder).mkdir(parents=True, exist_ok=True)

    dump_results(rq1_data, [f"{folder}/{case}/rq1" for case in use_cases])
    dump_results(ebs_cov, [f"{folder}/{case}/bs_cov" for case in use_cases])
    dump_results(efbs_cov, [f"{folder}/{case}/fbs_cov" for case in use_cases])

    # final observation coverage
    cases, obs_coverage_results, fobs_coverage_results = compute_obs_coverage(first_results)
    dump_results(obs_coverage_results, [f"{folder}/{case}/obs_cov" for case in cases])
    dump_results(fobs_coverage_results, [f"{folder}/{case}/fobs_cov" for case in cases])

    ########################### Plotting ###########################

    fig1, axs1 = plot_rq1_results(use_cases, colors_dict, rq1_data)
    for ax in axs1.flat:
        ax.tick_params(axis="both",labelsize=13)
        legend = ax.legend_
        if legend is not None:
            for line in legend.get_lines():
                plt.setp(line, linewidth=4)
    fig1.savefig("rq1.png")

    fig2, axs2 = plot_rq2_ebs_results(use_cases, colors_dict, ebs_cov, efbs_cov)
    axs2[0][-1].legend_ = None
    axs2[1][-1].legend_ = None
    legend = axs2[-1][-1].legend_
    for line in legend.get_lines():
        plt.setp(line, linewidth=4)
    for ax in axs2.flat:
        ax.tick_params(axis="both", labelsize=12)
    fig2.savefig("rq21.png")

    fig2, axs2 = plot_rq2_fobs_results(cases, colors_dict, obs_coverage_results, fobs_coverage_results)
    axs2[0][-1].legend_ = None
    axs2[1][-1].legend_ = None
    legend = axs2[-1][-1].legend_
    for line in legend.get_lines():
        plt.setp(line, linewidth=4)
    for ax in axs2.flat:
        ax.tick_params(axis="both", labelsize=12)
    fig2.savefig("rq22.png")


    with open("colors_dict.json", "w") as file:
        json.dump(colors_dict, file)

    print("DONE.")