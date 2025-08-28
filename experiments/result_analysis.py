import json
import os
from pathlib import Path
import warnings
from typing import Dict, Iterable, List, Tuple, Union

from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import torch
from bw_framework import EXPERT_INDICES
import matplotlib
from matplotlib import pyplot as plt
from collections.abc import Iterable

from common import (
    bin_observation,
    compute_cell_filling,
    dump_results,
    get_expert_bin_edges,
    read_results_from_folder,
)

FAULT_LABEL = "#Faults"
AXIS_LABEL_FONTSIZE = 19
AXIS_TICKLABELS_FONTSIZE = 14
TITLE_LABEL_FONTSIZE = 20
LEGEND_LINEWIDTH = 6

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


def fetch_results(folder_name: str):
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


##############################################################################################
#################################### Plotting Helpers #####################################


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


def post_process_label(label: str):
    linestyle = "solid"

    if label.startswith("MDPFuzz"):
        label += "$\mathbf{^*}$"

    elif "+" in label:
        label = label.replace("MAP-Elites", "ME")
        label = label.replace("Novelty Search", "NS")

        if "MAE+LS" in label:
            linestyle = "dotted"
        elif "MAE+ML" in label:
            linestyle = "dashed"
        elif "AAE+LS" in label:
            linestyle = "dashdot"

    return label, linestyle


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


##############################################################################################
############################## RQ1: Fault Discovery ##########################################


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
        axs = np.expand_dims(axs, axis=0)  # type: np.ndarray

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
            label, linestyle = post_process_label(label)

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
################# RQ2.1: Testing and Fault Diversity of the Final States ####################################


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
        axs = np.expand_dims(axs, axis=0)  # type: np.ndarray

    [ax.grid(axis="y", color="0.9", linestyle="-", linewidth=1) for ax in axs.flat]

    axs[-1][0].set_xlabel("#Iterations", fontsize=AXIS_LABEL_FONTSIZE)
    axs[-1][1].set_xlabel("#Iterations", fontsize=AXIS_LABEL_FONTSIZE)

    for u in range(nb_use_cases):
        case = use_cases[u]
        axs[u][0].set_ylabel(case, fontsize=AXIS_LABEL_FONTSIZE)
        for name in cov_results[u].keys():
            color = colors_dict[name]
            label = name
            label, linestyle = post_process_label(label)

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


#############################################################################################################
##################### RQ2.2: Testing and Fault Diversity in the Expert Space ################################


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
                    if case.startswith("Bipedal Walker"):
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


#############################################################################################################
######################## Main RQs' Plotting Functions #######################################################


def plot_summary_results(
        data_lists: List[List[Dict]],
        colors_dict: Dict[str, Tuple[float]],
        use_cases: List[str] = None,
        ylabels: List[str] = None,
        sharey: str = "row",
        figsize: Tuple[int, int] = (4, 5)):
    if use_cases is None:
        use_cases = np.arange(np.max([len(l) for l in data_lists]))

    nrows = len(data_lists)
    ncols = len(use_cases)

    s0, s1 = figsize
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(s0*nrows, s1*ncols), sharex="all", sharey=sharey)

    if ylabels is None:
        ylabels = ["" for _ in range(nrows)]
    for ax in axs.flat:
        ax.grid(axis="y", color="0.9", linestyle="-", linewidth=1)

    for r, data in enumerate(data_lists):
        axs[r][0].set_ylabel(ylabels[r], fontsize=AXIS_LABEL_FONTSIZE)
        for c in range(ncols):
            ax = axs[r][c]
            case_dict = data[c]
            for name in case_dict.keys():
                to_plot = case_dict[name]
                color = colors_dict[name]
                label = name  # type: str
                label, linestyle = post_process_label(label)

                if isinstance(to_plot, np.ndarray):
                    x = np.arange(len(to_plot))
                    assert len(x) == len(to_plot)
                    ax.plot(x, to_plot, color=color, label=label, linewidth=2, linestyle=linestyle)
                else:
                    # assert np.all([len(x) == len(tmp) for tmp in to_plot])
                    y, perc_25, perc_75 = to_plot
                    x = np.arange(len(y))
                    ax.plot(x, y, color=color, label=label, linestyle=linestyle, linewidth=2)
                    ax.fill_between(x, perc_25, perc_75, alpha=0.25, linewidth=0, color=color)
        if r % 2 == 0:
            ax = axs[r][1]
            legend = ax.legend(
                prop={"size": 13.5},
                labelspacing=0.7,
                handletextpad=0.8, # default
                borderpad=0.5,
                borderaxespad=1.0,
                ncol=2 if len(ax.get_legend_handles_labels()[1]) > 4 else 1,
                columnspacing=1.0 # default 2.0
            )
            legend_frame = legend.get_frame()
            legend_frame.set_facecolor("0.9")
            legend_frame.set_edgecolor("0.9")
    for c in range(ncols):
        axs[0][c].set_title(use_cases[c], fontsize=TITLE_LABEL_FONTSIZE)
    fig.tight_layout()
    return (fig, axs)


def plot_rq4_results(
        data_lists: List[List[Dict[str, list]]],
        colors_dict: Dict[str, Tuple[float]],
        env_seeds: List[int],
        use_cases: List[str] = None,
        ylabels: List[str] = None
    ):
    """Boxplot of the n study analysis."""

    if use_cases is None:
        use_cases = np.arange(np.max([len(l) for l in data_lists]))

    nrows = len(data_lists)
    ncols = len(use_cases)

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(6*ncols, 3.5*nrows), #4
        sharex="all", sharey="none")

    if ylabels is None:
        ylabels = ["" for _ in range(nrows)]
    for ax in axes.flat:
        ax.grid(axis="y", color="0.9", linestyle="-", linewidth=1)

    xticks = []
    xtick_labels = []

    num_methods = max(
        sum([[len(d.keys()) for d in sub_list] for sub_list in data_lists], [])
    )

    space_between_method = 15

    for i in range(num_methods):
        for k, env_seed in enumerate(env_seeds):
            xtick_labels.append(f"n={env_seed}")
            xticks.append(2 + (3 * k + space_between_method * i))

    def _box_data(median, q1, q3, label=None):
        box_data = {
            "label": label,
            "med": median,
            "q1": q1,
            "q3": q3,
            "whislo": q1,
            "whishi": q3
        }
        return box_data

    # FLAT_BOX_THRESHOLD = 30


    # per row (i.e. metric/result)
    for r, data in enumerate(data_lists):
        axes[r][0].set_ylabel(ylabels[r], fontsize=AXIS_LABEL_FONTSIZE)

        for c in range(ncols):
            ax = axes[r][c]
            case_dict = data[c]

            colors = []
            box_data = []
            positions = []

            for k, (name, res_list) in enumerate(case_dict.items()):
                # list of stats results per method (for different n values)
                for i, (y, perc_25, perc_75) in enumerate(res_list):
                    color = colors_dict[name]
                    colors.append(color)

                    label, _ = post_process_label(name)

                    box_data.append(_box_data(y[-1], perc_25[-1], perc_75[-1], label=label))
                    # does not use the xticks directly in case of missing data
                    positions.append(xticks[i + k * len(env_seeds)])

            boxplot = ax.bxp(box_data, positions=positions, showfliers=False, patch_artist=True, widths=1.0)
            for i, (box_patch, median_line, color) in enumerate(zip(boxplot["boxes"], boxplot["medians"], colors)):
                # path = box_patch.get_path().vertices
                # q1 = path[0][1]
                # q3 = path[2][1]
                # height = abs(q3 - q1)
                # print(height)

                # if height < FLAT_BOX_THRESHOLD:
                #     line_color = color
                # else:
                #     line_color = "black"
                box_patch.set_facecolor(color)
                box_patch.set_edgecolor("black")
                median_line.set_color("black")
                median_line.set_linewidth(3)


    for c in range(ncols):
        axes[0][c].set_title(use_cases[c], fontsize=TITLE_LABEL_FONTSIZE)

    axes[0][0].set_xticks(xticks)
    axes[0][0].set_xticklabels(xtick_labels)
    for ax in axes[-1]:
        ax.tick_params(axis="x", labelrotation=45, labelsize=12)
    ax = axes.flat[-1]
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin - 1, xmax + 1)

    fig.tight_layout()

    def legend_axis(ax, **kwargs):
        method_names = set()
        for data in data_lists:
            for d in data:
                for k in d.keys():
                    method_names.add(k)

        legend_handles = [
            Patch(facecolor=colors_dict[name], edgecolor="black", label=post_process_label(name)[0])
            for name in method_names
        ]
        legend_kwargs = {
            "handles": legend_handles,
            "labelspacing": 1.1,
            "handletextpad": 1.0,
            "handlelength": 3, # default is 2
            "borderpad": 0.5,
            "borderaxespad": 1.0,
            "prop": {"size": 12}
        }
        legend_kwargs.update(kwargs)
        legend = ax.legend(**legend_kwargs)
        legend_frame = legend.get_frame()
        legend_frame.set_facecolor("0.9")
        legend_frame.set_edgecolor("0.9")
        return ax

    return (fig, axes), legend_axis


##############################################################################
################################## MAIN ######################################


def load_results_data():
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


# exec(open('result_analysis.py').read())
if __name__ == "__main__":
    torch.set_num_threads(1)
    folder = "data_new"

    ####################### Raw data loading #######################

    first_results = load_results_data()

    use_cases, method_names, colors_dict = color_data(first_results)
    print(method_names, use_cases, len(first_results))

    ####################### Analysis computation #######################

    # fault detection
    rq1_data = compute_rq1_results(first_results)

    # expert behavior coverage
    ebs_cov, efbs_cov = compute_expert_behaviors_coverage(first_results)
    # stores the results of the analysis
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

    print("Analysis computation done. Attempting to plot the results...")

    ########################### Plotting ###########################

    try:
        fig1, axs1 = plot_rq1_results(use_cases, colors_dict, rq1_data)
        for ax in axs1.flat:
            ax.tick_params(axis="both", labelsize=AXIS_TICKLABELS_FONTSIZE)
            legend = ax.legend_
            if legend is not None:
                for line in legend.get_lines():
                    plt.setp(line, linewidth=LEGEND_LINEWIDTH)
        fig1.savefig(f"{folder}/rq1.png")
    except:
        print("failed to plot fault detection.")

    try:
        fig2, axs2 = plot_rq2_ebs_results(use_cases, colors_dict, ebs_cov, efbs_cov)
        axs2[0][-1].legend_ = None
        axs2[1][-1].legend_ = None
        legend = axs2[-1][-1].legend_
        for line in legend.get_lines():
            plt.setp(line, linewidth=AXIS_TICKLABELS_FONTSIZE)
        for ax in axs2.flat:
            ax.tick_params(axis="both", labelsize=LEGEND_LINEWIDTH)
        fig2.savefig(f"{folder}/rq21.png")
    except:
        print("failed to plot expert coverage.")

    try:
        fig2, axs2 = plot_rq2_fobs_results(cases, colors_dict, obs_coverage_results, fobs_coverage_results)
        axs2[0][-1].legend_ = None
        axs2[1][-1].legend_ = None
        legend = axs2[-1][-1].legend_
        for line in legend.get_lines():
            plt.setp(line, linewidth=LEGEND_LINEWIDTH)
        for ax in axs2.flat:
            ax.tick_params(axis="both", labelsize=AXIS_TICKLABELS_FONTSIZE)
        fig2.savefig(f"{folder}/rq22.png")
    except:
        print("failed to plot final state coverage.")

    try:
        fig, axs = plot_summary_results(
            [rq1_data, ebs_cov, efbs_cov, obs_coverage_results, fobs_coverage_results],
            colors_dict,
            USE_CASES,
            ["#Faults", "#Expert Behaviors", "#Faulty Expert Behaviors", "#Final States", "#Faulty Final States"],
            sharey="none",
            figsize=(3.65, 8)
        )
        for ax in axs.flat:
            ax.tick_params(axis="both", labelsize=AXIS_TICKLABELS_FONTSIZE)
            legend = ax.legend_
            if legend is not None:
                for line in legend.get_lines():
                    plt.setp(line, linewidth=LEGEND_LINEWIDTH)
        fig.set_facecolor("white")
        fig.savefig(f"{folder}/rq.png")
    except:
        print("failed to plot summary plot.")


    with open(f"{folder}/colors_dict.json", "w") as file:
        json.dump(colors_dict, file)

    print("DONE.")