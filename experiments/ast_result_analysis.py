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
from common import (
    dump_results,
    read_results_from_folder,
)

from result_analysis import AXIS_LABEL_FONTSIZE, TITLE_LABEL_FONTSIZE, color_data, compute_expert_behaviors_coverage, compute_obs_coverage, compute_rq1_results, plot_rq1_results, plot_rq2_ebs_results, plot_rq2_fobs_results

# USE_CASES = ["Bipedal Walker", "Lunar Lander", "Taxi"]

USE_CASES = ["Bipedal Walker", "Highway", "Lunar Lander"]

#################################################################################################################################
############################################################## HELPERS ##########################################################


def plot_rq3_results(
        data_lists: List[List[Dict]],
        colors_dict: Dict[str, Tuple[float]],
        use_cases: List[str] = None,
        ylabels: List[str] = None):
    if use_cases is None:
        use_cases = np.arange(np.max([len(l) for l in data_lists]))

    nrows = len(data_lists)
    ncols = len(use_cases)

    sharex = "all"

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4*nrows, 5*ncols), sharex=sharex, sharey="row")

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
                label = name
                if isinstance(to_plot, np.ndarray):
                    x = np.arange(len(to_plot))
                    assert len(x) == len(to_plot)
                    ax.plot(x, to_plot, color=color, label=label)
                else:
                    # assert np.all([len(x) == len(tmp) for tmp in to_plot])
                    y, perc_25, perc_75 = to_plot
                    x = np.arange(len(y))
                    ax.plot(x, y, color=color, label=label)
                    ax.fill_between(x, perc_25, perc_75, alpha=0.25, linewidth=0, color=color)
        if r % 2 == 0:
            ax = axs[r][1]
            legend = ax.legend(
                prop={"size": 10},
                labelspacing=1.1,
                handletextpad=1.05,
                borderpad=1.05,
                borderaxespad=1.0
            )
            legend_frame = legend.get_frame()
            legend_frame.set_facecolor("0.9")
            legend_frame.set_edgecolor("0.9")
    for c in range(ncols):
        axs[0][c].set_title(use_cases[c], fontsize=TITLE_LABEL_FONTSIZE)
    fig.tight_layout()
    return (fig, axs)


##############################################################################
################################## MAIN ######################################


def load_first_experiments_data():
    """Currently, loads the results of BW with descriptors [0, 1] and LL results using the force input space."""
    # LL
    ll_results = read_results_from_folder(
        "results/ll/qd/",
        include_final_states=True,
        include_expert_behaviors=True
        )
    [
        ll_results.extend(
            read_results_from_folder(f"results/ll/{m}/", include_final_states=True, include_expert_behaviors=True)
        )
        for m in ["ns", "rt", "mdpfuzz"]
    ]

    # BW
    bw_results = read_results_from_folder(
        "results/bw/qd/",
        include_final_states=True,
        include_expert_behaviors=True
    )
    [
        bw_results.extend(
            read_results_from_folder(f"results/bw/{m}/", include_final_states=True, include_expert_behaviors=True)
        )
        for m in ["ns", "rt", "mdpfuzz"]
    ]

    # TT
    # tt_results = read_results_from_folder(
    #     "results/tt/qd/",
    #     include_final_states=True,
    #     include_expert_behaviors=True
    # )
    # [
    #     tt_results.extend(
    #         read_results_from_folder(f"results/tt/{m}/", include_final_states=True, include_expert_behaviors=True)
    #     )
    #     for m in ["ns", "rt", "mdpfuzz"]
    # ]

    # return bw_results + ll_results + tt_results

    # HW
    hw_results = read_results_from_folder(
        "../highway/results_1/hw/qd/",
        include_final_states=True,
        include_expert_behaviors=True
    )
    [
        hw_results.extend(
            read_results_from_folder(f"../highway/results_1/hw/{m}/", include_final_states=True, include_expert_behaviors=True)
        )
        for m in ["ns", "rt", "mdpfuzz"]
    ]

    return bw_results + ll_results + hw_results


def load_second_experiments_data():
    bw_results = read_results_from_folder(
        "results/bw/qd/",
        include_final_states=True,
        include_expert_behaviors=True
    )
    [
        bw_results.extend(
            read_results_from_folder(f"results/bw/{m}/", include_final_states=True, include_expert_behaviors=True)
        )
        for m in ["ns", "rt", "mdpfuzz"]
    ]

    [
        bw_results.extend(
            read_results_from_folder(f"results/rq3/bw/{m}/", include_final_states=True, include_expert_behaviors=True)
        )
        for m in ["ns", "qd"]
    ]

    for d in bw_results:
        expert_indices = d["config"]["expert_indices"]
        d["config"]["use_case"] += f" {EXPERT_INDICES.index(expert_indices)}"

    return bw_results


# exec(open("result_analysis.py").read())
if __name__ == "__main__":
    torch.set_num_threads(1)

    ####################### Raw data loading #######################

    first_results = load_first_experiments_data()

    use_cases, method_names, colors_dict = color_data(first_results)
    print(method_names, use_cases, len(first_results))

    ####################### Analysis computation #######################

    # fault detection
    rq1_data = compute_rq1_results(first_results)

    # expert behavior coverage
    ebs_cov, efbs_cov = compute_expert_behaviors_coverage(first_results)
    # stores the results of the analysis
    folder = "data_ast"
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
    fig1.savefig(f"{folder}/rq1.png")

    fig2, axs2 = plot_rq2_ebs_results(use_cases, colors_dict, ebs_cov, efbs_cov)
    axs2[0][-1].legend_ = None
    axs2[1][-1].legend_ = None
    legend = axs2[-1][-1].legend_
    for line in legend.get_lines():
        plt.setp(line, linewidth=4)
    for ax in axs2.flat:
        ax.tick_params(axis="both", labelsize=12)
    fig2.savefig(f"{folder}/rq21.png")

    fig2, axs2 = plot_rq2_fobs_results(cases, colors_dict, obs_coverage_results, fobs_coverage_results)
    axs2[0][-1].legend_ = None
    axs2[1][-1].legend_ = None
    legend = axs2[-1][-1].legend_
    for line in legend.get_lines():
        plt.setp(line, linewidth=4)
    for ax in axs2.flat:
        ax.tick_params(axis="both", labelsize=12)
    fig2.savefig(f"{folder}/rq22.png")


    with open(f"{folder}/colors_dict.json", "w") as file:
        json.dump(colors_dict, file)

    print("FIRST ANALYSIS DONE. PERFORMING ANALYSIS FOR RQ3...")

    # fetches and renames the BW results

    bw_results = load_second_experiments_data()

    bw_cases, bw_names, bw_colors_dict = color_data(bw_results)
    print(bw_names, bw_cases)

    rq3_data = compute_rq1_results(bw_results)
    rq3_ebs_cov, rq3_efbs_cov = compute_expert_behaviors_coverage(bw_results)
    rq3_cases, rq3_obs_coverage_results, rq3_fobs_coverage_results = compute_obs_coverage(bw_results)

    # copies the BS-independant results (MDPFuzz and RT)
    rq3_results = [rq3_data, rq3_ebs_cov, rq3_efbs_cov, rq3_obs_coverage_results, rq3_fobs_coverage_results]
    for res in rq3_results:
        data_to_copy = {k: res[0][k] for k in ["Random Testing", "MDPFuzz"]}
        for r in res[1:]:
            r.update(data_to_copy)

    # saves the results

    folder += "_rq3"
    for case in bw_cases:
        sub_folder = f"{folder}/{case}"
        Path(sub_folder).mkdir(parents=True, exist_ok=True)

    for res_data, res_name in zip(rq3_results, ["rq1", "bs_cov", "fbs_cov", "obs_cov", "fobs_cov"]):
        dump_results(res_data, [f"{folder}/{case}/{res_name}" for case in bw_cases])

    # plots the results as done in the original paper
    fig4, axs4 = plot_rq3_results(
        rq3_results,
        bw_colors_dict,
        ["$Distance$ and $Hull$ $angle$", "$Torque$ and $Jump$", "$Hip$ $angles$", "$Hip$ $speeds$"],
        ["#Faults", "#Expert Behaviors", "#Faulty Expert Behaviors", "#Final States", "#Faulty Final States"]
    )
    fig4.savefig(f"{folder}/rq3.png")