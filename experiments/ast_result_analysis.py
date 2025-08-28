import json
from pathlib import Path

import torch
from bw_framework import EXPERT_INDICES
from matplotlib import pyplot as plt
from common import (
    dump_results,
    read_results_from_folder,
)

from result_analysis import (
    LEGEND_LINEWIDTH,
    AXIS_TICKLABELS_FONTSIZE,
    USE_CASES,
    color_data,
    compute_expert_behaviors_coverage,
    compute_obs_coverage,
    compute_rq1_results,
    plot_rq1_results,
    plot_rq2_ebs_results,
    plot_rq2_fobs_results,
    plot_summary_results,
)


"""
Script that computes the analysis of the results of the original experiments (from the AST paper).
As such, it assumes they have been run.
"""


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

    # HW
    hw_results = read_results_from_folder(
        "../highway/results/hw/qd/",
        include_final_states=True,
        include_expert_behaviors=True
    )
    [
        hw_results.extend(
            read_results_from_folder(f"../highway/results/hw/{m}/", include_final_states=True, include_expert_behaviors=True)
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
    folder = "data_ast"

    ####################### Raw data loading #######################

    first_results = load_first_experiments_data()

    use_cases, method_names, colors_dict = color_data(first_results)

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
            figsize=(3.65, 6.5)
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


    try:
        # plots the results as done in the original paper
        fig4, axs4 = plot_summary_results(
            rq3_results,
            bw_colors_dict,
            ["$Distance$ and $Hull$ $angle$", "$Torque$ and $Jump$", "$Hip$ $angles$", "$Hip$ $speeds$"],
            ["#Faults", "#Expert Behaviors", "#Faulty Expert Behaviors", "#Final States", "#Faulty Final States"],
            figsize=(4, 5) # default one
        )
        for ax in axs4.flat:
            ax.tick_params(axis="both", labelsize=AXIS_TICKLABELS_FONTSIZE)
            legend = ax.legend_
            if legend is not None:
                for line in legend.get_lines():
                    plt.setp(line, linewidth=LEGEND_LINEWIDTH)
        fig4.set_facecolor("white")
        fig4.savefig(f"{folder}/rq3.png")
    except:
        print("failed to plot summary plot.")