import os
from pathlib import Path
import json
import warnings

from matplotlib import pyplot as plt
from common import dump_results, reduce_results
from result_analysis import (
    compute_expert_behaviors_coverage,
    compute_obs_coverage,
    compute_rq1_results, fetch_result_data, color_data, plot_rq1_results,
    plot_rq2_ebs_results, plot_rq2_fobs_results
)

import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(
        prog="Perform Study",
        description="Perform computation and plotting of RQ1 and R2.1/2 given a result folder.",
    )
    parser.add_argument(
        "--result_folder",
        type=str,
        help="Name of the folder. We assume that the same folder exists in ../highway.",
        required=True,
    )
    parser.add_argument(
        "--data_folder",
        type=str,
        required=True,
        help="Output folder for the data.",
    )
    return parser.parse_args()


FD = True
EBS_COV = True
OBS_COV = True

N_RESULT_FOLDERS = ["results_5", "results_10"]

SUFFIX = "MAE+LS"

if __name__ == "__main__":
    args = parse_arguments()

    result_folder = args.result_folder  # type: str
    data_folder = args.data_folder  # type: str

    if all(task == False for task in [FD, EBS_COV, OBS_COV]):
        print("Nothing to do: exiting...")
        exit(0)

    if not os.path.isdir(f"../highway/{result_folder}"):
        warnings.warn(f"Result folder not found in ../highway: Highway use case will be missing.", RuntimeWarning)

    with open("colors_dict.json", "r") as f:
        colors_dict = json.load(f)
    for k in ["MAP-Elites", "Novelty Search"]:
        colors_dict[k] = colors_dict[f"{k} {SUFFIX}"]

    print(f"QD-based Results will be using colors of configuration \"{SUFFIX}\".")

    results = fetch_result_data(result_folder)

    print(f"found {len(results)} results in folder \"{result_folder}\".")
    data_size = min([len(d["logs"]) for d in results])
    if not all([len(d["logs"]) == data_size for d in results]):
        warnings.warn(f"Not all the data has the same size: size of {data_size} will be considered instead.", RuntimeWarning)
    valid_data = [d if len(d["logs"]) == data_size else reduce_results(d, data_size) for d in results]


    use_cases = color_data(valid_data)[0]
    for u in use_cases:
        Path(f"{data_folder}/{u}").mkdir(parents=True, exist_ok=True)

    if EBS_COV:
        ebs_cov, febs_cov = compute_expert_behaviors_coverage(valid_data)
        dump_results(ebs_cov,[f"{data_folder}/{case}/bs_cov" for case in use_cases])
        dump_results(febs_cov,[f"{data_folder}/{case}/fbs_cov" for case in use_cases])
        try:
            fig2, axs2 = plot_rq2_ebs_results(use_cases, colors_dict, ebs_cov, febs_cov)
            axs2[0][-1].legend_ = None
            axs2[1][-1].legend_ = None
            legend = axs2[-1][-1].legend_
            for line in legend.get_lines():
                plt.setp(line, linewidth=4)
            for ax in axs2.flat:
                ax.tick_params(axis="both", labelsize=12)
            fig2.savefig(f"{data_folder}/rq21.png")
        except:
            print("failed to plot expert coverage.")

    if OBS_COV:
        obs_cov, fobs_cov = compute_obs_coverage(valid_data)[1:]
        dump_results(obs_cov,[f"{data_folder}/{case}/obs_cov" for case in use_cases])
        dump_results(fobs_cov,[f"{data_folder}/{case}/fobs_cov" for case in use_cases])
        try:
            fig2, axs2 = plot_rq2_fobs_results(use_cases, colors_dict, obs_cov, fobs_cov)
            axs2[0][-1].legend_ = None
            axs2[1][-1].legend_ = None
            legend = axs2[-1][-1].legend_
            for line in legend.get_lines():
                plt.setp(line, linewidth=4)
            for ax in axs2.flat:
                ax.tick_params(axis="both", labelsize=12)
            fig2.savefig(f"{data_folder}/rq22.png")
        except:
            print("failed to plot final observation coverage.")


    if FD:
        rq1_results = compute_rq1_results(valid_data)
        dump_results(rq1_results, [f"{data_folder}/{case}/rq1" for case in use_cases])
        try:
            fig, axs = plot_rq1_results(use_cases, colors_dict, rq1_results)
            for ax in axs.flat:
                ax.tick_params(axis="both",labelsize=13)
                legend = ax.legend_
                if legend is not None:
                    for line in legend.get_lines():
                        plt.setp(line, linewidth=4)
            fig.savefig(f"{data_folder}/rq1.png")
        except:
            print("failed to plot fault detection.")







