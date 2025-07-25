import json

import torch
from common import assemble_n_results
from result_analysis import plot_n_results, plot_rq3_results


if __name__ == "__main__":
    torch.set_num_threads(1)

    with open("colors_dict.json", "r") as f:
        colors_dict = json.load(f)

    suffix = "MAE+LS"
    for k in ["MAP-Elites", "Novelty Search"]:
        colors_dict[k] = colors_dict[f"{k} {suffix}"]

    data_folders = ["data_1", "data_3", "data_5", "data_10"]
    env_seeds = [1, 3, 5, 10]
    use_cases = ["Bipedal Walker", "Highway", "Lunar Lander"]

    # FD
    rq1_data = assemble_n_results(
        data_folders=data_folders,
        suffix=suffix,
        metric="rq1"
    )
    # fig = plot_n_results(
    #     use_cases=use_cases,
    #     env_seeds=env_seeds,
    #     results=rq1_data,
    #     colors_dict=colors_dict,
    #     x_axis="executions"
    # )[0]
    # fig.set_facecolor("white")
    # fig.savefig(f"n_rq1.png")

    # EBS Coverage
    ebs_data = assemble_n_results(
        data_folders=data_folders,
        suffix=suffix,
        metric="bs_cov"
    )
    febs_data = assemble_n_results(
        data_folders=data_folders,
        suffix=suffix,
        metric="fbs_cov"
    )
    # fig = plot_n_results(
    #     use_cases=use_cases,
    #     env_seeds=env_seeds,
    #     results=ebs_data,
    #     colors_dict=colors_dict,
    #     additional_results=febs_data,
    #     x_axis="executions"
    # )[0]
    # fig.set_facecolor("white")
    # fig.savefig(f"n_ebs+febs_cov.png")

    # FOBS Coverage
    obs_data = assemble_n_results(
        data_folders=data_folders,
        suffix=suffix,
        metric="obs_cov"
    )
    fobs_data = assemble_n_results(
        data_folders=data_folders,
        suffix=suffix,
        metric="fobs_cov"
    )
    # removes Highway results in OBS cov
    # hw_data = obs_data[1]
    # for k in hw_data.keys():
    #     hw_data[k] = []

    # fig = plot_n_results(
    #     use_cases=use_cases,
    #     env_seeds=env_seeds,
    #     results=obs_data,
    #     colors_dict=colors_dict,
    #     additional_results=fobs_data,
    #     x_axis="executions"
    # )[0]
    # fig.set_facecolor("white")
    # fig.savefig(f"n_obs+fobs_cov.png")

    (fig, axes), add_legend = plot_rq3_results(
        data_lists=[rq1_data, ebs_data, febs_data, obs_data, fobs_data],
        colors_dict=colors_dict,
        env_seeds=env_seeds,
        use_cases=use_cases,
        ylabels=["#Faults", "#Expert Behaviors", "#Faulty Expert Behaviors", "#Final States", "#Faulty Final States"],
        x_axis="executions"
    )
    add_legend(axes[1][0])
    fig.set_facecolor("white")
    fig.savefig(f"n_analysis_no_shade.png")