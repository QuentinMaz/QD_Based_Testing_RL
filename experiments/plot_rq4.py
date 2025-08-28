import json
import torch

from matplotlib import patches
from common import assemble_n_results
from result_analysis import (
    USE_CASES,
    AXIS_TICKLABELS_FONTSIZE,
    LEGEND_LINEWIDTH,
    plot_rq4_results
)

if __name__ == "__main__":
    torch.set_num_threads(1)

    with open("colors_dict.json", "r") as f:
        colors_dict = json.load(f)

    suffix = "MAE+LS"
    for k in ["MAP-Elites", "Novelty Search"]:
        colors_dict[k] = colors_dict[f"{k} {suffix}"]

    env_seeds = [1, 3, 5, 10]
    use_cases = ["Bipedal Walker", "Highway", "Lunar Lander"]


    data_folders = ["data_n_1", "data_angle_3", "data_angle_5", "data_angle_10"]

    # FD
    rq1_data = assemble_n_results(
        data_folders=data_folders,
        use_cases=use_cases,
        suffix=suffix,
        metric="rq1"
    )
    # EBS Coverage
    ebs_data = assemble_n_results(
        data_folders=data_folders,
        use_cases=use_cases,
        suffix=suffix,
        metric="bs_cov"
    )
    febs_data = assemble_n_results(
        data_folders=data_folders,
        use_cases=use_cases,
        suffix=suffix,
        metric="fbs_cov"
    )
    # FOBS Coverage
    obs_data = assemble_n_results(
        data_folders=data_folders,
        use_cases=use_cases,
        suffix=suffix,
        metric="obs_cov"
    )
    fobs_data = assemble_n_results(
        data_folders=data_folders,
        use_cases=use_cases,
        suffix=suffix,
        metric="fobs_cov"
    )

    try:
        (fig, axes), legend_axis = plot_rq4_results(
            data_lists=[rq1_data, febs_data, fobs_data],
            colors_dict=colors_dict,
            env_seeds=env_seeds,
            use_cases=use_cases,
            ylabels=["#Faults", "#Faulty Expert Behaviors", "#Faulty Final States"],
        )

        legend_axis(axes[0][1])
        [ax.tick_params("x", labelsize=16) for ax in axes[-1]]
        [ax.tick_params("y", labelsize=14) for ax in axes.flat]
        fig.tight_layout()
        fig.set_facecolor("white")
        fig.savefig("boxplot_fault.png")
    except:
        print("Failed to boxplot for fault discovery and diversity.")

    try:
        (fig, axes), legend_axis = plot_rq4_results(
            data_lists=[ebs_data, obs_data],
            colors_dict=colors_dict,
            env_seeds=env_seeds,
            use_cases=use_cases,
            ylabels=["#Expert Behaviors", "#Final States"],
        )
        legend_axis(axes[0][1], ncol=2, loc="upper center", columnspacing=0.75)
        fig.set_facecolor("white")
        [ax.tick_params("x", labelsize=16) for ax in axes[-1]]
        [ax.tick_params("y", labelsize=14) for ax in axes.flat]

        ax1 = axes[0][1]
        rect = patches.Rectangle(
            (33.5, 335),
            width=6.25,
            height=45,
            edgecolor="black",
            alpha=0.9,
            linestyle="dashed",
            linewidth=2,
            fill=False,
        )
        ax1.patches.clear()
        ax1.add_patch(rect)

        ax2 = axes[1][0]
        rect2 = patches.Rectangle(
            (15.5, 2100),
            width=6.25,
            height=900,
            edgecolor="black",
            alpha=0.9,
            linestyle="dashed",
            linewidth=2,
            fill=False,
        )
        rect3 = patches.Rectangle(
            (45.5, 2350),
            width=6.25,
            height=1000,
            edgecolor="black",
            alpha=0.9,
            linestyle="dashed",
            linewidth=2,
            fill=False,
        )
        ax2.patches.clear()
        ax2.add_patch(rect2)
        ax2.add_patch(rect3)
        fig.savefig("boxplot_coverage.png")
    except:
        print("Failed to boxplot for testing diversity.")