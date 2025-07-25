from functools import partial
from matplotlib.axes import Axes
import pyperclip
import seaborn as sns
from typing import List
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.ticker import Locator
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import sys
from pathlib import Path


sns.set_context("paper", font_scale=2.5, rc={"figure.figsize": [18, 6]})
plt.rcParams.update(
    {
        "text.usetex": True,
        # "font.family": "Helvetica",
        "font.family": "STIXGeneral",
        # "mathtext.fontset":"stix",
        # "font.size": 60,
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amsfonts}",
        "figure.figsize": [18, 6],
    }
)


def plot_results(benchmark: pd.DataFrame, ax: List[Axes]):
    def log_floor(x, precision=0.0) -> float:
        return np.true_divide(np.floor(x * 10**precision), 10**precision)

    def apply_df(model_value, instance, baseline):
        baseline_value = baseline.loc[instance]
        return (model_value - baseline_value) / baseline_value

    baseline = benchmark[benchmark.name == "original"]
    bl_median = baseline.groupby("instance").median(numeric_only=True)
    bl_mean = baseline.groupby("instance").mean(numeric_only=True)
    # models = pd.DataFrame(benchmark[benchmark.name != "original"])
    apply_success = partial(apply_df, baseline=bl_mean["success"])
    benchmark["rel_s"] = benchmark.apply(
        lambda x: apply_success(x.success, x.instance), axis=1
    )
    apply_bcost = partial(apply_df, baseline=bl_median["best_cost"])
    benchmark["rel_bc"] = benchmark.apply(
        lambda x: apply_bcost(x.best_cost, x.instance), axis=1
    )
    apply_fcost = partial(apply_df, baseline=bl_median["first_cost"])
    benchmark["rel_fc"] = benchmark.apply(
        lambda x: apply_fcost(x.first_cost, x.instance), axis=1
    )
    apply_duration = partial(apply_df, baseline=bl_median["duration"])
    benchmark["rel_d"] = benchmark.apply(
        lambda x: apply_duration(x.duration, x.instance), axis=1
    )
    lintresh = []
    thresh_df = (
        benchmark.groupby("name")[["rel_d", "rel_fc", "rel_bc"]]
        .quantile([0.25, 0.75])  # pyright:ignore
        .abs()
        .dropna()
        .reset_index()
    )
    thresh_df = thresh_df[thresh_df.name == "model"]
    thresh_d = 10 ** log_floor(np.log10(thresh_df["rel_d"].min()))
    thresh_fc = 10 ** log_floor(np.log10(thresh_df["rel_fc"].min()))
    thresh_bc = 10 ** log_floor(np.log10(thresh_df["rel_bc"].min()))
    thresh = [thresh_d, thresh_fc, thresh_bc]
    # mean = benchmark.groupby("name")[["success"]].mean(numeric_only=True)
    # med = benchmark.groupby("name")[["rel_d", "rel_bc", "rel_fc"]].median(
    #     numeric_only=True
    # )
    # iqr = benchmark.groupby("name")[["rel_d", "rel_bc", "rel_fc"]].quantile(
    #     [0.25, 0.75], numeric_only=True
    # )
    # print(mean)
    # print(med)
    # print(iqr)
    agg_funcs = {
        "success": "mean",
        "rel_bc": ["median", lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)],
        "rel_fc": ["median", lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)],
        "rel_d": ["median", lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)],
    }

    # Group by 'name' and apply aggregations
    # if "delta_0" not in benchmark.columns:
    #     breakpoint()
    # result = benchmark.groupby(["delta_0", "n_mp", "name"]).agg(agg_funcs)
    result = benchmark.groupby(["name"]).agg(agg_funcs)

    # Rename columns for clarity
    result.columns = ["_".join(col).strip() for col in result.columns.values]
    result.rename(
        columns={
            "rel_bc_<lambda_0>": "rel_bc_25q",
            "rel_bc_<lambda_1>": "rel_bc_75q",
            "rel_fc_<lambda_0>": "rel_fc_25q",
            "rel_fc_<lambda_1>": "rel_fc_75q",
            "rel_d_<lambda_0>": "rel_d_25q",
            "rel_d_<lambda_1>": "rel_d_75q",
        },
        inplace=True,
    )
    # latex_table = result.to_latex(
    #     float_format="%.4f", caption="Aggregated Statistics by Name", label="tab:stats"
    # )
    # pdurationyperclip.copy(latex_table)
    result = result.reindex(
        [
            "success_mean",
            "rel_d_median",
            # "rel_d_25q",
            # "rel_d_75q",
            "rel_fc_median",
            # "rel_fc_25q",
            # "rel_fc_75q",
            "rel_bc_median",
            # "rel_bc_25q",
            # "rel_bc_75q",
        ],
        axis=1,
    )
    print(result.round(decimals=4))
    if len(sys.argv) == 2:
        return 0
    # fig, ax = plt.subplots(1, 3, sharey=True)
    # sns.barplot(benchmark, y="success", hue="name", ax=ax[0])
    order = ["Baseline", "Model"]
    benchmark["name"] = benchmark.apply(
        lambda x: x["name"].replace("model", "Model").replace("original", "Baseline"),
        axis=1,
    )
    g = sns.violinplot(benchmark, y="rel_d", hue="name", ax=ax[0], hue_order=order)
    # plt.legend(loc="upper left", bbox_to_anchor=(0, 0))
    handles, labels = ax[0].get_legend_handles_labels()
    g.get_legend().remove()
    # sns.move_legend(
    #     g,
    #     bbox_to_anchor=(1, 0),
    #     loc="lower right",
    #     bbox_transform=fig.transFigure,
    #     ncol=2,
    # )

    # g.get_legend().remove()
    g.set_title(r"$r_{d}[\%]$", pad=10)
    g.set(xlabel=None, ylabel=None)
    h = sns.violinplot(benchmark, y="rel_fc", hue="name", ax=ax[1], hue_order=order)
    h.set_title(r"$r_{c}^\mathrm{first}[\%]$", pad=10)
    h.set(xlabel=None, ylabel=None)
    h.get_legend().remove()
    i = sns.violinplot(benchmark, y="rel_bc", hue="name", ax=ax[2], hue_order=order)
    i.set_title(r"$r_{c}^\mathrm{best}[\%]$", pad=10)
    i.set(xlabel=None, ylabel=None)
    i.get_legend().remove()
    for i in range(3):
        lintresh = min(thresh)
        # ax[i].set_yscale("symlog", linthresh=lintresh)
        ax[i].axhline(y=0, color="red", linestyle="--")
        ax[i].set_xticks(ax[i].get_xticks(), ax[i].get_xticklabels())
        # ax[i].yaxis.set_major_locator(
        #     mticker.SymmetricalLogLocator(base=10, linthresh=lintresh)
        # )
        # ax[i].yaxis.set_minor_locator(
        #     mticker.SymmetricalLogLocator(
        #         base=10, linthresh=lintresh, subs=[x / 10 for x in range(1, 10)]
        #     )
        # )
        ax[i].grid(axis="y")
        # ax[i].set_xticks(
        #     ax[i].get_xticks(), ax[i].get_xticklabels(), rotation=90, ha="right"
        # )
    # fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)

    # .legend(frameon=False)
    # legend = plt.legend()
    # legend.get_frame().set_facecolor("none")
    # plt.savefig(sys.argv[2])
    # plt.show()


def main():
    # dynamics = sys.argv[1]
    paths = [
        Path(f"data/results/unicycle1_v0/unicycle1_v0.parquet"),
        Path(f"data/results/unicycle2_v0/unicycle2_v0.parquet"),
        Path(f"data/results/car1_v0/car1_v0.parquet"),
    ]

    # fig, axes = plt.subplots(1, 9, sharey=True)
    fig = plt.figure()
    gs = gridspec.GridSpec(1, 11, width_ratios=[1, 1, 1, 0.3, 1, 1, 1, 0.3, 1, 1, 1])
    axes = []
    first_ax = None  # Keep track of the first subplot for sharing y-axis
    for i in range(3):
        for j in range(3):
            ax = fig.add_subplot(gs[i * 4 + j], sharey=first_ax)  # Skip gap columns
            if first_ax is None:
                first_ax = ax  # First subplot retains the y-axis labels
            else:
                ax.label_outer()  # Hide redundant y-axis labels
            axes.append(ax)
    for i, path in enumerate(paths):
        benchmark = pd.read_parquet(path)

        plot_results(benchmark, axes[i * 3 : (i + 1) * 3])
    for ax in axes:
        ax.set_yscale("symlog", linthresh=1e-1)
    handles, labels = axes[0].get_legend_handles_labels()
    group_titles = [
        "$1^{\\text{st}}$-order Unicycle",
        "$2^{\\text{nd}}$-order Unicycle",
        "Car with trailer",
    ]
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)
    left_margin = fig.subplotpars.left  # Default ~0.125
    right_margin = fig.subplotpars.right  # Default ~0.9
    total_width = right_margin - left_margin  # Effective width of the subplots
    for i, title in enumerate(group_titles):
        middle_ax = axes[i * 3 + 1]  # Middle subplot of each group
        pos = middle_ax.get_position()  # Get subplot position
        x_position = (pos.x0 + pos.x1) / 2  # Compute the center x-position
        # x_position = left_margin + total_width * (
        #     (i * 3 + 1.5) / 9
        # )  # Adjust for margins
        fig.text(x_position, 0.95, title, ha="center", fontsize=24, fontweight="bold")
    # plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.subplots_adjust(bottom=0.2, top=0.85)
    fig.savefig("data/plots/combined_bench.eps")
    # plt.show()


if __name__ == "__main__":
    main()
