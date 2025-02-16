from functools import partial
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from pathlib import Path


sns.set_context("paper", font_scale=1.5, rc={"figure.figsize": [8, 5]})


def apply_df(model_value, instance, baseline):
    baseline_value = baseline.loc[instance]
    return (model_value - baseline_value) / baseline_value


def main():
    dynamics = sys.argv[1]
    assert dynamics in ["unicycle1_v0", "unicycle2_v0"]
    results_path = Path(f"data/results/{dynamics}/")
    benchmarks = list(results_path.glob("*.parquet"))
    benchmark = pd.read_parquet(benchmarks[0])
    baseline = benchmark[benchmark.name == "original"]
    bl_median = baseline.groupby("instance").median(numeric_only=True)
    bl_mean = baseline.groupby("instance").mean(numeric_only=True)
    models = pd.DataFrame(benchmark[benchmark.name != "original"])
    apply_success = partial(apply_df, baseline=bl_mean["success"])
    models["rel_s"] = models.apply(
        lambda x: apply_success(x.success, x.instance), axis=1
    )
    apply_cost = partial(apply_df, baseline=bl_median["cost"])
    models["rel_c"] = models.apply(lambda x: apply_cost(x.cost, x.instance), axis=1)
    apply_duration = partial(apply_df, baseline=bl_median["duration"])
    models["rel_d"] = models.apply(
        lambda x: apply_duration(x.duration, x.instance), axis=1
    )
    fig, ax = plt.subplots(1, 3, sharey=True)
    sns.boxplot(models, y="rel_s", ax=ax[0])
    sns.boxplot(models, y="rel_d", ax=ax[1])
    sns.boxplot(models, y="rel_c", ax=ax[2])
    for a in ax:
        a.set_yscale("symlog")
        a.axhline(y=0, color="red", linestyle="--")
        a.set_xticks(a.get_xticks(), a.get_xticklabels())
        a.grid(axis="y")
    plt.show()

    pass


if __name__ == "__main__":
    main()
