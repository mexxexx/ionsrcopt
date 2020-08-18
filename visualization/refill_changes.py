import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import argparse


def main(year):
    features = [
        "bias disc",
        "gas",
        "oven1",
        "RF",
        "solinj",
        "solcen",
        "solext",
        "HTI",
        "BCT25",
    ]
    features = [(f, "median") for f in features]

    if year == 2018:
        input_file = "./Results/2018_stable_refills_updated.csv"
        output_file = "./Results/2018_refill_changes.csv"
    elif year == 2016:
        input_file = "./Results/2016_stable_refills_updated.csv"
        output_file = "./Results/2016_refill_changes.csv"

    df = pd.read_csv(input_file, index_col=0, header=[0, 1])
    refill_changes = get_refill_changes(df, features)

    fig, axs = plt.subplots(3, 2)

    plot(axs[0, 0], refill_changes, "bias disc", "V")
    plot(axs[1, 0], refill_changes, "gas", "V")
    plot(axs[2, 0], refill_changes, "RF", "W")
    plot(axs[0, 1], refill_changes, "solinj", "A")
    plot(axs[1, 1], refill_changes, "solcen", "A")
    plot(axs[2, 1], refill_changes, "solext", "A")

    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    fig.suptitle("Parameter changes in between oven refills ({})".format(year))
    plt.subplots_adjust(
        left=0.05, bottom=0.07, right=0.95, top=0.93, wspace=None, hspace=0.4
    )

    plt.show()

    refill_changes.round(3).to_csv(output_file)


def plot(ax, refill_changes, feature, unit):
    ax.plot(refill_changes[feature], ls="", marker="o")
    ax.set_xticks(np.arange(refill_changes.index.min(), refill_changes.index.max() + 1))
    ax.grid(True)
    ax.set_ylabel("{} Δ in {}".format(feature, unit))
    ax.set_xlabel("Refill #")


def get_first_cluster(group):
    min_index_delta = np.argmin(group[("REFILL", "delta_in_hours")].values)
    return group.iloc[min_index_delta, :]


def get_last_cluster(group):
    max_index_delta = np.argmax(group[("REFILL", "delta_in_hours")].values)
    return group.iloc[max_index_delta, :]


def get_refill_changes(df, features):
    first_clusters = df.groupby(("REFILL", "index")).apply(get_first_cluster)
    last_clusters = df.groupby(("REFILL", "index")).apply(get_last_cluster)

    result = pd.DataFrame()

    for refill in first_clusters.index:
        if refill - 1 in last_clusters.index:
            new_row = (
                first_clusters.loc[refill, features]
                - last_clusters.loc[refill - 1, features]
            )
            new_row = pd.DataFrame(new_row).T
            new_row.index = [refill]
            result = result.append(new_row)

    result.columns = result.columns.droplevel(1)
    result.index.name = "refill"
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="How did settings change over refills?"
    )
    parser.add_argument(
        "-y", "--year", default=2018, type=int, help="The year you are interested in"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(year=args.year)
