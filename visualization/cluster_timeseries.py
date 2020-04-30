import pandas as pd

pd.plotting.register_matplotlib_converters()
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import Formatter
import numpy as np
import sys
import os

import argparse

sys.path.insert(1, os.path.abspath("../ionsrcopt"))
import load_data as ld
from source_features import SourceFeatures
from processing_features import ProcessingFeatures


def main(year, source_stability, cluster, show_breakdowns):
    ######################
    ###### SETTINGS ######
    ######################

    features = [
        # SourceFeatures.BIASDISCAQNV,
        # SourceFeatures.GASAQN,
        # SourceFeatures.OVEN1AQNP,
        # SourceFeatures.OVEN2AQNP,
        # SourceFeatures.SOLINJ_CURRENT,
        # SourceFeatures.SOLCEN_CURRENT,
        # SourceFeatures.SOLEXT_CURRENT,
        SourceFeatures.SOURCEHTAQNI,
    ]  # Features to be displayed

    if year == 2018:
        input_file = "../Data_Clustered/JanNov2018_sparks_clustered_forward.csv"
        # features.append(SourceFeatures.SAIREM2_FORWARDPOWER)
    elif year == 2016:
        input_file = "../Data_Clustered/JanNov2016.csv"
        # features.append(SourceFeatures.THOMSON_FORWARDPOWER)

    features.append(SourceFeatures.BCT25_CURRENT)

    ######################
    ######## CODE ########
    ######################

    # Load file into a data frame
    df = ld.read_data_from_csv(input_file, None, None)
    df = ld.fill_columns(df, None, fill_nan_with_zeros=True)
    df = ld.convert_column_types(df)

    if cluster is not None:
        df = df[(df[ProcessingFeatures.CLUSTER] == cluster)].copy()
    df = df.loc[df[ProcessingFeatures.SOURCE_STABILITY] == source_stability].copy()

    dates_nobreakdown = matplotlib.dates.date2num(
        df[df[ProcessingFeatures.HT_VOLTAGE_BREAKDOWN] == 0].index
    )
    dates_breakdown = matplotlib.dates.date2num(
        df[df[ProcessingFeatures.HT_VOLTAGE_BREAKDOWN] > 0].index
    )

    dates = df.index.values
    # datesIndices = np.arange(len(dates))

    fig, ax = plt.subplots(len(features), 1, sharex=True)
    for i, parameter in enumerate(features):
        # formatter = DateFormatter(dates)
        # ax[i].xaxis.set_major_formatter(formatter)
        ax[i].set_ylabel("{}".format(parameter), fontsize=13)
        ax[i].tick_params(axis="both", which="major")
        if show_breakdowns:
            # ax[i].plot(datesIndices[df[ProcessingFeatures.HT_VOLTAGE_BREAKDOWN] > 0], df.loc[df[ProcessingFeatures.HT_VOLTAGE_BREAKDOWN] > 0, parameter].values, linestyle='', marker='.', markersize=1, color='#ff7f0e')
            ax[i].plot_date(
                dates_breakdown,
                df.loc[
                    df[ProcessingFeatures.HT_VOLTAGE_BREAKDOWN] > 0, parameter
                ].values,
                linestyle="",
                marker=".",
                markersize=1,
                color="red",
            )
        # ax[i].plot(datesIndices[df[ProcessingFeatures.HT_VOLTAGE_BREAKDOWN] == 0], df.loc[df[ProcessingFeatures.HT_VOLTAGE_BREAKDOWN] == 0, parameter].values, linestyle='', marker='.', markersize=1, color='#1f77b4')
        ax[i].plot_date(
            dates_nobreakdown,
            df.loc[df[ProcessingFeatures.HT_VOLTAGE_BREAKDOWN] == 0, parameter].values,
            linestyle="",
            marker=".",
            markersize=1,
            color="black",
        )
        ax[i].grid(True)

    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    fig.suptitle("Time development of cluster {}".format(cluster))
    plt.subplots_adjust(
        left=0.05, bottom=0.05, right=0.95, top=0.93, wspace=None, hspace=0.4
    )
    plt.show()


class DateFormatter(Formatter):
    def __init__(self, dates, fmt="%Y-%m-%d %H:%M"):
        self.dates = dates
        self.fmt = fmt

    def __call__(self, x, pos=0):
        "Return the label for time x at position pos"
        ind = int(np.round(x))
        if ind >= len(self.dates) or ind < 0:
            return ""

        return pd.to_datetime(str(self.dates[ind])).strftime(self.fmt)


def parse_args():
    parser = argparse.ArgumentParser(description="View time development of clusters")
    parser.add_argument(
        "-s",
        "--source_stability",
        default=1,
        type=int,
        help="1 if you want to look at the stable source, 0 else",
    )
    parser.add_argument(
        "-y", "--year", default=2018, type=int, help="The year you are interested in"
    )
    parser.add_argument(
        "-c",
        "--cluster",
        default=None,
        type=int,
        help="The cluster you want to look at, or None for all data",
    )
    parser.add_argument(
        "-b",
        "--show_breakdowns",
        default=False,
        type=bool,
        help="True or False (default) if you want to display the breakdown points (in a different color)",
    )

    args = parser.parse_args()

    return {
        "source_stability": args.source_stability,
        "cluster": args.cluster,
        "show_breakdowns": args.show_breakdowns,
        "year": args.year,
    }


if __name__ == "__main__":
    args = parse_args()
    year = args["year"]
    source_stability = args["source_stability"]
    cluster = args["cluster"]
    show_breakdowns = args["show_breakdowns"]

    main(year, source_stability, cluster, show_breakdowns)
