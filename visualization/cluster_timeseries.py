""" With this script you can visualize a cluster as a time series. You
can select to see the the HT voltage breakdowns or hide them. Furthermore,
you can look at a whole stability period instead of only at a certain cluster.

How to use it
-------------
Before you can run the script, you will probably need to set it up.

In the main function edit the input files for the different years and if
applicable add more years. The `input_file` has to be a .csv file as produced
by the clustering notebook, in particular every row represents a data point, 
the first row have to be the column names, the first column the timestamps of 
each data point.

Edit the `features` list to include all source features you are interested 
in seeing. Note that these features have to be columns in your input file.


Command line Arguments
-----------------------
Once everything is configured properly you can run the script with
``python cluster_timeseries.py [-param][option]``. You can give it
several command line parameters that are described also with the
help flag (-h).

-y: Here you can pass the year you are interested in, depending on what
you have configured in the main method. (default 2018)

-s: Pass a 1 to see the clusters of the stable periods and a 0 for the 
unstable ones. (default 1)

-c: Pass the cluster id of the cluster you want to visualize. If you do not
pass anything (None), then the whole stability period will be plotted. 
(default None)

-b: Pass True if you want to display the voltage breakdowns and False
otherwise. (default False)
"""

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

    if year == 2018:
        input_file = "../Data_Clustered/JanNov2018_sparks_clustered_forward.csv"
        # features.append(SourceFeatures.SAIREM2_FORWARDPOWER)
    elif year == 2016:
        input_file = "../Data_Clustered/JanNov2016.csv"
        # features.append(SourceFeatures.THOMSON_FORWARDPOWER)

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
