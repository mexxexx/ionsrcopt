""" The ``cluster_description.py`` script creates the summaries for the clusters, i.e. you 
can use it do display median, standard deviation, and other
statistical properties, as well as further information such as oven
refills, breakdown rate and duration.

The result are sorted in descending order based on the total duration.

How to use it
-------------
Before you use the script for the first time, you probably have 
to configure it.

In the main function edit the input files for the different years and if
applicable add more years. The `input_file` has to be a .csv file as produced
by the clustering notebook, in particular every row represents a data point, 
the first row have to be the column names, the first column the timestamps of 
each data point.

Edit the `features` list to include all source features you are interested 
in seeing. Note that these features have to be columns in your input file.

Then, configure the statistics you are interested in by adding them to
the `statistics` list. Available are

* mean
* median
* std (standard deviation)
* std% (standard deviation in percent of the mean)
* avg_dev (average deviation of mean, i.e. \sum_i \abs{x_i - mean})
* min (minimum value)
* 25% (lower quartile)
* 75% (upper quartile)
* max (maximum value)

Command line Arguments
-----------------------
Once everything is configured properly you can run the script with
``python cluster_description.py [-param][option]``. You can give it
several command line parameters that are described also with the
help flag (-h).

-y: Here you can pass the year you are interested in, depending on what
you have configured in the main method. (default 2018)

-s: Pass a 1 to see the clusters of the stable periods and a 0 for the 
unstable ones. (default 1)

-f: Pass a y if you want to save the results to a file, otherwise (passing n) 
they will be printed to the console. (default n)

-b: Pass a y if you want to know the breakdown rate per hour, otherwise n.
(default n)

-v: The number of clusters to visualize. (default 20)
"""

import pandas as pd

pd.set_option("display.max_columns", 500)
pd.set_option("display.expand_frame_repr", False)
import numpy as np
import sys
import os

from statsmodels.stats.weightstats import DescrStatsW
from sklearn import preprocessing

import argparse

sys.path.insert(1, os.path.abspath("../ionsrcopt"))
import load_data as ld
from source_features import SourceFeatures
from processing_features import ProcessingFeatures
import cluster_metrics


def main(
    year,
    source_stability,
    count_breakdowns_per_cluster,
    num_clusters_to_visualize,
    print_to_file,
    display_metrics,
):
    ######################
    ###### SETTINGS ######
    ######################

    features = [
        SourceFeatures.BIASDISCAQNV,
        SourceFeatures.GASAQN,
        SourceFeatures.OVEN1AQNP,
        # SourceFeatures.OVEN2AQNP,
        SourceFeatures.SOLINJ_CURRENT,
        SourceFeatures.SOLCEN_CURRENT,
        SourceFeatures.SOLEXT_CURRENT,
        SourceFeatures.SOURCEHTAQNI,
        SourceFeatures.BCT25_CURRENT,
    ]  # Features to load

    if year == 2018:
        input_file = "../Data_Clustered/JanNov2018_sparks_clustered.csv"
        output_file = "./Results/2018_{}_sparks.csv".format(source_stability)
        features.append(SourceFeatures.SAIREM2_FORWARDPOWER)
    elif year == 2016:
        input_file = "../Data_Clustered/JanNov2016_sparks_clustered.csv"
        output_file = "./Results/2016_{}_sparks.csv".format(source_stability)
        features.append(SourceFeatures.THOMSON_FORWARDPOWER)
    elif year == 2015:
        input_file = "../Data_Clustered/MayDec2015_sparks_clustered.csv"
        output_file = "./Results/2015_{}_sparks.csv".format(source_stability)
        features.append(SourceFeatures.THOMSON_FORWARDPOWER)

    statistics = ["median", "std%"]  # Statistics we are interested in

    ######################
    ######## CODE ########
    ######################

    # Load file into a data frame
    df = ld.read_data_from_csv(input_file, None, None)
    df = ld.fill_columns(df, None, fill_nan_with_zeros=True)
    df = ld.convert_column_types(df)

    for feature in features:
        if feature not in df.columns:
            print(
                "{} does not exist as a feature in the loaded file. Aborting.".format(
                    feature
                )
            )
            return

    # Calculate oven refills
    oven_refill_ends = calculate_oven_refill_ends(df[SourceFeatures.OVEN1AQNP])
    if year == 2018:
        oven_refill_ends = clear_refills_2018(oven_refill_ends)
    elif year == 2016:
        oven_refill_ends = clear_refills_2016(oven_refill_ends)

    print("There were {} oven refills.".format(len(oven_refill_ends)))

    # Select only the stability interested in
    df = df[df[ProcessingFeatures.SOURCE_STABILITY] == source_stability].copy()
    total_duration = df[ProcessingFeatures.DATAPOINT_DURATION].sum() / 3600

    # Describe the clusters
    print("Calculating statistics...")
    described = df.groupby(ProcessingFeatures.CLUSTER).apply(
        describe_cluster,
        features=features,
        weight_column=ProcessingFeatures.DATAPOINT_DURATION,
        oven_refills=oven_refill_ends,
    )
    described[("DENSITY", "percentage")] = (
        described[("DURATION", "in_hours")] / total_duration * 100
    )

    # Gather statistics to output
    wanted_statistics = get_wanted_statistics(features, statistics) + [
        ("DENSITY", "percentage"),
        ("DURATION", "in_hours"),
        ("DURATION", "longest_in_hours"),
        ("DURATION", "num_splits"),
        ("REFILL", "index"),
        ("REFILL", "delta_in_hours"),
    ]
    if count_breakdowns_per_cluster:
        wanted_statistics += [("num_breakdowns", "per_hour")]

    # Calculate metrics
    if display_metrics:
        metrics = calculate_metrics(df, features)
        print("DBI is {}".format(np.mean(metrics["DBI"])))
        described.loc[described.index >= 0, ("METRICS", "DBI")] = metrics["DBI"]
        print("Silhouette is {}".format(np.mean(metrics["silhouette"])))
        described.loc[described.index >= 0, ("METRICS", "silhouette")] = metrics[
            "silhouette"
        ]

        wanted_statistics += [("METRICS", "DBI"), ("METRICS", "silhouette")]

    described.sort_values(by=[("DENSITY", "percentage")], ascending=False, inplace=True)

    print("Rounding values...")
    printable_clusters = described[wanted_statistics].head(n=num_clusters_to_visualize)
    print(
        "Sum of densities of printed clusters: {:.1f}%".format(
            printable_clusters[("DENSITY", "percentage")].sum()
        )
    )
    print(
        "Sum of duration of printed clusters when source was running: {:.1f}".format(
            printable_clusters.loc[
                printable_clusters.index >= 0, ("DURATION", "in_hours")
            ].sum()
        )
    )
    printable_clusters = round_described(
        printable_clusters,
        {
            SourceFeatures.BIASDISCAQNV: 0,
            SourceFeatures.GASAQN: 2,
            SourceFeatures.OVEN1AQNP: 1,
            SourceFeatures.OVEN2AQNP: 1,
            SourceFeatures.THOMSON_FORWARDPOWER: 0,
            SourceFeatures.SAIREM2_FORWARDPOWER: 0,
            SourceFeatures.SOLINJ_CURRENT: 0,
            SourceFeatures.SOLCEN_CURRENT: 0,
            SourceFeatures.SOLEXT_CURRENT: 0,
            SourceFeatures.SOURCEHTAQNI: 2,
            SourceFeatures.BCT25_CURRENT: 3,
        },
    )
    printable_clusters.rename(
        {
            SourceFeatures.BIASDISCAQNV: "bias disc",
            SourceFeatures.GASAQN: "gas",
            SourceFeatures.OVEN1AQNP: "oven1",
            SourceFeatures.OVEN2AQNP: "oven2",
            SourceFeatures.SAIREM2_FORWARDPOWER: "RF",
            SourceFeatures.THOMSON_FORWARDPOWER: "RF",
            SourceFeatures.SOLINJ_CURRENT: "solinj",
            SourceFeatures.SOLCEN_CURRENT: "solcen",
            SourceFeatures.SOLEXT_CURRENT: "solext",
            SourceFeatures.SOURCEHTAQNI: "HTI",
            SourceFeatures.BCT25_CURRENT: "BCT25",
        },
        axis="columns",
        inplace=True,
    )

    if print_to_file:
        printable_clusters.to_csv(output_file)
        print("Saved result to {}".format(output_file))
    else:
        print(printable_clusters)


def round_described(described, decimals):
    """ For printing purposes the results should be rounded.
    This is done with this function and the desired number of
    decimals are passed per parameter as an argument.

    Parameters
    ----------
    described : DataFrame
        The DataFrame you want to print
    decimals : dict
        Keys are parameters (columns of the described DF) and values
        are the number of decimal places you want to see. 
    
    Returns
    -------
    DataFrame
        Rounded version of ``described``
    """

    for k, v in decimals.items():
        described = described.round(
            {
                (k, "mean"): v,
                (k, "std"): min(v + 1, 3),
                (k, "std%"): min(v + 1, 3),
                (k, "avg_dev"): v,
                (k, "min"): v,
                (k, "25%"): v,
                (k, "median"): v,
                (k, "75%"): v,
                (k, "max"): v,
            }
        )

    return described.round(
        {
            ("DENSITY", "percentage"): 1,
            ("DURATION", "in_hours"): 1,
            ("DURATION", "longest_in_hours"): 1,
            ("DURATION", "num_splits"): 0,
            ("num_breakdowns", "per_hour"): 2,
            ("REFILL", "index"): 0,
            ("REFILL", "delta_in_hours"): 1,
        }
    )


def clear_refills_2018(oven_refill_ends):
    """ Some of the automatically found refills in 2018 are incorrect and
    result e.g. from short oven stops. They are cleared out with this
    function based on manual selection and comparison with CALS.

    Parameters
    ----------
    oven_refill_ends : list of timestamps
        The automatically detected _ends_ of the refill periods (time when the
        oven was ramped up again)

    Returns
    -------
    list of timestamps
        *End*-points of the periods that were real refills in 2018
    """

    remove_dates = [
        "2018-02-20 15h",
        "2018-03-27 10h",
        "2018-06-19 12h",
        "2018-06-23 03h",
        "2018-07-29 05h",
        "2018-08-15 15h",
        "2018-08-17 05h",
    ]

    return [
        r for r in oven_refill_ends if not r.strftime("%Y-%m-%d %Hh") in remove_dates
    ]


def clear_refills_2016(oven_refill_ends):
    """ Some of the automatically found refills in 2016 are incorrect and
    result e.g. from short oven stops. They are cleared out with this
    function based on manual selection and comparison with CALS.

    Parameters
    ----------
    oven_refill_ends : list of timestamps
        The automatically detected _ends_ of the refill periods (time when the
        oven was ramped up again)

    Returns
    -------
    list of timestamps
        *End*-points of the periods that were real refills in 2016
    """

    remove_dates = [
        "2016-01-04",
        "2016-01-19",
        "2016-01-20",
        "2016-01-22",
        "2016-01-26",
        "2016-02-15",
    ]

    return [r for r in oven_refill_ends if not r.strftime("%Y-%m-%d") in remove_dates]


def calculate_oven_refill_ends(oven1):
    """ Calculates possible time spans that mark an oven refill.

    These are periods where the *IP.NSRCGEN:OVEN1AQNP* was below 0.1W (off) 
    for at least 45min continuously.

    TODO: As the results have to be compared with the real data anyways, it
    might be better to manually enter the refill ends instead of calculating candidates,
    as there are not many for one year (20-30).

    Parameters
    ----------
    oven1 : Series
        Series with *IP.NSRCGEN:OVEN1AQNP* values

    Returns
    -------
    list of timestamps
        *End*-points of the periods that could have been oven refills.
        This list might contain false positives, so a manual comparison
        with the data from CALS has to be done.
    """

    result = []
    times_where_oven_is_off = oven1[oven1 < 0.1].index

    continuous_periods_starts = np.flatnonzero(
        (times_where_oven_is_off.to_series(keep_tz=True).diff(1))
        .apply(timedelta_breaks, min_lenght=3600)
        .values
    )
    continuous_periods_ends = continuous_periods_starts - 1

    continuous_periods_ends = np.concatenate((continuous_periods_ends[1:], [-1]))

    for start, end in zip(continuous_periods_starts, continuous_periods_ends):
        start_time = times_where_oven_is_off[start]
        end_time = times_where_oven_is_off[end]

        duration = (end_time - start_time).total_seconds()
        if duration >= 60 * 45:  # More than fourty five minutes means a real refill
            result.append(end_time)

    return result


def describe_cluster(cluster_df, features, weight_column, oven_refills):
    """ Create the statistics for a cluster. Datapoints that are part of a breakdown
    period are excluded.

    Parameters
    ----------
    cluster_df : DataFrame
        A dataframe that contains all points of the cluster you want to describe.
    features : list of source features
        All source feature for which the statistics should be generated
    weight_column : string
        Name of the column to use for weighting data points, typically
        `datapoint_duration` (``ProcessingFeatures.DATAPOINT_DURATION``)
    oven_refills : list of timestamp
        End of the oven refill periods

    Returns
    -------
    Series
        A Series of the following statistics

        For each parameter in `features`:

            1. mean
            2. std
            3. std% (std in percent of mean)
            4. avg_dev (average deviation of mean)
            5. min
            6. 25% (lower quartile)
            7. median
            8. 75% (upper quartile)
            9. max

        Once for the cluster:

            10. Density/count (number of data points in the cluster)
            11. Duration/in_hours (total duration of cluster)
            12. Duration/longest (duration of longest fragment)
            13. Duration/num_splits (number of fragments)
            14. Refill/index (index of oven refill that came directly before 
                the beginning of the longest fragment)
            15. Refill/delta_in_hours (delta from the end of the closest oven refill)
            16. num_breakdowns/per_hour (number of breakdowns per hour)
    """

    values = ["mean", "std", "std%", "avg_dev", "min", "25%", "median", "75%", "max"]
    index = pd.MultiIndex.from_tuples(
        [(p, v) for p in features for v in values]
        + [
            ("DENSITY", "count"),
            ("DURATION", "in_hours"),
            ("DURATION", "longest_in_hours"),
            ("DURATION", "num_splits"),
            ("REFILL", "index"),
            ("REFILL", "delta_in_hours"),
            ("num_breakdowns", "per_hour"),
        ]
    )

    data = cluster_df.loc[
        (cluster_df[ProcessingFeatures.HT_VOLTAGE_BREAKDOWN] == 0), features
    ].values  # TODO maybe only include non breakdown here???
    weights = cluster_df.loc[
        (cluster_df[ProcessingFeatures.HT_VOLTAGE_BREAKDOWN] == 0), weight_column
    ].values
    if data.size == 0:
        return None

    stats = DescrStatsW(data, weights, ddof=1)

    mean = np.array(stats.mean)  # np.mean(data, axis=0)
    std = np.array(stats.std)  # np.std(data, axis=0)
    quantiles = stats.quantile([0, 0.25, 0.5, 0.75, 1], return_pandas=False)
    # np.quantile(data, [0, 0.25, 0.5, 0.75, 1], axis=0)
    avg_dev = np.dot(weights, np.absolute(data - mean)) / np.sum(weights)

    count = len(data)

    duration_in_seconds = cluster_df[ProcessingFeatures.DATAPOINT_DURATION].sum()
    duration_in_hours = duration_in_seconds / 3600

    (
        duration_longest_start,
        duration_longest,
        duration_num_splits,
    ) = get_cluster_duration(cluster_df, weight_column)
    duration_longest /= 3600

    closest_refill = None
    for i, refill in reversed(list(enumerate(oven_refills))):
        if duration_longest_start > refill:
            closest_refill = i
            break

    refill_delta = -1
    if not closest_refill is None:
        refill_delta = (
            pd.Timestamp(duration_longest_start) - oven_refills[closest_refill]
        ).total_seconds() / 3600

    description = [
        [
            mean[i],
            std[i],
            np.abs(std[i] / mean[i]) * 100,
            avg_dev[i],
            quantiles[0][i],
            quantiles[1][i],
            quantiles[2][i],
            quantiles[3][i],
            quantiles[4][i],
        ]
        for i in range(len(features))
    ]
    description = [item for sublist in description for item in sublist]
    description.append(count)
    description.append(duration_in_hours)
    description.append(duration_longest)
    description.append(duration_num_splits)

    description.append(closest_refill)
    description.append(refill_delta)

    description.append(
        cluster_df.loc[
            cluster_df[ProcessingFeatures.HT_SPARKS_COUNTER] > 0,
            ProcessingFeatures.HT_SPARKS_COUNTER,
        ].nunique()
        / duration_in_hours
    )

    return pd.Series(description, index=index)


def timedelta_breaks(timedelta, min_lenght):
    if not pd.isnull(timedelta):
        return timedelta.total_seconds() > min_lenght
    else:
        return True


def get_cluster_duration(cluster_df, weight_column):
    """ Computes the duration of the longest fragment of a cluster.
    Between two fragments there has to be a break of at least one hour,
    otherwise they are merged together.

    cluster_df : DataFrame
        A dataframe that contains all points of the cluster.
    weight_column : string
        Name of the column to use for weighting data points, typically
        `datapoint_duration` (``ProcessingFeatures.DATAPOINT_DURATION``)

    Returns
    -------
    duration_longest_start : timestamp
        Starting time of the longest fragment
    duration_longest : double
        Duration of the longest fragment in seconds
    duration_num_splits : int
        Total number of fragments
    """

    continuous_periods_starts = np.flatnonzero(
        (cluster_df.index.to_series(keep_tz=True).diff(1))
        .apply(timedelta_breaks, min_lenght=3600)
        .values
    )

    duration_longest = 0
    duration_longest_start = None
    duration_num_splits = len(continuous_periods_starts)
    continuous_periods_starts = np.append(
        continuous_periods_starts, cluster_df.index.size
    )

    for i in range(1, len(continuous_periods_starts)):
        time_end = cluster_df.index[continuous_periods_starts[i] - 1]
        time_start = cluster_df.index[continuous_periods_starts[i - 1]]
        duration = cluster_df.loc[
            (cluster_df.index >= time_start) & (cluster_df.index <= time_end),
            weight_column,
        ].sum()

        if duration > duration_longest:
            duration_longest = duration
            duration_longest_start = time_start

    return duration_longest_start, duration_longest, duration_num_splits


def np_shift(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result


def get_wanted_statistics(features, statistics):
    """ Select the wanted statistics from the whole description

    Parameters
    ----------
    features : list of source features
        All source feature which you are interested in
    statistics : list of string
        The statistics you want to select
    
    Returns
    -------
    list of tuple
        Tuples in the form (feature, statistic) for all statistics and parameters
        passed as arguments
    """

    result = [[(param, stat) for stat in statistics] for param in features]
    result = [item for sublist in result for item in sublist]
    return result


def calculate_metrics(df, features):
    X = df.loc[
        (df[ProcessingFeatures.HT_VOLTAGE_BREAKDOWN] == 0)
        & (df[ProcessingFeatures.CLUSTER] >= 0),
        features,
    ].values
    scaler = preprocessing.RobustScaler((10, 90)).fit(X)
    X = scaler.transform(X)
    labels = df.loc[
        (df[ProcessingFeatures.HT_VOLTAGE_BREAKDOWN] == 0)
        & (df[ProcessingFeatures.CLUSTER] >= 0),
        ProcessingFeatures.CLUSTER,
    ].values

    dbi = cluster_metrics.DBI(X, labels)
    silhouette = cluster_metrics.silhouette(X, labels, 70000)

    return {"DBI": dbi, "silhouette": silhouette}


def parse_args():
    parser = argparse.ArgumentParser(description="Describe clusters")
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
        "-b",
        "--count_breakdowns_per_cluster",
        default="n",
        type=str,
        help="Count how many breakdowns occur per cluster? [y/n]",
    )
    parser.add_argument(
        "-v",
        "--num_clusters_to_visualize",
        default=20,
        type=int,
        help="How many clusters shall be displayed",
    )
    parser.add_argument(
        "-f",
        "--print_to_file",
        default="n",
        type=str,
        help="Print the results to a file? (y/n)",
    )
    parser.add_argument(
        "-m",
        "--display_metrics",
        default="n",
        type=str,
        help="Print clustering metrics? (y/n)",
    )

    args = parser.parse_args()

    return {
        "source_stability": args.source_stability,
        "year": args.year,
        "count_breakdowns_per_cluster": True
        if args.count_breakdowns_per_cluster == "y"
        else False,
        "num_clusters_to_visualize": args.num_clusters_to_visualize,
        "print_to_file": True if args.print_to_file == "y" else False,
        "display_metrics": True if args.display_metrics == "y" else False,
    }


if __name__ == "__main__":
    args = parse_args()
    year = args["year"]
    source_stability = args["source_stability"]
    count_breakdowns_per_cluster = args["count_breakdowns_per_cluster"]
    num_clusters_to_visualize = args["num_clusters_to_visualize"]
    print_to_file = args["print_to_file"]
    display_metrics = args["display_metrics"]

    main(
        year,
        source_stability,
        count_breakdowns_per_cluster,
        num_clusters_to_visualize,
        print_to_file,
        display_metrics,
    )
