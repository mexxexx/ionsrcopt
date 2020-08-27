""" For our analysis we distinguished the source operation into two
modes: stable operation and unstable operation. 

We make this distinction based on rolling windows over the BCT25 currents
mean and standard deviation/variance. If they exceed certain thresholds, we
consider the current as unstable.
"""

import pandas as pd
import numpy as np


def calculate_source_running(source_current):
    """ Determines whether the source was running, i.e. if the `source_current` 
    (typically BCT05 current) was above 0.004mA.

    Parameters:
        source_current (Series): A series of values of the current you want to use (typically BCT05)

    Returns:
        Series: A series with `1` at the indices where the current was above 0.004mA, and a `0` elsewhere.
    """

    is_zero_threshold = 0.004
    result = np.zeros(source_current.size, dtype=bool)
    result[source_current > is_zero_threshold] = 1
    return result


def stability_mean_variance_classification(
    df,
    value_column,
    weight_column,
    sliding_window_size_mean=500,
    sliding_window_size_std=1000,
    minimum_mean=0.025,
    maximum_variance=0.00005,
):
    """ Classifies all points in the data frame into the categories source stable/unstable, based on a rolling window and a minimum mean and maximum variance in this window.

    Parameters:
        df (DataFrame): The data input loaded as a DataFrame
        current_column (string): name of the column that contains the beam current we are interested in, typically BCT25
        sliding_window_size (int): size of the sliding window, by default 5000 (100 Minutes of data every 1.2 seconds)
        minimum_mean (double): minimal intensity of the beam in the sliding window for it to be considered stable
        maximum_variance (double): maximum variance of intensity of the beam in the sliding window for it to be considered stable

    Returns:
        Series: A series that for every data point indicates if the source was running stable or not (1 is stable, 0 is unstable)
    """

    df["wvalue"] = df[value_column] * df[weight_column]

    mean_weight_sum = (
        df[["wvalue", weight_column]]
        .rolling("{}s".format(sliding_window_size_mean), closed="left")
        .sum()
    )
    wmean = mean_weight_sum["wvalue"] / mean_weight_sum[weight_column]
    wmean.name = "wmean"

    df["wdeviation"] = df[value_column] - wmean
    df["wdeviation"] = df["wdeviation"] ** 2
    df["wdeviation"] *= df[weight_column]
    var_weight_sum = (
        df[["wdeviation", weight_column]]
        .rolling("{}s".format(sliding_window_size_mean), closed="left")
        .sum()
    )
    wvar = var_weight_sum["wdeviation"] / (var_weight_sum[weight_column] - 1)
    wvar.name = "wvar"

    df.drop(["wvalue", "wdeviation"], axis=1, inplace=True)

    stats = pd.concat([wmean, wvar], axis=1)
    stats["result"] = 0
    stats.loc[
        (stats["wmean"] > minimum_mean) & (stats["wvar"] < maximum_variance), "result"
    ] = 1

    return stats["result"]
