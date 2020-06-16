""" Sometimes, the high voltage of the source can break down.
This disturbs the beam and should be avoided. Hence we wrote
this module to count how often it happens. Furthermore, as during
these breakdowns the HT current makes a spark, we want to exclude
data points that belong to a breakdown from the analysis, to not
induce noise into the results.

During a spark the following happens: First, the voltage breaks down,
from around 20000V during operation to <1000V. This can also be seen in
the HT current, that typically rapidly drops to zero A, shoots up to three A,
and then normalizes again. Shortly after this is registered by the system,
the extraction solenoid current is ramped down to around 850A.

This module provides two tools: 

1. The first one, `detect_breakdowns` finds
periods where the HT current variance exceeds a threshold in a short window.
Typically, the current has a low variance, and hence the sparks above can be
found reliably with this method. It marks the whole window as a breakdown,
so that all these data points can be ignored in the future analysis.

2. The second one, `detect_sparks`, detects where exactly the the voltage 
broke down. If two breakdowns happened shortly after each other, method 1
would count only one breakdown, but we are interested in the exact number.
This methods counts the local minima of the HT voltage that are below a
certain threshold.
"""

import pandas as pd
import numpy as np

from scipy import signal


def classify_using_var_threshold(values, threshold):
    """ Classify values based on the variance exceeding a certain threshold """

    var = np.var(values)
    return int(var >= threshold)


def detect_breakdowns(df, ht_current_column, window_size=40, threshold=0.5):
    """ Detection of high voltage breakdown based on standard deviation exceding a certain threshold that has to be determined by experiments.
    
    Parameters:
        df (DataFrame): The frame containing the data
        column (string): High voltage current, typically this should be 'IP.NSRCGEN:SOURCEHTAQNI' 
        window_size (int): Size of the rolling window. Once a breakdown is detected, every value in this window will be set to 1.
        threshold (double): Threshold for the standard deviation.
    
    Returns: 
        np.array: For each data point that lies inside of a breakdown window, this array contains the timestamp of the start of the window, 
        otherwise it is zero. So for each value greater that zero, all data points with the same value were in the same breakdown window.
    """

    if not ht_current_column in df:
        raise ValueError("Error: The column cannot be found in the dataframe.")

    result = np.zeros(len(df.index))
    values = df[ht_current_column].values
    times = (df.index.astype("int64") * 1e-9).values

    current_breakdown = 0
    for i in range(len(values) - window_size):
        is_breakdown = classify_using_var_threshold(
            values[i : i + window_size], threshold
        )
        if is_breakdown:
            if not result[i]:
                current_breakdown = times[i]

            result[i : (i + window_size)] = current_breakdown

    return result


def detect_sparks(ht_voltage, breakdowns, threshold=1000):
    """ Detect all sparks, i.e. the number of downward peaks of the HT voltage below a certain threshold.

    Parameters
    ----------
        ht_voltage (np.array): The HT voltage
        breakdowns (np.array): An array where the breakdown windows are marked (output of `detect_breakdowns`).    
                            Only peaks in these windows are counted as sparks.
        threshold (float): Maximum value of the HT current for a peak to be counted as breakdowns

    Returns
    -------
        np.array: At each point where a spark occurred its timestamp, otherwise zero.
    """

    ht_voltage = ht_voltage.copy()
    ht_voltage[breakdowns == 0] = threshold + 1

    result = np.zeros(len(ht_voltage.index), dtype="int64")
    values = ht_voltage.values
    times = (ht_voltage.index.astype("int64") * 1e-9).values

    peaks, _ = signal.find_peaks(-values, height=-threshold, prominence=threshold / 2)
    result[peaks] = times[peaks]

    return result
