""" After the clustering was performed, all voltage breakdown points
are still unclustered. We chose to assign all these points to the cluster that
happened previously if they contained at least one spark. Suppose for example 
that from minute 10-20 (of an arbitrary day/hour) the source was running stable in 
cluster 1, from minute 20-30 unstable in cluster 3 and minute 30-31 was marked as a
voltage breakdown with two sparks happening. Then we would assign this minute to
cluster 3 with an unstable source. If during minutes 20-30 the source would have been stable, 
we would assign minute 30-31 to cluster 3 of a stable source.

This is purely a matter of definition which we chose because we wanted to see how various settings
of the source lead to a different number of breakdowns.

How to use it
-------------
The program needs two command line arguments: The input file and the output file.
"""

import numpy as np
import pandas as pd

import sys, os

import argparse

sys.path.insert(1, os.path.abspath("../ionsrcopt"))
import load_data as ld
from source_features import SourceFeatures
from processing_features import ProcessingFeatures


def main(input_file, output_file):
    folder = "../Data_Clustered/"
    input_file = f"{folder}{input_file}.csv"
    output_file = f"{folder}{output_file}.csv"

    df = ld.read_data_from_csv(input_file, None, None)

    df = fill_columns(df)
    df = reset_breakdown_clusters(df)
    df = assign_clusters(df)

    mask = (df.shift(1) == df).fillna(value=True).astype(bool)
    df = df.where(~mask, np.nan)
    # df = df.round(4)

    df.to_csv(output_file)


def fill_columns(df):
    df[ProcessingFeatures.SOURCE_STABILITY] = df[
        ProcessingFeatures.SOURCE_STABILITY
    ].fillna(method="ffill")
    df[ProcessingFeatures.HT_VOLTAGE_BREAKDOWN] = df[
        ProcessingFeatures.HT_VOLTAGE_BREAKDOWN
    ].fillna(method="ffill")
    df[ProcessingFeatures.CLUSTER] = df[ProcessingFeatures.CLUSTER].fillna(
        method="ffill"
    )
    return df


def assign_clusters(df):
    df[ProcessingFeatures.CLUSTER] = df[ProcessingFeatures.CLUSTER].fillna(
        method="ffill"
    )
    df[ProcessingFeatures.SOURCE_STABILITY] = df[
        ProcessingFeatures.SOURCE_STABILITY
    ].fillna(method="ffill")
    return df


def reset_breakdown_clusters(df):
    voltage_breakdown_beginnings = df.index[
        (df[ProcessingFeatures.HT_VOLTAGE_BREAKDOWN] > 0)
        & (df.shift(1)[ProcessingFeatures.HT_VOLTAGE_BREAKDOWN] == 0)
    ]
    voltage_breakdown_ends = df.index[
        (df[ProcessingFeatures.HT_VOLTAGE_BREAKDOWN] > 0)
        & (df.shift(-1)[ProcessingFeatures.HT_VOLTAGE_BREAKDOWN] == 0)
    ]

    if voltage_breakdown_ends[0] < voltage_breakdown_beginnings[0]:
        voltage_breakdown_beginnings.insert(0, df.index.values[0])

    if voltage_breakdown_ends[-1] < voltage_breakdown_beginnings[-1]:
        voltage_breakdown_ends.append(df.index.values[-1])

    for start, end in zip(voltage_breakdown_beginnings, voltage_breakdown_ends):
        num_of_sparks = (
            df.loc[
                (df.index >= start) & (df.index <= end),
                ProcessingFeatures.HT_SPARKS_COUNTER,
            ]
            > 0
        ).sum()
        if num_of_sparks > 0:
            df.loc[
                (df.index >= start) & (df.index <= end), ProcessingFeatures.CLUSTER
            ] = np.nan
            df.loc[
                (df.index >= start) & (df.index <= end),
                ProcessingFeatures.SOURCE_STABILITY,
            ] = np.nan

    return df


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    parser.add_argument("output_file")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.input_file, args.output_file)
