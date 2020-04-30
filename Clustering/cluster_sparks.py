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

    mask = (df.shift(1)==df).fillna(value=True).astype(bool)
    df = df.where(~mask, np.nan)
    #df = df.round(4)

    df.to_csv(output_file)

def fill_columns(df):
    df[ProcessingFeatures.SOURCE_STABILITY] = df[ProcessingFeatures.SOURCE_STABILITY].fillna(method="ffill")
    df[ProcessingFeatures.HT_VOLTAGE_BREAKDOWN] = df[ProcessingFeatures.HT_VOLTAGE_BREAKDOWN].fillna(method="ffill")
    df[ProcessingFeatures.CLUSTER] = df[ProcessingFeatures.CLUSTER].fillna(method="ffill")
    return df

def assign_clusters(df):
    df[ProcessingFeatures.CLUSTER] = df[ProcessingFeatures.CLUSTER].fillna(method="ffill")
    df[ProcessingFeatures.SOURCE_STABILITY] = df[ProcessingFeatures.SOURCE_STABILITY].fillna(method="ffill")
    return df


def reset_breakdown_clusters(df):
    df.loc[df[ProcessingFeatures.HT_VOLTAGE_BREAKDOWN]>0, ProcessingFeatures.CLUSTER] = np.nan
    df.loc[df[ProcessingFeatures.HT_VOLTAGE_BREAKDOWN]>0, ProcessingFeatures.SOURCE_STABILITY] = np.nan
    return df


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    parser.add_argument("output_file")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args.input_file, args.output_file)