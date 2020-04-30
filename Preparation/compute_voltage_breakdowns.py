import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.plotting.register_matplotlib_converters()

import argparse
import sys, os

sys.path.insert(1, os.path.abspath("../ionsrcopt"))
import load_data as ld
from source_features import SourceFeatures
from processing_features import ProcessingFeatures
from voltage_breakdowns import detect_sparks, detect_breakdowns

months = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]

def main(plot):
    year = 2018
    start_month = 'Jan'
    end_month = 'Nov'

    data_path = "../Data_Preprocessed"

    for i, m in enumerate(months[months.index(start_month) : months.index(end_month) + 1]):
        file_path = f"{data_path}/{m}{year}_htv.csv"
        print(f"HT sparks for {file_path}")

        previous_month_file = None
        if i > 0:
            m_prev = months[months.index(m) - 1]
            previous_month_file = f"{data_path}/{m_prev}{year}_htv.csv"

        df = ld.read_data_from_csv(file_path, None, None)
        df = ld.fill_columns(df, previous_month_file, fill_nan_with_zeros=True)
        #df = ld.convert_column_types(df)

        # First we mark all time periods where the variance of the HT current is 
        # above a certain threshold to exclude all these windows from our analysis
        window_size = 40
        threshold_breakdowns = 0.25
        breakdowns = detect_breakdowns(
            df, SourceFeatures.SOURCEHTAQNI, window_size, threshold_breakdowns
        ).astype("int64")

        # Then we search for all downward spikes in the HT voltage that fall below 1000V
        # and have a prominence of 500V, i.e. are significant compared to the background.
        # The are the actual sparks and can be compared with IP.NSRCGEN:SPARKS for 2018
        threshold_sparks = 1000
        sparks = detect_sparks(
            df[SourceFeatures.SOURCEHTAQNV], breakdowns, threshold_sparks
        )

        df[ProcessingFeatures.HT_VOLTAGE_BREAKDOWN] = breakdowns
        df[ProcessingFeatures.HT_VOLTAGE_BREAKDOWN] = df[ProcessingFeatures.HT_VOLTAGE_BREAKDOWN].astype('Int32')

        df[ProcessingFeatures.HT_SPARKS_COUNTER] = sparks
        df[ProcessingFeatures.HT_SPARKS_COUNTER] = df[ProcessingFeatures.HT_SPARKS_COUNTER].astype('Int32')
        #df.loc[df[ProcessingFeatures.HT_SPARKS_COUNTER] == 0, ProcessingFeatures.HT_SPARKS_COUNTER] = np.nan

        if plot:
            plot_breakdowns(df)

        mask = (df.shift(1)==df).fillna(value=True).astype(bool)
        df = df.where(~mask, np.nan)

        df.to_csv(file_path)
        print("Saved HT spark search to {}.".format(file_path))

def plot_breakdowns(df):
    fig, ax = plt.subplots()
    ax.plot(df[SourceFeatures.SOURCEHTAQNI])
    ax.vlines(df[~np.isnan(df[ProcessingFeatures.HT_SPARKS_COUNTER])].index, 0, 5)

    ax2 = ax.twinx()
    ax2.plot(df[SourceFeatures.SOURCEHTAQNV], color='orange')

    plt.show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--plot", action="store_true", help="Add this flag if you want to see a plot of the results.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    main(args.plot)
