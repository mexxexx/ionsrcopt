import numpy as np
import pandas as pd

pd.plotting.register_matplotlib_converters()

import sys, os
import matplotlib.pyplot as plt

sys.path.insert(1, os.path.abspath("../ionsrcopt"))
from source_features import SourceFeatures
from processing_features import ProcessingFeatures
from voltage_breakdowns import count_sparks, detect_breakdowns
from source_stability import calculate_source_running
import load_data as ld


def main():
    input_file = "../Data_Raw/May2018.csv"
    columns = [
        SourceFeatures.TIMESTAMP,
        SourceFeatures.BCT05_CURRENT,
        SourceFeatures.SOURCEHTAQNV,
        SourceFeatures.SOURCEHTAQNI,
        SourceFeatures.SPARK_COUNTER,
    ]

    df = ld.read_data_from_csv(input_file, columns, None)
    df = ld.fill_columns(df, None, fill_nan_with_zeros=True)
    df = ld.convert_column_types(df)

    source_running = calculate_source_running(df[SourceFeatures.BCT05_CURRENT])
    window_size = 20
    threshold = 0.25
    breakdowns = detect_breakdowns(
        df, SourceFeatures.SOURCEHTAQNI, window_size, threshold
    ).astype("int64")

    threshold = 1000
    df[ProcessingFeatures.HT_SPARKS_COUNTER] = count_sparks(
        df[SourceFeatures.SOURCEHTAQNV], breakdowns, threshold
    )
    df.loc[
        df[ProcessingFeatures.HT_SPARKS_COUNTER] == 0,
        ProcessingFeatures.HT_SPARKS_COUNTER,
    ] = np.nan
    df.loc[
        df[ProcessingFeatures.HT_SPARKS_COUNTER] > 0,
        ProcessingFeatures.HT_SPARKS_COUNTER,
    ] = np.arange(1, (df[ProcessingFeatures.HT_SPARKS_COUNTER] > 0).sum() + 1)
    df[ProcessingFeatures.HT_SPARKS_COUNTER] = df[
        ProcessingFeatures.HT_SPARKS_COUNTER
    ].ffill()

    df.loc[
        df[SourceFeatures.SPARK_COUNTER] == df[SourceFeatures.SPARK_COUNTER].shift(1),
        SourceFeatures.SPARK_COUNTER,
    ] = np.nan
    df.loc[df[SourceFeatures.SPARK_COUNTER] == 0, SourceFeatures.SPARK_COUNTER] = np.nan
    df.loc[
        df[SourceFeatures.SPARK_COUNTER] > 0, SourceFeatures.SPARK_COUNTER
    ] = np.arange(1, (df[SourceFeatures.SPARK_COUNTER] > 0).sum() + 1)
    df[SourceFeatures.SPARK_COUNTER] = df[SourceFeatures.SPARK_COUNTER].ffill()

    fig, ax = plt.subplots(2, 1, sharex=True)

    ax_htv = ax[0].twinx()
    ax_hti = ax[0].twinx()
    ax_hti.spines["right"].set_position(("axes", 1.04))
    # make_patch_spines_invisible(par2)
    ax_hti.spines["right"].set_visible(True)

    # ax[0].plot(df[ProcessingFeatures.HT_SPARKS_COUNTER], color='red')
    ax[0].plot(df[SourceFeatures.BCT05_CURRENT], color="red")
    ax_htv.plot(df[SourceFeatures.SOURCEHTAQNV])
    ax_hti.plot(df[SourceFeatures.SOURCEHTAQNI], color="orange")

    sparks_real = df[SourceFeatures.SPARK_COUNTER]
    ax12 = ax[1].twinx()
    ax[1].plot(df[ProcessingFeatures.HT_SPARKS_COUNTER], color="red")
    ax[1].plot(sparks_real, color="orange")

    plt.show()


if __name__ == "__main__":
    main()
