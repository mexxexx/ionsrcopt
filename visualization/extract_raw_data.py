import pandas as pd
import argparse
import sys, os

sys.path.insert(1, os.path.abspath("../ionsrcopt"))
import load_data as ld
from source_features import SourceFeatures
from processing_features import ProcessingFeatures


def main():
    ######################
    ###### SETTINGS ######
    ######################

    clustered_data_folder = "../Data_Clustered/"  # Base folder of clustered data
    filename = "MayDec2015_htv.csv"  # The file to load
    feature = SourceFeatures.SOURCEHTAQNV

    args = parse_args()
    source_stability = args["source_stability"]

    output_filename = filename + feature + str(source_stability) + ".csv"

    ######################
    ######## CODE ########
    ######################

    # Load file into a data frame
    path = clustered_data_folder + filename
    df = ld.read_data_from_csv(path, None, None)
    df = ld.fill_columns(df, None, fill_nan_with_zeros=True)
    df = ld.convert_column_types(df)

    selector_stability = df[ProcessingFeatures.SOURCE_STABILITY] == source_stability
    selector_running = df[ProcessingFeatures.SOURCE_RUNNING] == 1

    df_new = df.loc[selector_stability & selector_running, feature].copy()
    df_new.to_csv(output_filename, header=True)


def parse_args():
    parser = argparse.ArgumentParser(description="View time development of clusters")
    parser.add_argument(
        "-s",
        "--source_stability",
        default=1,
        type=int,
        help="1 if you want to look at the stable source, 0 else",
    )

    args = parser.parse_args()

    return {"source_stability": args.source_stability}


if __name__ == "__main__":
    main()
