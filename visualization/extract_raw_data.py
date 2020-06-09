""" With the `extract_raw_data.py` script, you can extract a .csv
file that contains all observations of a given feature from a given
file for a specific source stability.

How to use it
-------------
In the code you need to make two selections. First you have to specify
the folder and filename of the file you want to extract the data from.
Then, you have to set `feature` to a columns of this input file.

The program will output a .csv file with the name 
`<filename><feature><source_stability>.csv`. This file will contain two
columns with the header names `TIMESTAMP` and `<feature>`. Each of the rows
representing one observation at the specific timestamp. 

Command line arguments
----------------------
As command line argument you can provide the source stability you are
interested in.

-s: Pass a 1 to see the clusters of the stable periods and a 0 for the 
unstable ones. (default 1)
"""

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
    parser = argparse.ArgumentParser(description="Extract raw data.")
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
