""" Sometimes it may be necessary to add a column to your preprocessing
or clustering files, for example if you downloaded a new parameter you are
interested in. Then you maybe do not want to run the scripts again. In this case
you can use the ``add_column.py`` script. It supports two different modes: 
Adding to a preprocessed file and adding to a clustered file. You have to
uncomment the function call in `main` that you are interested in.

Adding to a preprocessed file
-----------------------------

In the case of preprocessed files, every month is still stored in a separate .csv
file. You select the year and the start and end months, and the new column will be added
to every month you selected. The `new_column` parameter controls the name of the column
that will be added. The information is taken from the files in the `input_folder` and
appended to the corresponding files in the `output_folder` (in this case, typically
`input_folder="Data_Raw` and `output_folder=Data_Preprocessed`). In the function
`add_column_to_preprocessing` you can specify a suffix that will be appended to the input
file name. `<month><year>.csv` becomes `<month><year><suffix>.csv`

Adding to a clustered file
--------------------------

For one year, all results of the cluster analysis are written into the same file. Inside
the function you have to specify an input file (this will be the file you want to append to)
and an output file (the appended version).
"""

import pandas as pd
import numpy as np

import sys, os

sys.path.insert(1, os.path.abspath("../ionsrcopt"))
import load_data as ld
from source_features import SourceFeatures
from processing_features import ProcessingFeatures

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


def main():
    year = 2016
    start_month = "Jan"
    end_month = "Nov"
    input_folder = "../Data_Preprocessed/"
    output_folder = "../Data_Clustered/"
    new_column_name = ProcessingFeatures.HT_SPARKS_COUNTER

    # add_column_to_preprocessing(year, start_month, end_month, new_column_name, input_folder, output_folder)
    add_column_to_clustered(
        year, start_month, end_month, new_column_name, input_folder, output_folder
    )


def add_column_to_preprocessing(
    year, start_month, end_month, new_column_name, input_folder, output_folder
):
    """ Add a column to a preprocessing file
    """
    suffix = "_htv"

    for m in months[months.index(start_month) : months.index(end_month) + 1]:
        print(f"Adding column for month {m}{year}")
        new_column = load_new_column(year, m, m, new_column_name, input_folder)
        file_to_add_to = f"{output_folder}{m}{year}.csv"
        df = load_existing_data(file_to_add_to)
        df = add_column(df, new_column, new_column_name)

        output_file = f"{output_folder}{m}{year}{suffix}.csv"
        df = df.round(4)
        df.to_csv(output_file)


def add_column_to_clustered(
    year, start_month, end_month, new_column_name, input_folder, output_folder
):
    """ Add a column to a clustering file
    """
    input_file = f"{output_folder}/JanNov2016_htv.csv"
    output_file = f"{output_folder}/JanNov2016_sparks.csv"

    new_column = load_new_column(
        str(year) + "_htv", start_month, end_month, new_column_name, input_folder
    )
    df = load_existing_data(input_file)
    df = add_column(df, new_column, new_column_name)

    df.to_csv(output_file)


def load_new_column(year, start_month, end_month, new_column_name, data_folder):
    """ For a given year loads the column `new_column_name` for all months between `start_month` and 
    `end_month` (inclusive) as one long pandas series.
    """
    result = pd.Series()
    result.index = pd.to_datetime(result.index).tz_localize("UTC")
    for m in months[months.index(start_month) : months.index(end_month) + 1]:
        filename = data_folder + "{}{}.csv".format(m, year)

        df = pd.read_csv(
            filename,
            index_col=SourceFeatures.TIMESTAMP,
            usecols=[SourceFeatures.TIMESTAMP, new_column_name],
        )
        df.index = pd.to_datetime(df.index)

        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        elif df.index.tz != "UTC":
            df.index = df.index.tz_convert("UTC")

        result = result.append(df[new_column_name])

    result = result.dropna()
    return result


def load_existing_data(filename):
    """ Loads a complete file and correctly localizes it to UTC.
    """
    df = pd.read_csv(filename, index_col=SourceFeatures.TIMESTAMP)
    df.index = pd.to_datetime(df.index)

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    elif df.index.tz != "UTC":
        df.index = df.index.tz_convert("UTC")

    return df


def add_column(df, new_column, new_column_name):
    """ Adds the new column to the `df` DataFrame keeping the index of df as is
    and forward filling the data of `new_column`. Consecutive repeated values are
    replaced with `np.nan` and only the first occurrence is kept.
    """
    new_column = new_column.reindex(index=df.index, method="ffill")
    new_column[new_column.shift(1) == new_column] = np.nan

    df[new_column_name] = new_column
    return df


if __name__ == "__main__":
    main()
