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

    #add_column_to_preprocessing(year, start_month, end_month, new_column_name, input_folder, output_folder)
    add_column_to_clustered(year, start_month, end_month, new_column_name, input_folder, output_folder)

def add_column_to_preprocessing(year, start_month, end_month, new_column_name, input_folder, output_folder):
    suffix = "_htv"

    for m in months[months.index(start_month) : months.index(end_month) + 1]:
        print(f"Adding column for month {m}{year}")
        new_column = load_new_column(
            year, m, m, new_column_name, input_folder
        )
        file_to_add_to = f"{output_folder}{m}{year}.csv"
        df = load_existing_data(file_to_add_to)
        df = add_column(df, new_column, new_column_name)

        output_file = f"{output_folder}{m}{year}{suffix}.csv"
        df = df.round(4)
        df.to_csv(output_file)
    

def add_column_to_clustered(year, start_month, end_month, new_column_name, input_folder, output_folder):
    input_file = f"{output_folder}/JanNov2016_htv.csv"
    output_file = f"{output_folder}/JanNov2016_sparks.csv"

    new_column = load_new_column(
        str(year) + "_htv", start_month, end_month, new_column_name, input_folder
    )
    df = load_existing_data(input_file)
    df = add_column(df, new_column, new_column_name)

    df.to_csv(output_file)


def load_new_column(year, start_month, end_month, new_column_name, data_folder):
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
    df = pd.read_csv(filename, index_col=SourceFeatures.TIMESTAMP)
    df.index = pd.to_datetime(df.index)

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    elif df.index.tz != "UTC":
        df.index = df.index.tz_convert("UTC")

    return df


def add_column(df, new_column, new_column_name):
    new_column = new_column.reindex(index=df.index, method="ffill")
    new_column[new_column.shift(1) == new_column] = np.nan

    df[new_column_name] = new_column
    return df


if __name__ == "__main__":
    main()
