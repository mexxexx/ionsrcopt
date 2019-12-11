import pandas as pd
import numpy as np

from source_features import SourceFeatures
from processing_features import ProcessingFeatures

def read_data_from_csv(filenames, cols_to_read, rows_to_read):
    """ Read a csv file into a DataFrame

    Parameters:
        filenames (list string): Filenames. Concatenates all into one data frame
        cols_to_read (list of string): The column names to read, None if everything should be read
        rows_to_read (list of int): The rown numbers to read, None if everything should be read

    Returns:
        DataFrame
    """

    if isinstance(filenames, str):
        filenames = [filenames]

    dfs = []

    for filename in filenames:
        print("Loading data from csv file \'{}\'".format(filename))

        try:
            if cols_to_read is None:
                df = pd.read_csv(filename)
            else:
                df = pd.read_csv(filename, usecols=cols_to_read)
        except:
            print("File {} does not exist or is not a csv file". format(filename))
            exit()

        if not SourceFeatures.TIMESTAMP in df.columns:
            print("No timestamp column was found. It must be named {}.".format(SourceFeatures.TIMESTAMP))
            exit()

        df[SourceFeatures.TIMESTAMP] = pd.to_datetime(df[SourceFeatures.TIMESTAMP]) 
        df = df.set_index(SourceFeatures.TIMESTAMP)
        
        if not rows_to_read is None:
            df = df.iloc[rows_to_read].copy()

        dfs.append(df)        

    result = pd.concat(dfs, axis=0, sort=False)
    return result.sort_index()

def convert_column_types(df):
    """ Convert all columns of a Dataframe of measurements to single precision values.

    Parameters:
        df (DataFrame): DataFrame to be altered

    Returns:
        DataFrame
    """

    print("Converting column types...")
    conversions = {
        SourceFeatures.BIASDISCAQNV : 'float32',
        SourceFeatures.GASSASAQN : 'float32',
        SourceFeatures.GASAQN : 'float32',
        SourceFeatures.SOLINJ_CURRENT : 'float32',
        SourceFeatures.SOLEXT_CURRENT : 'float32',
        SourceFeatures.SOLCEN_CURRENT : 'float32',
        SourceFeatures.OVEN1AQNP : 'float32',
        SourceFeatures.OVEN2AQNP : 'float32',
        SourceFeatures.SAIREM2_FORWARDPOWER : 'float32',
        SourceFeatures.SOURCEHTAQNI : 'float32',
        SourceFeatures.BCT05_CURRENT : 'float32',
        SourceFeatures.BCT25_CURRENT : 'float32',
        ProcessingFeatures.SOURCE_STABILITY : 'int32',
        ProcessingFeatures.HT_VOLTAGE_BREAKDOWN : 'int32',
        ProcessingFeatures.DATAPOINT_DURATION : 'float32',
        ProcessingFeatures.CLUSTER : 'int32',
        ProcessingFeatures.SOURCE_RUNNING : 'bool'
    }

    conversions_to_apply = { c : conversions[c] for c in df.columns }
    return df.astype(conversions_to_apply)

def add_previous_data(df, previous_data, fill_nan_with_zeros):
    """ Given the data from the previous time interval, this method selects for each feature where past data exists the last row where it was non null and inserts these rows into the frame at the beginning

    Parameters:
        df (DataFrame): The data frame with the data from the current time interval
        previous_data (None or String or DataFrame): The data from the previous interval. If None, then this method does nothing. If it is a file, it loads the data from the file. If it is a data frame, the dataa is taken directly from there.

    Returns:
        Timestamp: This is the first timestamp of the original data frame. Everything before was added from previous data
        DataFrame: The altered frame. It has a few rows at the beginning that include the data from before
    """
    old_first_index = df.index[0]
    new_rows = []

    if not previous_data is None:
        if isinstance(previous_data, str):
            previous_data = read_data_from_csv(previous_data, [df.index.name] + list(df.columns), None)
            previous_data = convert_column_types(previous_data)

        if not isinstance(previous_data, pd.DataFrame):
            raise TypeError("previous_data hast to either be None, a filename or a DataFrame")

        for column in df.columns:
            if not column in previous_data.columns:
                continue

            last_index_with_data = previous_data[column].last_valid_index()
            if last_index_with_data:
                new_rows.append(previous_data.loc[last_index_with_data])

    new_rows.sort(key=lambda x: x.name)

    if fill_nan_with_zeros:
        new_row = pd.Series(data=np.zeros(len(df.columns)), index=df.columns)
        oldest_index = old_first_index if not new_rows else new_rows[0].name
        new_row.name = oldest_index - pd.Timedelta('1 day')
        new_rows.insert(0, new_row)

    df_previous = pd.DataFrame(new_rows).drop_duplicates()
    df_previous.index.name = df.index.name
    df = df_previous.append(df)
    return old_first_index, df

def fill_columns(df, previous_data, fill_nan_with_zeros=False):
    print("Forward filling missing values...")

    old_first_index, df = add_previous_data(df, previous_data, fill_nan_with_zeros)
    df = df.fillna(method='ffill')
    return df.loc[df.index >= old_first_index]
