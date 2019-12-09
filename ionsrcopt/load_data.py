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
                df = pd.read_csv(filename).fillna(method='ffill')
            else:
                df = pd.read_csv(filename, usecols=cols_to_read).fillna(method='ffill')
        except:
            print("File {} does not exist or is not a csv file". format(filename))
            exit()

        if not (SourceFeatures.Timestamp in df.columns or 'Timestamp (UTC_TIME)' in df.columns):
            print("No timestamp column was found. It must be named either {} or \'Timestamp (UTC_TIME)\'.".format(SourceFeatures.Timestamp))
            exit()

        df = df.rename(columns={'Timestamp (UTC_TIME)' : SourceFeatures.Timestamp})
        df[SourceFeatures.Timestamp] = pd.to_datetime(df[SourceFeatures.Timestamp]) 
        df = df.set_index(SourceFeatures.Timestamp)
        
        if not rows_to_read is None:
            df = df.iloc[rows_to_read].copy()

        dfs.append(df)        

    result = pd.concat(dfs, axis=0, sort=False)
    return result.sort_index() 

def convert_column(df, column, type):
    """ Converts the dtype of a column

    Parameters:
        df (DataFrame): The DataFrame containing the column
        column (string): The column name
        type (string): dtype the column should be converted to

    Returns:
        DataFrame: The altered DataFrame or the old one, if it did not contain the specified column
    """

    if column in df.columns:
        print("Converting column \'{}\' to \'{}\'".format(column, type))
        return df.astype({column:type})
    else:
        #print("Column \'{}\' does not exist".format(column))
        return df

def convert_column_types(df):
    """ Convert all columns of a Dataframe of measurements to single precision values.

    Parameters:
        df (DataFrame): DataFrame to be altered

    Returns:
        DataFrame
    """

    print("Started type conversion of columns...")
    df = convert_column(df, SourceFeatures.BIASDISCAQNV, 'float32')
    df = convert_column(df, SourceFeatures.GASSASAQN, 'float32')
    df = convert_column(df, SourceFeatures.GASAQN, 'float32')
    df = convert_column(df, SourceFeatures.SOLINJ_CURRENT, 'float32')
    df = convert_column(df, SourceFeatures.SOLCEN_CURRENT, 'float32')
    df = convert_column(df, SourceFeatures.SOLEXT_CURRENT, 'float32')
    df = convert_column(df, SourceFeatures.OVEN1AQNP, 'float32')
    df = convert_column(df, SourceFeatures.OVEN2AQNP, 'float32')
    df = convert_column(df, SourceFeatures.SAIREM2_FORWARDPOWER, 'float32')
    df = convert_column(df, SourceFeatures.SOURCEHTAQNI, 'float32')
    df = convert_column(df, SourceFeatures.BCT25_CURRENT, 'float32')
    df = convert_column(df, ProcessingFeatures.SOURCE_STABILITY, 'int32')
    df = convert_column(df, ProcessingFeatures.HT_VOLTAGE_BREAKDOWN, 'int32')
    df = convert_column(df, ProcessingFeatures.DATAPOINT_DURATION, 'float32')
    df = convert_column(df, ProcessingFeatures.CLUSTER, 'int32')
    return df