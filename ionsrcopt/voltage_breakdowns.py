import pandas as pd
import numpy as np

def classify_using_std_threshold(values, threshold):
    """ Classify values nased on the standard deviation exceding a certain threshold """

    std = np.std(values)
    return int(std >= threshold)

def detect_breakdowns(df, column='IP.NSRCGEN:SOURCEHTAQNI', window_size=40, threshold=0.5):
    """ Detection of high voltage breakdown based on standard deviation exceding a certain threshold that has to be determined by experiments.
    
    Parameters:
        df (DataFrame): The frame containing the data
        column (string): High voltage current, typically this should be 'IP.NSRCGEN:SOURCEHTAQNI' 
        window_size (int): Size of the rolling window. Once a breakdown is detected, every value in this window will be set to 1.
        threshold (double): Threshold for the standard deviation.
    
    Returns: 
        Series: Series that has ones, wherever a breakdown is found and is zero otherwise
    """

    if not column in df:
        raise ValueError("Error: The column cannot be found in the dataframe.")

    min_index = df.iloc[0].name

    result = np.zeros(len(df.index))
    values = df[column].values

    for i in range(len(values) - window_size):
        is_breakdown = classify_using_std_threshold(values[i:i+window_size], threshold)
        if is_breakdown:
            result[(df.index >= min_index + i) & (df.index < min_index + i + window_size)] = 1

    return pd.Series(result, name='is_breakdown')