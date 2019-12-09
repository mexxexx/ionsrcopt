import pandas as pd
import numpy as np

def classify_using_var_threshold(values, threshold):
    """ Classify values based on the variance exceeding a certain threshold """

    var = np.var(values)
    return int(var >= threshold)

def detect_breakdowns(df, ht_current_column, window_size=40, threshold=0.5):
    """ Detection of high voltage breakdown based on standard deviation exceding a certain threshold that has to be determined by experiments.
    
    Parameters:
        df (DataFrame): The frame containing the data
        column (string): High voltage current, typically this should be 'IP.NSRCGEN:SOURCEHTAQNI' 
        window_size (int): Size of the rolling window. Once a breakdown is detected, every value in this window will be set to 1.
        threshold (double): Threshold for the standard deviation.
    
    Returns: 
        np.array: array that has ones, wherever a breakdown is found and is zero otherwise
    """

    if not ht_current_column in df:
        raise ValueError("Error: The column cannot be found in the dataframe.")

    result = np.zeros(len(df.index))
    values = df[ht_current_column].values
    times = (df.index.astype('int64') * 1E-9).values
    
    current_breakdown = 0
    for i in range(len(values) - window_size):
        is_breakdown = classify_using_var_threshold(values[i:i+window_size], threshold)
        if is_breakdown:
            if not result[i]:
                current_breakdown = times[i]
                
            result[i:(i + window_size)] = current_breakdown

    return result