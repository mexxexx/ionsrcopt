import pandas as pd
import numpy as np

def stability_mean_variance_classification(df, value_column, weight_column, sliding_window_size_mean=500, sliding_window_size_std=1000, minimum_mean=0.025, maximum_variance=0.00005):
    """ Classifies all points in the data frame into the categories source stable/unstable, based on a rolling window and a minimum mean and maximum variance in this window.

    Parameters:
        df (DataFrame): The data input loaded as a DataFrame
        current_column (string): name of the column that contains the beam current we are interested in, typically BCT25
        sliding_window_size (int): size of the sliding window, by default 5000 (100 Minutes of data every 1.2 seconds)
        minimum_mean (double): minimal intensity of the beam in the sliding window for it to be considered stable
        maximum_variance (double): maximum variance of intensity of the beam in the sliding window for it to be considered stable

    Returns:
        Series: A series that for every data point indicates if the source was running stable or not (1 is stable, 0 is unstable)
    """

    df['wvalue'] = df[value_column] * df[weight_column]

    mean_weight_sum = df[['wvalue', weight_column]].rolling('{}s'.format(sliding_window_size_mean), closed='left').sum()
    wmean = mean_weight_sum['wvalue'] / mean_weight_sum[weight_column]
    wmean.name = 'wmean'

    df['wdeviation'] = df[value_column] - wmean
    df['wdeviation'] = df['wdeviation'] ** 2
    df['wdeviation'] *= df[weight_column]
    var_weight_sum = df[['wdeviation', weight_column]].rolling('{}s'.format(sliding_window_size_mean), closed='left').sum()
    wvar = var_weight_sum['wdeviation'] / (var_weight_sum[weight_column] - 1)
    wvar.name = 'wvar'

    df.drop(['wvalue', 'wdeviation'], axis=1, inplace=True)

    stats = pd.concat([wmean, wvar], axis=1)
    stats['result'] = 0
    stats.loc[(stats['wmean'] > minimum_mean) & (stats['wvar'] < maximum_variance), 'result'] = 1

    return stats['result']