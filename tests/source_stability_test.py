import pandas as pd
import numpy as np

from ionsrcopt import source_stability as stab

class TestSourceStability:
    def test_stability_mean_variance_classification(self):
        def timedelta_to_seconds(timedelta):
            if not pd.isnull(timedelta):
                return timedelta.total_seconds()
            else:
                return np.nan

        values = 10000 * [1]
        timestamps = range(0, (int)(1e13), (int)(1e9))

        df = pd.DataFrame()
        df['Timestamp'] = timestamps
        df['value'] = values
        df['Timestamp'] = pd.to_datetime(df['Timestamp']) 
        df = df.set_index(pd.DatetimeIndex(df['Timestamp']))

        weights = (df.index.to_series().diff(-1)).apply(timedelta_to_seconds).values
        weights *= -1
        df['weight'] = weights

        df['is_stable'] = stab.stability_mean_variance_classification(df, 'value', 'weight')
        assert False #NOT IMPLEMENTED