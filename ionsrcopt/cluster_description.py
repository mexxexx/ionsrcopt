import pandas as pd
pd.set_option('display.max_columns', 500)
pd.set_option('display.expand_frame_repr', False)
import numpy as np
import contextlib
import io
import sys

from statsmodels.stats.weightstats import DescrStatsW

import argparse

import load_data as ld

def main():
    ######################
    ###### SETTINGS ######
    ######################

    clustered_data_folder = 'Data_Clustered/' # Base folder of clustered data
    filename = 'JanNov2018_lowbandwidth.csv' # The file to load

    args = parse_args()
    source_stability = args['source_stability']
    count_breakdowns_per_cluster = args['count_breakdowns_per_cluster']
    num_clusters_to_visualize = args['num_clusters_to_visualize']

    parameters = ['IP.NSRCGEN:BIASDISCAQNV', 'IP.NSRCGEN:GASSASAQN', 'IP.SOLCEN.ACQUISITION:CURRENT', 'IP.SOLEXT.ACQUISITION:CURRENT', 'IP.NSRCGEN:OVEN1AQNP', 'ITF.BCT25:CURRENT'] # Parameters to be displayed
    statistics = ['50%', 'std', 'avg_dev'] # Statistics we are interested in
 

    ######################
    ######## CODE ########
    ######################

    # Load file into a data frame
    path = clustered_data_folder + filename
    df = ld.read_data_from_csv(path, None, None)
    with nostdout():
        df = ld.convert_column_types(df)
    df.dropna()

    # Select only the stability interested in
    df = df[df['source_stable'] == source_stability].copy() 
    total_duration = df['duration_seconds'].sum() / 3600

    from sklearn.preprocessing import StandardScaler
    #df[parameters] = StandardScaler().fit_transform(df[parameters].values)
    
    # Describe the clusters
    described = df.groupby('optigrid_cluster').apply(describe_cluster, parameters=parameters, weight_column='duration_seconds')
    described[('DENSITY', 'percentage')] = described[('DURATION', 'in_hours')] / total_duration * 100
    described.sort_values(by=[('DENSITY', 'percentage')], ascending=False, inplace = True)

    # Gather statistics to output
    wanted_statistics = get_wanted_statistics(parameters, statistics) + [('DENSITY', 'percentage'), ('DURATION', 'in_hours')] 
    if count_breakdowns_per_cluster:
        wanted_statistics += [('num_breakdowns', 'per_hour')]

    printable_clusters = described[wanted_statistics].head(n=num_clusters_to_visualize)
    print("Sum of densities of printed clusters: {:.1f}%".format(printable_clusters[('DENSITY', 'percentage')].sum()))
    print(printable_clusters.round(3))

def describe_cluster(cluster_df, parameters, weight_column):
    values = ['mean', 'std', 'avg_dev', 'min', '25%', '50%', '75%', 'max']
    index = pd.MultiIndex.from_tuples([(p, v) for p in parameters for v in values] + [('DENSITY', 'count'), ('DURATION', 'in_hours'), ('num_breakdowns', 'per_hour')])
    
    data = cluster_df.loc[(cluster_df['duration_seconds'] < 60) & (cluster_df['is_breakdown'] == 0), parameters].values # TODO maybe only include non breakdown here???
    weights = cluster_df.loc[(cluster_df['duration_seconds'] < 60) & (cluster_df['is_breakdown'] == 0), weight_column].values

    stats = DescrStatsW(data, weights, ddof=1)

    mean = np.array(stats.mean) #np.mean(data, axis=0)
    std = np.array(stats.std) #np.std(data, axis=0)
    quantiles = stats.quantile([0, 0.25, 0.5, 0.75, 1], return_pandas=False) #np.quantile(data, [0, 0.25, 0.5, 0.75, 1], axis=0)
    avg_dev = np.dot(weights, np.absolute(data - mean)) / np.sum(weights)

    count = len(data)

    duration_in_seconds = cluster_df['duration_seconds'].sum()
    duration_in_hours = duration_in_seconds / 3600

    description = [[mean[i], std[i], avg_dev[i], quantiles[0][i], quantiles[1][i], quantiles[2][i], quantiles[3][i], quantiles[4][i]] for i in range(len(parameters))]
    description = [item for sublist in description for item in sublist]
    description.append(count)
    description.append(duration_in_hours)
    description.append(cluster_df.loc[cluster_df['is_breakdown'] > 0, 'is_breakdown'].nunique() / duration_in_hours)
    
    return pd.Series(description, index=index)

def get_cluster_duration(cluster_df):
    index_data = cluster_df.index.values
    continuous_interval_beginning_points =  index_data[np_shift(index_data, num=1, fill_value=-1) != index_data - 1]
    continuous_interval_end_points = index_data[np_shift(index_data, num=-1, fill_value=-1) != index_data + 1]
    
    duration = 0
    for start, end in zip(continuous_interval_beginning_points, continuous_interval_end_points):
        time_start = cluster_df.loc[start, 'Timestamp']
        time_end = cluster_df.loc[end, 'Timestamp']
        delta = time_end - time_start
        duration += delta.total_seconds()

    return duration

def np_shift(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result

def get_wanted_statistics(parameters, statistics):
    result = [[(param, stat) for stat in statistics] for param in parameters]
    result = [item for sublist in result for item in sublist]
    return result

def parse_args():
    parser = argparse.ArgumentParser(description='Describe clusters')
    parser.add_argument('-s', '--source_stability', default=1, type=int, help='1 if you want to look at the stable source, 0 else')
    parser.add_argument('-b', '--count_breakdowns_per_cluster', default=True, type=bool, help='Count how many breakdowns occur per cluster, True or False')
    parser.add_argument('-v', '--num_clusters_to_visualize', default=20, type=int, help='How many clusters shall be displayed')

    args = parser.parse_args()

    return {'source_stability' : args.source_stability, 
            'count_breakdowns_per_cluster' : args.count_breakdowns_per_cluster,
            'num_clusters_to_visualize' : args.num_clusters_to_visualize}

### This is used to supress output to the console
class DummyFile(object):
    def write(self, x): pass

@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = DummyFile()
    yield
    sys.stdout = save_stdout

if __name__ == "__main__":
    main()