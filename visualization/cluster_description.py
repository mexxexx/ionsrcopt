import pandas as pd
pd.set_option('display.max_columns', 500)
pd.set_option('display.expand_frame_repr', False)
import numpy as np
import sys
import os

from statsmodels.stats.weightstats import DescrStatsW
from sklearn import preprocessing

import argparse

sys.path.insert(1, os.path.abspath('../ionsrcopt'))
import load_data as ld
from source_features import SourceFeatures
from processing_features import ProcessingFeatures
import cluster_metrics

def main():
    ######################
    ###### SETTINGS ######
    ######################

    #clustered_data_folder = '../Data_Clustered/' # Base folder of clustered data
    #filename = 'JanNov2018_lowbandwidth.csv' # The file to load

    input_file = '../Data_Clustered/JanNov2016.csv'
    output_file = './Results/JanNov2016.csv'

    features = [
        SourceFeatures.BIASDISCAQNV, 
        SourceFeatures.GASAQN, 
        SourceFeatures.OVEN1AQNP,
        SourceFeatures.THOMSON_FORWARDPOWER,
        SourceFeatures.SOLINJ_CURRENT,
        SourceFeatures.SOLCEN_CURRENT,
        SourceFeatures.SOLEXT_CURRENT,
        SourceFeatures.SOURCEHTAQNI,
        SourceFeatures.BCT25_CURRENT] # Features to be displayed
    statistics = ['50%', 'std'] # Statistics we are interested in
 
    args = parse_args()
    source_stability = args['source_stability']
    count_breakdowns_per_cluster = args['count_breakdowns_per_cluster']
    num_clusters_to_visualize = args['num_clusters_to_visualize']
    print_to_file = args['print_to_file']
    display_metrics = args['display_metrics']

    ######################
    ######## CODE ########
    ######################

    # Load file into a data frame
    df = ld.read_data_from_csv(input_file, None, None)
    df = ld.fill_columns(df, None, fill_nan_with_zeros=True)
    df = ld.convert_column_types(df)

    for feature in features:
        if feature not in df.columns:
            print("{} does not exist as a feature in the loaded file. Aborting.".format(feature))
            return

    # Select only the stability interested in
    df = df[df[ProcessingFeatures.SOURCE_STABILITY] == source_stability].copy() 
    total_duration = df[ProcessingFeatures.DATAPOINT_DURATION].sum() / 3600
    
    # Describe the clusters
    print("Calculating statistics...")
    described = df.groupby(ProcessingFeatures.CLUSTER).apply(describe_cluster, features=features, weight_column=ProcessingFeatures.DATAPOINT_DURATION)
    described[('DENSITY', 'percentage')] = described[('DURATION', 'in_hours')] / total_duration * 100

    # Gather statistics to output
    wanted_statistics = get_wanted_statistics(features, statistics) + [('DENSITY', 'percentage'), ('DURATION', 'in_hours')] 
    if count_breakdowns_per_cluster:
        wanted_statistics += [('num_breakdowns', 'per_hour')]

    # Calculate metrics
    if display_metrics:
        metrics = calculate_metrics(df, features)
        print('DBI is {}'.format(np.mean(metrics['DBI'])))
        described.loc[described.index >= 0, ('METRICS', 'DBI')] = metrics['DBI']
        print('Silhouette is {}'.format(np.mean(metrics['silhouette'])))
        described.loc[described.index >= 0, ('METRICS', 'silhouette')] = metrics['silhouette']

        wanted_statistics += [('METRICS', 'DBI'), ('METRICS', 'silhouette')]

    described.sort_values(by=[('DENSITY', 'percentage')], ascending=False, inplace=True)

    print("Rounding values...")
    printable_clusters = described[wanted_statistics].head(n=num_clusters_to_visualize)
    print("Sum of densities of printed clusters: {:.1f}%".format(printable_clusters[('DENSITY', 'percentage')].sum()))
    printable_clusters = round_described(printable_clusters, {
        SourceFeatures.BIASDISCAQNV : 0, 
        SourceFeatures.GASAQN : 2, 
        SourceFeatures.OVEN1AQNP : 1,
        SourceFeatures.THOMSON_FORWARDPOWER : 0,
        SourceFeatures.SAIREM2_FORWARDPOWER : 0,
        SourceFeatures.SOLINJ_CURRENT : 0,
        SourceFeatures.SOLCEN_CURRENT : 0,
        SourceFeatures.SOLEXT_CURRENT : 0,
        SourceFeatures.SOURCEHTAQNI : 2,
        SourceFeatures.BCT25_CURRENT : 3
    })
    printable_clusters.rename({
        SourceFeatures.BIASDISCAQNV : 'bias disc', 
        SourceFeatures.GASAQN : 'gas', 
        SourceFeatures.OVEN1AQNP : 'oven1',
        SourceFeatures.SAIREM2_FORWARDPOWER : 'RF', 
        SourceFeatures.THOMSON_FORWARDPOWER : 'RF',
        SourceFeatures.SOLINJ_CURRENT : 'solinj',
        SourceFeatures.SOLCEN_CURRENT : 'solcen',
        SourceFeatures.SOLEXT_CURRENT : 'solext',
        SourceFeatures.SOURCEHTAQNI : 'HTI',
        SourceFeatures.BCT25_CURRENT : 'BCT25'
    }, axis='columns', inplace=True)


    if print_to_file:
        printable_clusters.to_csv(output_file)
        print("Saved result to {}".format(output_file))
    else:
        print(printable_clusters)

def round_described(described, decimals):
    for k, v in decimals.items():
        described = described.round({
            (k, 'mean') : v,
            (k, 'std') : min(v+1, 3),
            (k, 'avg_dev') : v,
            (k, 'min') : v,
            (k, '25%') : v,
            (k, '50%') : v,
            (k, '75%') : v,
            (k, 'max') : v
            })
    
    return described.round({
        ('DENSITY', 'percentage') : 1,
        ('DURATION', 'in_hours') : 1,
        ('num_breakdowns', 'per_hour') : 2
    })

def describe_cluster(cluster_df, features, weight_column):
    values = ['mean', 'std', 'avg_dev', 'min', '25%', '50%', '75%', 'max']
    index = pd.MultiIndex.from_tuples([(p, v) for p in features for v in values] + [('DENSITY', 'count'), ('DURATION', 'in_hours'), ('num_breakdowns', 'per_hour')])
    
    data = cluster_df.loc[(cluster_df[ProcessingFeatures.HT_VOLTAGE_BREAKDOWN] == 0), features].values # TODO maybe only include non breakdown here???
    weights = cluster_df.loc[(cluster_df[ProcessingFeatures.HT_VOLTAGE_BREAKDOWN] == 0), weight_column].values
    if data.size == 0:
        return None

    stats = DescrStatsW(data, weights, ddof=1)

    mean = np.array(stats.mean) #np.mean(data, axis=0)
    std = np.array(stats.std) #np.std(data, axis=0)
    quantiles = stats.quantile([0, 0.25, 0.5, 0.75, 1], return_pandas=False) #np.quantile(data, [0, 0.25, 0.5, 0.75, 1], axis=0)
    avg_dev = np.dot(weights, np.absolute(data - mean)) / np.sum(weights)

    count = len(data)

    duration_in_seconds = cluster_df[ProcessingFeatures.DATAPOINT_DURATION].sum()
    duration_in_hours = duration_in_seconds / 3600

    description = [[mean[i], std[i], avg_dev[i], quantiles[0][i], quantiles[1][i], quantiles[2][i], quantiles[3][i], quantiles[4][i]] for i in range(len(features))]
    description = [item for sublist in description for item in sublist]
    description.append(count)
    description.append(duration_in_hours)
    description.append(cluster_df.loc[cluster_df[ProcessingFeatures.HT_VOLTAGE_BREAKDOWN] > 0, ProcessingFeatures.HT_VOLTAGE_BREAKDOWN].nunique() / duration_in_hours)
    
    return pd.Series(description, index=index)

def get_cluster_duration(cluster_df):
    index_data = cluster_df.index.values
    continuous_interval_beginning_points =  index_data[np_shift(index_data, num=1, fill_value=-1) != index_data - 1]
    continuous_interval_end_points = index_data[np_shift(index_data, num=-1, fill_value=-1) != index_data + 1]
    
    duration = 0
    for start, end in zip(continuous_interval_beginning_points, continuous_interval_end_points):
        time_start = cluster_df.loc[start, SourceFeatures.TIMESTAMP]
        time_end = cluster_df.loc[end, SourceFeatures.TIMESTAMP]
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

def get_wanted_statistics(features, statistics):
    result = [[(param, stat) for stat in statistics] for param in features]
    result = [item for sublist in result for item in sublist]
    return result

def calculate_metrics(df, features):
    X = df.loc[(df[ProcessingFeatures.HT_VOLTAGE_BREAKDOWN] == 0) & (df[ProcessingFeatures.CLUSTER] >= 0), features].values
    scaler = preprocessing.RobustScaler((10,90)).fit(X)
    X = scaler.transform(X)
    labels = df.loc[(df[ProcessingFeatures.HT_VOLTAGE_BREAKDOWN] == 0) & (df[ProcessingFeatures.CLUSTER] >= 0), ProcessingFeatures.CLUSTER].values

    dbi = cluster_metrics.DBI(X, labels)
    silhouette = cluster_metrics.silhouette(X, labels, 70000)

    return { 'DBI' : dbi, 'silhouette' : silhouette }

def parse_args():
    parser = argparse.ArgumentParser(description='Describe clusters')
    parser.add_argument('-s', '--source_stability', default=1, type=int, help='1 if you want to look at the stable source, 0 else')
    parser.add_argument('-b', '--count_breakdowns_per_cluster', default='n', type=bool, help='Count how many breakdowns occur per cluster? [y/n]')
    parser.add_argument('-v', '--num_clusters_to_visualize', default=20, type=int, help='How many clusters shall be displayed')
    parser.add_argument('-f', '--print_to_file', default='n', type=str, help='Print the results to a file? (y/n)')
    parser.add_argument('-m', '--display_metrics', default='n', type=str, help='Print clustering metrics? (y/n)')

    args = parser.parse_args()

    return {'source_stability' : args.source_stability, 
            'count_breakdowns_per_cluster' : True if args.count_breakdowns_per_cluster=='y' else False,
            'num_clusters_to_visualize' : args.num_clusters_to_visualize,
            'print_to_file' : True if args.print_to_file=='y' else False,
            'display_metrics' : True if args.display_metrics=='y' else False}

if __name__ == "__main__":
    main()