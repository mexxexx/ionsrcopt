import pandas as pd
pd.set_option('display.max_columns', 500)
pd.set_option('display.expand_frame_repr', False)
import numpy as np
import contextlib
import io
import sys

import load_data as ld

def main():
    ######################
    ###### SETTINGS ######
    ######################

    clustered_data_folder = 'Data_Clustered/' # Base folder of clustered data 
    filename = 'Nov2016.csv' # The file to load

    source_stability = 0 # 1 if we want to look at a stable source, 0 else
    count_breakdowns_per_cluster = True # Whether we want to see the number of high voltage breakdowns per cluster or not

    parameters = ['IP.NSRCGEN:BIASDISCAQNV', 'IP.NSRCGEN:GASSASAQN', 'IP.SOLCEN.ACQUISITION:CURRENT', 'IP.SOLEXT.ACQUISITION:CURRENT', 'IP.NSRCGEN:OVEN1AQNP', 'ITF.BCT25:CURRENT'] # Parameters to be displayed
    statistics = ['mean', 'std'] # Statistics we are interested in

    num_clusters_to_visualize = 10 # Number of clusters we want to see




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
    num_values = len(df.index)
    
    # Describe the clusters
    described = df.groupby('optigrid_cluster').apply(describe_cluster, parameters=parameters)
    described[('DENSITY', 'percentage')] = described[('DENSITY', 'count')] / num_values * 100
    described.sort_values(by=[('DENSITY', 'percentage')], ascending=False, inplace = True)

    # Gather statistics to output
    wanted_statistics = get_wanted_statistics(parameters, statistics) + [('DENSITY', 'percentage')]
    if count_breakdowns_per_cluster:
        wanted_statistics += [('num_breakdowns', '')]

    printable_clusters = described[wanted_statistics].head(n=num_clusters_to_visualize)
    print("Sum of densities of printed clusters: {:.1f}%".format(printable_clusters[('DENSITY', 'percentage')].sum()))
    print(printable_clusters.round(3))

def describe_cluster(cluster_df, parameters):
    values = ['mean', 'std', 'min', '25%', '50%', '75%', 'max']
    index = pd.MultiIndex.from_tuples([(p, v) for p in parameters for v in values] + [('DENSITY', 'count'), ('num_breakdowns', '')])
    
    data = cluster_df[parameters].values # TODO maybe only include non breakdown here???
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    quantiles = np.quantile(data, [0, 0.25, 0.5, 0.75, 1], axis=0)

    count = len(data)

    description = [[mean[i], std[i], quantiles[0][i], quantiles[1][i], quantiles[2][i], quantiles[3][i], quantiles[4][i]] for i in range(len(parameters))]
    description = [item for sublist in description for item in sublist]
    description.append(count)
    description.append(cluster_df.loc[cluster_df['is_breakdown'] > 0, 'is_breakdown'].nunique())
    
    return pd.Series(description, index=index)

def get_wanted_statistics(parameters, statistics):
    result = [[(param, stat) for stat in statistics] for param in parameters]
    result = [item for sublist in result for item in sublist]
    return result

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