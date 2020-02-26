import pandas as pd
import numpy as np
import seaborn as sns
import sys, os
import matplotlib.pyplot as plt

import argparse

sys.path.insert(1, os.path.abspath('../ionsrcopt'))
import load_data as ld
from source_features import SourceFeatures
from processing_features import ProcessingFeatures

def main():
    ######################
    ###### SETTINGS ######
    ######################

    clustered_data_folder = '../Data_Clustered/' # Base folder of clustered data 
    filename = 'JanNov2016.csv' # The file to load
    
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

    args = parse_args()
    source_stability = args['source_stability']
    cluster = args['cluster']
    sample_size = args['sample_size']

    ######################
    ######## CODE ########
    ######################
    
    path = clustered_data_folder + filename
    df = ld.read_data_from_csv(path, None, None)
    df = ld.fill_columns(df, None, fill_nan_with_zeros=True)
    df = ld.convert_column_types(df)

    df = df.loc[(df[ProcessingFeatures.SOURCE_STABILITY] == source_stability)].copy()
    if not cluster is None:
        df = df.loc[(df[ProcessingFeatures.CLUSTER] == cluster)].copy()

    index_length = len(df.index)
    indices = np.random.permutation(range(index_length))[:min(sample_size, index_length)]

    data = df.loc[df.index[indices]].copy()
    
    sns.pairplot(data, vars=features, hue=ProcessingFeatures.CLUSTER)
    plt.show()

def parse_args():
    parser = argparse.ArgumentParser(description='View time development of clusters')
    parser.add_argument('-s', '--source_stability', default=1, type=int, help='1 if you want to look at the stable source, 0 else')
    parser.add_argument('-c', '--cluster', default=None, type=int, help='The cluster you want to look at, or None for all data')
    parser.add_argument('-n', '--sample_size', default=1000, type=int, help='Number of datapoints to display')

    args = parser.parse_args()

    return {'source_stability' : args.source_stability, 
            'cluster' : args.cluster,
            'sample_size' : args.sample_size}

if __name__ == "__main__":
    main()