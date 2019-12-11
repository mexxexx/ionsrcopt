import pandas as pd
pd.set_option('display.max_columns', 500)
pd.set_option('display.expand_frame_repr', False)
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

from scipy.stats import gaussian_kde

sys.path.insert(1, os.path.abspath('../ionsrcopt'))
import load_data as ld
from source_features import SourceFeatures
from processing_features import ProcessingFeatures

def main():
    ######################
    ###### SETTINGS ######
    ######################

    clustered_data_folder = '../Data_Clustered/' # Base folder of clustered data 
    filename = 'JanNov2018.csv' # The file to load

    source_stability = 1 # 1 if we want to look at a stable source, 0 else
    cluster = 23 # The cluster to plot or None if you want to plot all data

    features = [
        SourceFeatures.BIASDISCAQNV, 
        SourceFeatures.GASAQN, 
        SourceFeatures.OVEN1AQNP,
        SourceFeatures.SAIREM2_FORWARDPOWER,
        SourceFeatures.SOLINJ_CURRENT,
        SourceFeatures.SOLCEN_CURRENT,
        SourceFeatures.SOLEXT_CURRENT,
        SourceFeatures.SOURCEHTAQNI,
        SourceFeatures.BCT25_CURRENT] # Features to be displayed

    normalize = True # Do we want to standard scale the data? 
    bandwidth = np.array([1, 0.001, 0.5, 5, 0.1, 0.0001]) *5 # bandwidth for unnormalized data
    bandwidth = 0.06

    ######################
    ######## CODE ########
    ######################

    # Load file into a data frame
    path = clustered_data_folder + filename
    df = ld.read_data_from_csv(path, None, None)
    df = ld.fill_columns(df, None, fill_nan_with_zeros=True)
    df = ld.convert_column_types(df)

    df = df.loc[df[ProcessingFeatures.SOURCE_STABILITY] == source_stability, :].copy()
    total_duration = df[ProcessingFeatures.DATAPOINT_DURATION].sum()

    data = df[features].values
    if normalize:
        #data = (data - np.mean(data, axis=0)) / np.std(data, axis=0) #Standard scaling
        data = (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0)) #MinMax scaling

    if cluster is not None:
        data = data[df[ProcessingFeatures.CLUSTER]==cluster]

    resolution = 10000
    #if cluster is not None:
    #    bandwidth *= 0.2
    num_kde_samples = 40000
    weights = df[ProcessingFeatures.DATAPOINT_DURATION].values
    cluster_duration = np.sum(weights)
    percentage_of_values = cluster_duration/total_duration

    plot_cluster(data, weights, features, feature_ranges=None, median=None, resolution=resolution, bandwidth=bandwidth, num_kde_samples=num_kde_samples, cluster=cluster, percentage_of_values=percentage_of_values)

def plot_cluster(data, weights, features, feature_ranges, median, resolution, bandwidth, num_kde_samples, cluster, percentage_of_values):
    if isinstance(bandwidth, float):
        bandwidth = [bandwidth for i in range(len(features))]
    
    fig, ax = plt.subplots(len(features), 1, sharex=True)
    for i, feature in enumerate(features):
        grid, kde = estimate_distribution(data, weights, i, resolution, bandwidth=bandwidth[i], num_kde_samples=num_kde_samples)
        ax[i].set_title("{}".format(feature))
        ax[i].tick_params(axis='both', which='major')
        if feature_ranges:
            ax[i].set_xlim(*feature_ranges[i])
            #ax.set_ylim(*feature_ranges[i][1])
            
        if median is not None:
            ax[i].axvline(x=median[i], color='red')
        
        ax[i].grid(True)
        ax[i].plot(grid, kde)

    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    fig.suptitle('Densities of specified features of cluster {}'.format(cluster))
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.93, wspace=None, hspace=0.4)
    plt.show()

def estimate_distribution(data, weights, current_dimension, num_steps, bandwidth = 0.1, num_kde_samples=15000, percentage_of_values=1):
    sample_size = min(num_kde_samples, len(data))
    sample = np.random.randint(0, len(data), size=sample_size)
    datapoints = data[sample, current_dimension]

    weights_sample = None
    if not weights is None:
        weights_sample = weights[sample]

    min_val = np.amin(datapoints)
    max_val = np.amax(datapoints)
    grid = np.linspace(min_val, max_val, num_steps)

    kde = gaussian_kde(dataset=datapoints, bw_method='scott', weights=weights_sample)
    dens = kde.evaluate(grid)
    return grid, dens * percentage_of_values

if __name__ == "__main__":
    main()