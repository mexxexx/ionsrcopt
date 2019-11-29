import pandas as pd
pd.set_option('display.max_columns', 500)
pd.set_option('display.expand_frame_repr', False)
import matplotlib.pyplot as plt
import numpy as np
import contextlib
import io
import sys

from scipy.stats import gaussian_kde

import load_data as ld

def main():
    ######################
    ###### SETTINGS ######
    ######################

    clustered_data_folder = 'Data_Clustered/' # Base folder of clustered data 
    filename = 'Nov2016.csv' # The file to load

    source_stability = 1 # 1 if we want to look at a stable source, 0 else
    cluster = None # The cluster to plot or None if you want to plot all data

    parameters = ['IP.NSRCGEN:BIASDISCAQNV', 'IP.NSRCGEN:GASSASAQN', 'IP.SOLCEN.ACQUISITION:CURRENT', 'IP.SOLEXT.ACQUISITION:CURRENT', 'IP.NSRCGEN:OVEN1AQNP', 'ITF.BCT25:CURRENT'] # Parameters to be displayed    

    ######################
    ######## CODE ########
    ######################

    # Load file into a data frame
    path = clustered_data_folder + filename
    df = ld.read_data_from_csv(path, None, None)
    with nostdout():
        df = ld.convert_column_types(df)
    df.dropna()    

    if cluster is not None:
        df = df[df['optigrid_cluster'] == cluster].copy()
    df = df.loc[df['source_stable'] == source_stability, parameters].copy()

    resolution = 200
    bandwidth = np.array([1, 0.001, 0.5, 5, 0.1, 0.0001])
    if cluster is not None:
        bandwidth *= 0.2
    num_kde_samples = 40000

    plot_cluster(df.values, None, parameters, parameter_ranges=None, median=None, resolution=resolution, bandwidth=bandwidth, num_kde_samples=num_kde_samples, cluster=cluster)

def plot_cluster(data, weights, parameters, parameter_ranges, median, resolution, bandwidth, num_kde_samples, cluster):
    if isinstance(bandwidth, float):
        bandwidth = [bandwidth for i in range(len(parameters))]
    
    fig = plt.figure()
    
    for i, parameter in enumerate(parameters):
        grid, kde = estimate_distribution(data, weights, i, resolution, bandwidth=bandwidth[i], num_kde_samples=num_kde_samples)
        ax = plt.subplot('{}1{}'.format(len(parameters), i+1))
        ax.set_title("{}".format(parameter))
        ax.tick_params(axis='both', which='major')
        if parameter_ranges:
            ax.set_xlim(*parameter_ranges[i])
            #ax.set_ylim(*parameter_ranges[i][1])
            
        if median is not None:
            ax.axvline(x=median[i], color='red')
        
        ax.plot(grid, kde)

    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    fig.suptitle('Densities of specified parameters of cluster {}'.format(cluster))
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.93, wspace=None, hspace=0.4)
    plt.show()

def estimate_distribution(data, weights, current_dimension, num_steps, bandwidth = 0.1, num_kde_samples=15000):
    sample_size = min(num_kde_samples, len(data))
    sample = np.random.randint(0, len(data), size=sample_size)
    datapoints = data[sample, current_dimension]

    weights_sample = None
    if not weights is None:
        weights_sample = weights[sample]

    min_val = np.amin(datapoints)
    max_val = np.amax(datapoints)
    grid = np.linspace(min_val, max_val, num_steps)

    kde = gaussian_kde(dataset=datapoints, bw_method=bandwidth / datapoints.std(ddof=1), weights=weights_sample)
    dens = kde.evaluate(grid)
    return grid, dens

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