""" Implementation of the Optigrid Algorithm described in "Optimal Grid-Clustering: Towards Breaking the Curse of
Dimensionality in High-Dimensional Clustering" by Hinneburg and Keim """

import pandas as pd
import numpy as np
import random
from sklearn.neighbors import KernelDensity

import itertools

import matplotlib.pyplot as plt

def estimate_distribution(data, cluster_indices, current_dimension, num_steps, bandwidth = 0.2, percentage_of_values=1):
    num_samples = 15000
    sample_size = min(num_samples, len(cluster_indices))
    sample = np.random.choice(cluster_indices, size=sample_size)
    datapoints = np.expand_dims(data[sample][:,current_dimension], -1)
    min_val = np.amin(datapoints)
    max_val = np.amax(datapoints)
    grid = np.linspace([min_val], [max_val], num_steps)
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth, atol=1E-6, rtol=1E-4).fit(datapoints)
    log_dens = kde.score_samples(grid)
    return grid, np.exp(log_dens) * percentage_of_values

def create_cuts_kde(data, cluster_indices, q, max_cut_score, noise_level, current_dimension, bandwidth=0.1, resolution=100, percentage_of_values=1):
    grid, kde = estimate_distribution(data, cluster_indices, current_dimension, resolution, bandwidth=bandwidth, percentage_of_values=percentage_of_values) 
    
    #plt.plot(grid, kde)
    #plt.title("Current dimension: {}".format(current_dimension))
    #plt.show()
    
    kde = np.append(kde, 0)

    max=[]
    prev = 0
    current = kde[0]
    for bin in range(1, resolution+1): # Find all peaks that are above the noise level
        next = kde[bin] 
        if current > prev and current > next and current >= noise_level:
            max.append(bin-1)
        prev = current
        current = next
    
    if not max:
        return []

    max = [max[0]] + sorted(sorted(max[1:-1], key=lambda x: kde[x], reverse=True)[:q-1]) + [max[len(max)-1]] # and get the q-1 most important peaks between the leftest and rightest one.

    best_cuts = [] 
    for i in range(len(max)-1): # between these peaks search for the optimal cutting plane
        current_min = 1
        current_min_index = -1
        for j in range(max[i]+1, max[i+1]):
            if kde[j] < current_min:
                current_min = kde[j]
                current_min_index = j
        
        if current_min_index >= 0 and current_min < max_cut_score:
            best_cuts.append((grid[current_min_index], current_dimension, current_min)) # cutting plane format: (cutting coordinate, dimension in which we cut, density at minimum)
    return best_cuts

def create_cuts_histogram(data, cluster_indices, q, max_cut_score, noise_level, current_dimension, bins):
    hist, edges = np.histogram([data[ind][current_dimension] for ind in cluster_indices], density=True, bins=bins[current_dimension]) # First create the histogram of this dimension, 
    hist = np.append(hist, 0) # adding a zero density at the end to avoid the special case in the next loop, when searching for the maxima
    hist *= edges[1] - edges[0]
    
    max=[]
    prev = 0
    current = hist[0]
    for bin in range(len(bins[current_dimension])): # Find all peaks that are above the noise level
        next = hist[bin]
        if current > prev and current > next and current >= noise_level:
            max.append(bin-1)
        prev = current
        current = next
    
    if not max:
        return []

    max = [max[0]] + sorted(sorted(max[1:-1], key=lambda x: hist[x], reverse=True)[:q-1]) + [max[len(max)-1]] # and get the q-1 most important peaks between the leftest and rightest one.

    best_cuts = [] 
    for i in range(len(max)-1): # between these peaks search for the optimal cutting plane
        current_min = 1
        current_min_index = -1
        for j in range(max[i]+1, max[i+1]):
            if hist[j] < current_min:
                current_min = hist[j]
                current_min_index = j
        
        if current_min_index >= 0 and current_min < max_cut_score:
            best_cuts.append(((edges[current_min_index] + edges[current_min_index+1])/2, current_dimension, current_min)) # cutting plane format: (cutting coordinate, dimension in which we cut, density at minimum)
    return best_cuts

def fill_grid(data, cluster_indices, cuts):
    """ Partitions the grid based on the selected cuts and assignes each cell the corresponding data points (as indices)"""
    
    num_cuts = len(cuts)
    grid_index = np.zeros(len(cluster_indices))
    for i, cut in enumerate(cuts):
        cut_val = 2 ** i
        grid_index[np.take(np.take(data, cut[1], axis=1), cluster_indices) > cut[0]] += cut_val

    return [cluster_indices[grid_index==key] for key in range(2**num_cuts)]

def optigrid(data, d, q, max_cut_score, noise_level, cluster_indices=np.array([]), percentage_of_values=1):
    """ Main entry point of the algorithm. 

    Parameters:
        data (list of datapoints): The whole set of datapoints to be considered
        cluster_indices (list of int): The currently considered cluster. This is a list of indices with which the datapoints can be looked up in the data list. If None, then the whole set is considered as one cluster, typically whe the algorithm is started.
        d (int): The number of dimensions of the data
        q (int): number of cutting planes in each iteration
        max_cut_score (double): The maximum density (percentage) in the density estimation histograms that will be used when creating cutting planes. The lower the more different peaks will be grouped inside one cluster.
        noise_level (double): The background noise, everything below this threshold will not influence the cutting planes. As percentage of density.
    
    Returns:
        list of list of int: Each list in this list represents a cluster. The values are again indices which that the datapoints can be looked up in the data list.
    """
    
    if cluster_indices.size == 0:
        cluster_indices = np.array(range(0, len(data)))

    cuts = []
    for i in range(d): # First create all best cuts
        cuts += create_cuts_kde(data, cluster_indices, q, max_cut_score, noise_level, current_dimension=i, percentage_of_values=percentage_of_values)
    
    if not cuts:
        return [cluster_indices]
    
    cuts = sorted(cuts, key=lambda x: x[2]) # Sort the cuts based on the density at the minima

    grid = fill_grid(data, cluster_indices, cuts[:q]) # Fill the subgrid based on the cuts
    
    result = []
    for cluster in grid:
        if cluster.size==0:
            continue
        print("In current cluster: {}".format(percentage_of_values*len(cluster)/len(cluster_indices)))
        result += optigrid(data=data, d=d, q=q, max_cut_score=max_cut_score, noise_level=noise_level, cluster_indices=cluster, percentage_of_values=percentage_of_values*len(cluster)/len(cluster_indices)) # Run Optigrid on every subgrid
    
    return result

def describe_cluster(cluster, columns):
    """ Generate descriptive statistics for a cluster

    Parameters:
        cluster (DataFrame): A dataframe, that contains density informations for every bin in the cluster
        columns (list of string): The names of the columns for which to generate statistics

    Returns: 
        Series: All statistics for the selected columns
    """
    
    mean = cluster.mean(axis = 0) 
    std = cluster.std(axis=0)
    quantiles = cluster.quantile([.25, .5, .75], axis=0)
    mins = cluster.min(axis=0)
    maxs = cluster.max(axis=0)
    
    count = cluster.count(axis=0)[0]
    
    result_columns = [[mean[i], std[i], std[i] / abs(mean[i]) * 100, mins[i], quantiles.iloc[0, i], quantiles.iloc[1, i], quantiles.iloc[2, i], maxs[i]] for i in range(len(columns))]
    result = list(itertools.chain(*result_columns)) + [count]
    
    value_columns = [[(col, 'mean'), (col, 'std'), (col, 'varC (%)'), (col, 'min'), (col, '25%'), (col, '50%'), (col, '75%'), (col, 'max')] for col in columns]
    index = list(itertools.chain(*value_columns)) + [('DENSITY', 'count')]
    
    return pd.Series(result, index=pd.MultiIndex.from_tuples(index))


def describe_clusters(df, columns):
    """ Summarize all clusters and sort them by density

    Parameters:
        df (DataFrame): A frame containing density and cluster information about every bin
        columns (list of string): The names of the columns for which to generate statistics
    
    Returns:
        DataFrame: Descriptive frame sorted by density
    """

    result = df[columns + ['OPTIGRID_CLUSTER']].groupby('OPTIGRID_CLUSTER').apply(describe_cluster, columns)
    return result.sort_values(('DENSITY', 'count'), ascending=0)