""" Implementation of the Optigrid Algorithm described in Optimal Grid-Clustering: Towards Breaking the Curse of
Dimensionality in High-Dimensional Clustering by Hinneburg and Keim """

import numpy as np

def create_cuts(data, cluster_indices, q, max_cut_score, noise_level, i, bins):
    hist, edges = np.histogram([data[ind][i] for ind in cluster_indices], density=True, bins=bins[i]) # First create the histogram of this dimension, 
    hist = np.append(hist, 0)                                                                         # adding a zero density at the end to avoid the special case in the next loop, when searching for the maxima
    noise_level *= edges[1] - edges[0]

    max=[]
    prev = 0
    current = hist[0]
    for bin in range(1, len(bins[i])): # Find all peaks that are above the noise level
        next = hist[bin]
        if current > prev and current > next and current >= noise_level:
            max.append(bin-1)
        prev = current
        current = next
    
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
            best_cuts.append(((edges[current_min_index] + edges[current_min_index+1])/2, i, current_min)) # cutting plane format: (cutting coordinate, dimension in which we cut, density at minimum)
    return best_cuts

def fill_grid(grid, data, cluster_indices, cuts):
    """ Partitions the grid based on the selected cuts and assignes each cell the corresponding data points (as indices)"""
    for ind in cluster_indices:
        grid_index = 0
        value = data[ind]
        for cut in cuts:
            if value[cut[1]] >= cut[0]:
                grid_index += 2 ** cut[2]
        
        grid[grid_index].append(ind)

def create_bins(data, number_of_bins):
    mins = np.nanmin(data, axis=0)
    maxs = np.nanmax(data, axis=0)
    bins = [[mins[j] + i / number_of_bins * (maxs[j] - mins[j]) for i in range(number_of_bins+1)] for j in range(len(mins))]
    for i in range(len(mins)):
        if mins[i] == maxs[i]:
            bins[i] = [mins[i], mins[i]+1]

    return bins

def optigrid(data, d, q, max_cut_score, noise_level, cluster_indices=None, bins=None, number_of_bins=25):
    """ Main entry point of the algorithm. 

    Parameters:
        data (list of datapoints): The whole set of datapoints to be considered
        cluster_indices (list of int): The currently considered cluster. This is a list of indices with which the datapoints can be looked up in the data list. If None, then the whole set is considered as one cluster, typically whe the algorithm is started.
        d (int): The number of dimensions of the data
        q (int): number of cutting planes in each iteration
        max_cut_score (double): The maximum density (percentage) in the density estimation histograms that will be used when creating cutting planes. The lower the more different peaks will be grouped inside one cluster.
        noise_level (double): The background noise, everything below this threshold will not influence the cutting planes. As percentage of density.
        bins: The bins to use for the histograms in every dimension. If None, then number_of_bins bins will be created per dimension.
    
    Returns:
        list of list of int: Each list in this list represents a cluster. The values are again indices which that the datapoints can be looked up in the data list.
    """
    
    if not bins:
        bins = create_bins(data, number_of_bins)
    if not cluster_indices:
        cluster_indices = list(range(0, len(data)))

    cuts = []
    for i in range(d): # First create all best cuts
        cuts += create_cuts(data, cluster_indices, q, max_cut_score, noise_level, i, bins)
    
    if not cuts:
        return [cluster_indices]
    
    cuts = sorted(cuts, key=lambda x: x[2]) # Sort the cuts based on the density at the minima
    cuts = [(cuts[i][0], cuts[i][1], i) for i in range(q)] # and select the q best ones

    grid = [[] for i in range(2 ** q)]
    fill_grid(grid, data, cluster_indices, cuts) # Fill the subgrid based on the cuts

    result = []
    for cluster in grid:
        if not cluster:
            continue
        result += optigrid(data=data, d=d, q=q, max_cut_score=max_cut_score, noise_level=noise_level, cluster_indices=cluster, bins=bins) # Run Optigrid on every subgrid
    
    return result
        
import matplotlib.pyplot as plt

np.random.seed(0)
v = list(np.random.normal(-2.4, 1 , 200000))
#v += [3.3] * 50 #list(np.random.normal(7, 1, 200))
v += list(np.random.normal(1.4, 1, 2000000))
#v += [6.7] * 60 #list(np.random.normal(7, 1, 200))
v += list(np.random.normal(2.9, 1, 2000000))
v = [[x, 0] for x in v]
#print(v)

d=2
q=1
max_cut_score = 0.1
noise_level = 0.02
opt = optigrid(v, d, 1, max_cut_score, noise_level)
cluster = [[v[i] for i in cl] for cl in opt]
print(len(cluster))

plt.hist(x=[x[0] for x in v], bins = 25, density=True)
plt.show()