import pandas as pd
import numpy as np

def generate_density_histogram(df, columns, bins):
    H, edges = np.histogramdd(df[columns].values, bins=bins, density=True)

    normalization_factor = np.prod([abs(edges[i][1] - edges[i][0]) for i in range(len(bins))])
    H = H * normalization_factor
    return H, edges

# Nearest neighbour clustering
def do_BFS_step(p, clusters, current_cluster, bins, histogram_edges, threshold):
    result = []
    if clusters[p] >= 0 or H[p] < threshold: # Node has to be ignored. Else assign to cluster and find children.
        return result
    
    clusters[p] = current_cluster
    for i in range(len(p)):
        if p[i] > 0:
            result.append(list(p))
            result[-1][i] -= 1
        if p[i] + 1 < bins[i]:
            result.append(list(p))
            result[-1][i] += 1
    return result

def nearest_neighbour_clustering(histogram_values, bins, density_threshold):
    clusters = np.ones(bins)
    clusters *= -1
    current_cluster = 0

    perms = [(a, b, c) for a in range(bins[0]) for b in range(bins[1]) for c in range(bins[2])]
    for p in perms:
        if clusters[p] >= 0 or histogram_values[p] < density_threshold:
            continue
        
        print("Started search for cluster {}".format(current_cluster))
        nodes_to_check = do_BFS_step(p, clusters, current_cluster, bins, histogram_values, density_threshold)
        while len(nodes_to_check) > 0:
            node = nodes_to_check.pop(0)
            nodes_to_check.extend(do_BFS_step(tuple(node), clusters, current_cluster, bins, histogram_values, density_threshold))
        
        current_cluster += 1

    print("Found {} cluster(s)".format(current_cluster))