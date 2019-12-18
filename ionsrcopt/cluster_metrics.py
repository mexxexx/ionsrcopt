import pandas as pd
import numpy as np

from sklearn.preprocessing import RobustScaler
import warnings

def DBI(X, labels):
    num_clusters = labels.max() + 1
    cluster_centers = np.array([X[labels == c].mean(axis=0) for c in range(num_clusters)])
    S = np.array([np.mean(np.linalg.norm(X[labels == c] - cluster_centers[c], axis=1)) for c in range(num_clusters)])

    #M = np.zeros((num_clusters, num_clusters))
    #for i in range(num_clusters):
        #M[:, i] = np.linalg.norm(cluster_centers-cluster_centers[i], axis=1)
    M = np.linalg.norm(cluster_centers[:,None]-cluster_centers, axis=1)

    M = np.where(M == 0, np.inf, M)
    R = S[:,None] + S
    
    R /= M
    D = np.nanmax(R, axis=1)
    return D

def silhouette(X, labels, sample_size):
    num_clusters = np.max(labels) + 1
    sample = np.random.permutation(len(X))[:sample_size]
    X = X[sample]
    labels = labels[sample]

    from sklearn import metrics
    silhouette_samples = metrics.silhouette_samples(X, labels)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        silhouette = np.array([np.mean(silhouette_samples[labels == c]) for c in range(num_clusters)])

    return silhouette