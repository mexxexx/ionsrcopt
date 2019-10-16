#%% 
import numpy as np
import pandas as pd

#%% 
# We will generate various samples, where we know how the clusters look like

# First, a multivariate normal that streches in the x and y directions
mean = [0,0,0]
cov = [[10, 0, 0], [0, 10, 0], [0, 0, 0.1]]
x, y, z = np.random.multivariate_normal(mean, cov, 2500000).T
df = pd.DataFrame({'X':x, 'Y':y, 'Z':z})
#%% 

#%% Visualizing the data
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams["figure.figsize"] = (15,15)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(df['X'], df['Y'], df['Z'], c='r', marker='o')
#ax.scatter(x1, y1, z1, c='r', marker='o')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()

#%%
# Clustering using nearest-neighbour-clustering

import importlib
import histogram_clustering as hc
importlib.reload(hc)

#%%
bins = [100, 100, 100] 
cols_to_cluster = ['X', 'Y', 'Z']
H, edges = hc.generate_density_histogram(df, cols_to_cluster, bins)
hist_mean = np.mean(H)
hist_std = np.std(H)
threshold = hist_mean
print("Threshold: {}".format(threshold))

clusters = hc.nearest_neighbour_clustering(H, bins, threshold)
cl_df = hc.create_cluster_frame(edges, H, bins, clusters, cols_to_cluster)
cl_df.group_by('CLUSTER').describe().sort_values([('DENSITY', 'count')], ascending=0)

#%%
