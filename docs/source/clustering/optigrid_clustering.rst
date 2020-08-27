.. _clustering:

Optigrid Clustering
===================

Once everything is prepared and preprocessed, it is time to find the clusters. You do this by running the 
`Optigrid_Clustering.ipynb` notebook in the `Clustering` package.

Note that the implementation of Optigrid is not completely deterministic, so
different runs can yield slightly different result. This is due to a kernel density
estimation that is only done on a randomly selected sample. However, as the sample
size is chosen very large (15000), the kernel density will be very similar, so the
clusters should not be too different when running it a second time.

First, you need to specify all columns you are interested in. There are three types: 
Parameters, these are the ones that will be clustered later on, Measurements, these will
be added to the output file but don't affect the clustering and columns from preprocessing.

Second, you have to specify the important file. In the input_files list add all files
of the month you want to cluster. They will be all clustered together, so specifying
`input_file=["Sep2016.csv", "Oct2016.csv", "Nov2016.csv"]` will result in clusters that span
over these three months.

Third, we select the values specified in `parameters` and use the column `ProcessingFeatures.DATAPOINT_DURATION`
as weights. We scale the data using a `(10, 90)` quantile Robust scaler, to not be influenced too
much by outliers but to also guarantee that the data dimensions are more or less on the same scale.

After this, we have to select the parameters for the optigrid algorithm. Every iteration we use only one
cutting plane (`q=1`). The `max_cut_score=0.04` and `noise_level=0.05` indicate when a group
of points is considered as noise. In every step, Optigrid performs a density estimation for every
dimension in the current subgrid. By definition, the integral under this function is one, so it is multiplied
with the percentage of points in the current subgrid. A `max_cut_score` of `0.04` then says that
if a local minimum of this adjusted kernel density has a value below `0.04`, a cut can be performed there.
This is not equivalent to saying that at a cluster contains at least 4% of all points, as this would involve
integrating the kernel density function. Therefore this value of `0.04` cannot be directly translated into
something visual and was found by tweaking. If you feel that your resulting clusters separate points either too much
or not enough, you should try to change these parameters. Another parameter is the `kde_bandwidth` parameter,
that controls the bandwidth of the kernels in the density estimation. Depending on how exact the cluster should be,
meaning how much variation you want to allow, you can change them for each parameter individually.

The notebook then does two independent cluster runs. Once for the stable source and once for the unstable source.
This is why, to tell to which cluster a point belongs, the source stability always is also a needed information.
For example, each run creates a cluster with the index zero. However, points in the cluster 0 with a stable source
obviously are different from the ones with cluster zero and an unstable source. For the cluster analysis, all
HT voltage breakdown points are ignored. They are then assigned to a cluster using the `cluster_sparks.py` script
described on the next page.