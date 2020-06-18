Optigrid Clustering
===================

Once everything is prepared and preprocessed, it is time to find the clusters.

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