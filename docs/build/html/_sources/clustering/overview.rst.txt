The Clustering package
======================

The clustering package does the clustering in two steps: The `Optigrid_Clustering.ipynb` 
notebook is there to apply Optigrid to all data points where the source was on and not 
in a breakdown phase, and the `cluster_sparks.py` script later assigns individual sparks 
to a cluster, so that we are able to compare the breakdown rate across settings.

.. toctree::
    :maxdepth: 1
    :caption: Modules

    optigrid_clustering
    cluster_sparks