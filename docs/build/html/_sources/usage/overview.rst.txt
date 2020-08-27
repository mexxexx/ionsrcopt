Overview
========

The package provides a set of tools for the cluster analysis of the Linac3 GTS Ion Source.

How to use it
*************

This is a quick guide on what steps need to be performed. For a more detailed overview on how to use the separate modules, please look at their documentation.

1. Download the data
--------------------

As the first step, you obviously need to download all the data required for your analysis. To learn how to easily extract data from CALS have a look at :ref:`data-loading`.

2. Preprocessing
----------------

After you downloaded the necessary files, you need to perform some preprocessing. This will for example annotate the source stability/instability periods and voltage breakdowns. This is described in :ref:`preprocessing`.

3. Clustering
-------------

Once this is done, you can run the clustering algorithm. This will create one big file that contains all the data from the month you specified as input and a new column with the cluster number each data point belongs to.
You learn how to do this in :ref:`clustering`. Then, it is necessary to assign the voltage breakdown data points to the correct clusters. This is explained here: :ref:`cluster-sparks`.

4. Analysis
-----------

When this is finished, you can explore the scripts of the `visualization` package to analyze the resulting clusters. For a description of the possibilities see :ref:`visualization`.