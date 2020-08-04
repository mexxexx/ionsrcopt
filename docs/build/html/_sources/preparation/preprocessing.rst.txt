Preprocessing
=============

Next, we need to do some preprocessing using the `Preprocessing.ipynb` notebook. The notebook will do the following things:

1. Compute the duration of each data point
2. Compute the stability
3. Detect breakdowns
4. Check if the source was running

1. Computing the duration of individual points
----------------------------------------------

The data is stored only when something changes, hence it might be that between two consecutive points a different amount of time passes.
We want to consider this in our cluster analysis, and we do this by weighting each individual point by the number of settings it remained
unchanged. This is what we call the duration of a data point.

2. Computing the stability
--------------------------

Each point is classified either as a stable or an unstable point, meaning if it occurred during a phase where the source was running stable or not.
This is determined using two sliding windows, one for the mean and one for the variance. As a threshold for the mean we used a value of `15 uA` 
in a rolling window of 1500s, and for the variance a threshold of `0.000035` over a rolling window of 2000s. These values were determined in experiments
to match the interpretation of source experts.

3. Detecting breakdowns
-----------------------

Furthermore, we need to exclude points that happened around voltage breakdowns, as there the HT current is unstable and might introduce noise into
our analysis. Here, sparks should be detected too, but this is currently done in `detect_breakdowns.py` (TODO: add spark detection code to notebook).

4. Check if the source was running
----------------------------------

Finally, we exclude all data points without beam in the BCT05, because for them we have no information about the quality of the shots. 