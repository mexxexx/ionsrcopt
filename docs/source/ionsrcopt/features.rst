Features
========
All our data is stored in .csv files and in memory in pandas DataFrames.
To make usage easier and to avoid hard coding strings, the package provides
two classes, that provide python variables for feature names.

On the one hand, there are features that are extracted from CALS. These can
be found in the `SourceFeatures` class. On the other hand, some features
are computed in the Processing or Clustering steps. They can be found in 
the `PrecessingFeatures` class.


`SourceFeatures`
----------------
This class provides the following variables.

====================  =============================
Variable name         String
====================  =============================
TIMESTAMP             UTC_TIME
BIASDISCAQNV          IP.NSRCGEN:BIASDISCAQNV
GASAQN                IP.NSRCGEN:GASAQN
GASSASAQN             IP.NSRCGEN:GASSASAQN
SOLINJ_CURRENT        IP.SOLINJ.ACQUISITION:CURRENT
SOLCEN_CURRENT        IP.SOLCEN.ACQUISITION:CURRENT
SOLEXT_CURRENT        IP.SOLEXT.ACQUISITION:CURRENT
OVEN1AQNP             IP.NSRCGEN:OVEN1AQNP
OVEN2AQNP             IP.NSRCGEN:OVEN2AQNP
SOURCEHTAQNI          IP.NSRCGEN:SOURCEHTAQNI
SOURCEHTAQNV          IP.NSRCGEN:SOURCEHTAQNV
SAIREM2_FORWARDPOWER  IP.SAIREM2:FORWARDPOWER
THOMSON_FORWARDPOWER  IP.NSRCGEN:RFTHOMSONAQNFWD
SPARK_COUNTER         IP.NSRCGEN:SPARKS
BCT05_CURRENT         ITL.BCT05:CURRENT
BCT25_CURRENT         ITF.BCT25:CURRENT
BCT41_CURRENT         ITH.BCT41:CURRENT
====================  =============================

The String value is the parameter name in CALS and also the used column name in 
output files and Data Frames.

`ProcessingFeatures`
--------------------
This class provides the following variables.

====================  =====================  =======================================================
Variable name         String                 Description
====================  =====================  =======================================================
SOURCE_RUNNING        source_running         A `1` if the source is running (BCT05 > 0), 0 otherwise
SOURCE_STABILITY      source_stable          A `1` if the source is considered stable, 0 otherwise
CLUSTER               optigrid_cluster       The number of the cluster tha data point belongs to
HT_VOLTAGE_BREAKDOWN  ht_voltage_breakdown   0 if the Point does not belong to a voltae breakdown, 
                                             otherwise the UNIX timestamp of the beginning of the breakdown.
HT_SPARKS_COUNTER     ht_sparks_counter      A counter that increases by one for every spark that was detected
DATAPOINT_DURATION    datapoint_duration     The difference between the timestamp of this and the 
                                             next data point in second
====================  =====================  =======================================================
