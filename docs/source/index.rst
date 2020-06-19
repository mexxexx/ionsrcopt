Welcome to ionsrcopt's documentation!
=====================================

What is this package?
---------------------

This package was developed in the CERN BE-ABP-HSL department
to study the operation of the Linac3 GTS ECR ion source.

The main goal is to understand, whether there were certain settings that
were used for long periods of time and lead to a stable source. For
this a cluster analysis was performed to identify groups of similar settings
and their duration and influence on different source properties such
as stability and HT voltage breakdown rate. 

This package provides all tools that were used for this analysis so that
it can also be carried out for future runs.

This documentation is written to be read side by side with the code to facilitate
the understanding, because most scripts need the user to set parameters in the
code and don't support a command line only usage. Everything was used on Linux/Ubuntu,
however it should run on Windows and Mac too.

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   usage/installation
   usage/overview
   ionsrcopt/overview
   preparation/overview
   clustering/overview
   visualization/overview


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
