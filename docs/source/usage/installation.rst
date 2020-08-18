Installation
============

No installation is required. Just clone this package anywhere onto your disk and you are ready to start.

::

    git clone <URL>

However, this package has several dependencies that have to be fulfilled. It is recommended to use a dedicated ``anaconda`` environment with ``python3``. You create it using

::

    conda create -n ionsrcopt python=3

and whenever you want to use it you have to activate it with

::

    conda activate ionsrcopt

The dependencies you have to install (with ``conda install`` or with ``pip install`` if the package cannot be installed with conda) into this environment, i.e. you have to activate it first, are

* numpy
* pandas
* scipy
* matplotlib
* statsmodels
* jupyter (if you want to use any of the jupyter notebooks)
* sphinx and sphinx_rtd_theme (for building the documentation)

Setup
=====

We will need various folders so it is best to create them now.

Under the root create three folders called `Data_Raw`, `Data_Preprocessed` and `Data_Clustered`. Then, in the `visualization` directory create a folder `Results`. 
To avoid later troubles, please use the same capitalization. The first three folders will store the data during the steps of our analysis, and in the last folder the summarized cluster results will be saved.

To build the documentation as `html` files, run ``make html`` from inside the `docs` directory. Please make sure you have ``shpinx`` and ``sphinx_rtd_theme`` installed.