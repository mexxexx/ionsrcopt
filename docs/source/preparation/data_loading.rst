Downloading Data from CALS
==========================
The entry point of the analysis is to download the data from
CALS. It is possible to do this manually using Timber, but as
this is cumbersome, you can use the ``Data_Loading.ipynb``
notebook that uses `pyTimber` (see `<https://github.com/rdemaria/pytimber>`_). Note however, that you need to
be on the CERN technical network for this. The recommended approach is
to upload this notebook to ``SWAN`` (see `here <https://swan.web.cern.ch/>`_), 
because pyTimber can be used there. 

Before you can use the notebook, you have to configure it. For this
the two last cells are important. 

In the second to last cell you have to specify which parameters you want
to download. You can either load them as they are in the database (by adding
them to the ``parameters_raw`` list), or transformed (``parameters_scaled``). For the 
transformations you need to add them in the form ``parameter:dict``, where ``dict`` is a
dictionary with the following entries:

* ``scale``: The way you want to transform the data, e.g. average it.
* ``interval``: The time interval to use, e.g. seconds, minutes.
* ``size``: How long the interval should be, e.g. 10. (this has to be an integer)

Please see the pyTimber documentation for a list of available options.

Afterwards, you can configure what time span to download. In the last cell you can set the year, 
the start and the end month. The script always downloads data in monthly batches. There are two more
options:

* ``replace_file``: If you set this to ``True``, then any existing file for the same month (``<month><year>.csv``) will be overwritten.
* ``replace_column``: If you don't replace an existing file, then new columns will be appended to it. This setting regulates what happens with already existing columns that were downloaded again. If it is set to ``True``, then they will be overwritten, otherwise the original will be kept and newly downloaded data discarded.

Once configuration is finished, run the whole notebook. Data will be saved in the 
``output_folder``, and each month will be a separate .csv file with the name ``<month><year>.csv``.
The column names will be the same as specified in the parameter lists.