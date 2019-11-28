import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import contextlib
import io
import sys

import load_data as ld

def main():
    ######################
    ###### SETTINGS ######
    ######################

    clustered_data_folder = 'Data_Clustered/' # Base folder of clustered data 
    filename = 'Nov2018.csv' # The file to load

    source_stability = 1 # 1 if we want to look at a stable source, 0 else
    cluster = 5 # The cluster to plot or None if you want to plot all data

    parameters = ['IP.NSRCGEN:BIASDISCAQNV', 'IP.NSRCGEN:GASSASAQN','IP.SOLCEN.ACQUISITION:CURRENT'] #['IP.NSRCGEN:OVEN1AQNP', 'ITF.BCT25:CURRENT'] # Parameters to be displayed    

    ######################
    ######## CODE ########
    ######################

    # Load file into a data frame
    path = clustered_data_folder + filename
    df = ld.read_data_from_csv(path, None, None)
    with nostdout():
        df = ld.convert_column_types(df)
    df.dropna()    

    if cluster is not None:
        df = df[df['optigrid_cluster'] == cluster].copy()
    df = df.loc[df['source_stable'] == source_stability, ['Timestamp'] + parameters].copy()

    dates = matplotlib.dates.date2num(df['Timestamp'].values)

    fig = plt.figure()
    for i, parameter in enumerate(parameters):
        ax = plt.subplot('{}1{}'.format(len(parameters), i+1))
        ax.set_title("{}".format(parameter))
        ax.tick_params(axis='both', which='major')
        ax.plot_date(dates, df[parameter].values, fmt='.')

    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    fig.suptitle('Time development of cluster {}'.format(cluster))
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.93, wspace=None, hspace=0.4)
    plt.show()

### This is used to supress output to the console
class DummyFile(object):
    def write(self, x): pass

@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = DummyFile()
    yield
    sys.stdout = save_stdout

if __name__ == "__main__":
    main()