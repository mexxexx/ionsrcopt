import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import contextlib
import io
import sys

import argparse

sys.path.insert(1, '../ionsrcopt')
import load_data as ld

def main():
    ######################
    ###### SETTINGS ######
    ######################

    clustered_data_folder = 'Data_Clustered/' # Base folder of clustered data 
    filename = 'JanNov2018_lowbandwidth.csv' # The file to load

    args = parse_args()
    source_stability = args['source_stability']
    cluster = args['cluster']

    parameters = ['IP.NSRCGEN:BIASDISCAQNV', 'IP.NSRCGEN:GASSASAQN', 'IP.SOLCEN.ACQUISITION:CURRENT', 'IP.SOLEXT.ACQUISITION:CURRENT','IP.NSRCGEN:OVEN1AQNP', 'ITF.BCT25:CURRENT'] # Parameters to be displayed    

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
        df = df[(df['optigrid_cluster'] == cluster)  & (df['is_breakdown'] == 0)].copy()
    df = df.loc[df['source_stable'] == source_stability, parameters].copy()

    dates = matplotlib.dates.date2num(df.index.values)

    fig, ax = plt.subplots(len(parameters), 1, sharex=True)
    for i, parameter in enumerate(parameters):
        ax[i].set_title("{}".format(parameter))
        ax[i].tick_params(axis='both', which='major')
        ax[i].plot_date(dates, df[parameter].values, linestyle='', marker='.', markersize=1)
        ax[i].grid(True)

    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    fig.suptitle('Time development of cluster {}'.format(cluster))
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.93, wspace=None, hspace=0.4)
    plt.show()

def parse_args():
    parser = argparse.ArgumentParser(description='View time development of clusters')
    parser.add_argument('-s', '--source_stability', default=1, type=int, help='1 if you want to look at the stable source, 0 else')
    parser.add_argument('-c', '--cluster', default=None, type=int, help='The cluster you want to look at, or None for all data')

    args = parser.parse_args()

    return {'source_stability' : args.source_stability, 
            'cluster' : args.cluster}

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