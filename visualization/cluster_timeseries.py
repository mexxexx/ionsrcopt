import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import contextlib
import io
import sys
import os

import argparse

sys.path.insert(1, os.path.abspath('../ionsrcopt'))
import load_data as ld
from source_features import SourceFeatures
from processing_features import ProcessingFeatures

def main():
    ######################
    ###### SETTINGS ######
    ######################

    clustered_data_folder = '../Data_Clustered/' # Base folder of clustered data 
    filename = 'JanNov2018_lowbandwidth.csv' # The file to load
    
    features = [
        SourceFeatures.BIASDISCAQNV, 
        SourceFeatures.GASAQN, 
        SourceFeatures.OVEN1AQNP,
        SourceFeatures.SAIREM2_FORWARDPOWER,
        SourceFeatures.SOLINJ_CURRENT,
        SourceFeatures.SOLCEN_CURRENT,
        SourceFeatures.SOLEXT_CURRENT,
        SourceFeatures.SOURCEHTAQNI,
        SourceFeatures.BCT25_CURRENT] # Features to be displayed 

    args = parse_args()
    source_stability = args['source_stability']
    cluster = args['cluster']
    show_breakdows = args['show_breakdows']

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
        df = df[(df[ProcessingFeatures.CLUSTER] == cluster)].copy()
    df = df.loc[df[ProcessingFeatures.SOURCE_STABILITY] == source_stability].copy()

    dates_nobreakdown = matplotlib.dates.date2num(df[df[ProcessingFeatures.HT_VOLTAGE_BREAKDOWN] == 0].index.values)
    dates_breakdown = matplotlib.dates.date2num(df[df[ProcessingFeatures.HT_VOLTAGE_BREAKDOWN] > 0].index.values)

    fig, ax = plt.subplots(len(features), 1, sharex=True)
    for i, parameter in enumerate(features):
        ax[i].set_title("{}".format(parameter))
        ax[i].tick_params(axis='both', which='major')
        if show_breakdows:
            ax[i].plot_date(dates_breakdown, df.loc[df[ProcessingFeatures.HT_VOLTAGE_BREAKDOWN] > 0, parameter].values, linestyle='', marker='.', markersize=1, color='#ff7f0e')
        ax[i].plot_date(dates_nobreakdown, df.loc[df[ProcessingFeatures.HT_VOLTAGE_BREAKDOWN] == 0, parameter].values, linestyle='', marker='.', markersize=1, color='#1f77b4')
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
    parser.add_argument('-b', '--show_breakdows', default=False, type=bool, help='True or False (default) if you want to display the breakdown points (in a different color)')

    args = parser.parse_args()

    return {'source_stability' : args.source_stability, 
            'cluster' : args.cluster,
            'show_breakdows' : args.show_breakdows}

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