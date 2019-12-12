import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

sys.path.insert(1, os.path.abspath('../ionsrcopt'))
import load_data as ld
from source_features import SourceFeatures
from processing_features import ProcessingFeatures

def main():
    ######################
    ###### SETTINGS ######
    ######################

    clustered_data_folder = '../Data_Clustered/' # Base folder of clustered data 
    filename = 'JanNov2018_robust.csv' # The file to load

    ######################
    ######## CODE ########
    ######################

    columns = [SourceFeatures.TIMESTAMP, SourceFeatures.BCT25_CURRENT, ProcessingFeatures.SOURCE_STABILITY] 

    # Load file into a data frame
    path = clustered_data_folder + filename
    df = ld.read_data_from_csv(path, columns, None)
    df = ld.fill_columns(df, None, fill_nan_with_zeros=True)
    df = ld.convert_column_types(df)
    
    dates_stable = matplotlib.dates.date2num(df.loc[df[ProcessingFeatures.SOURCE_STABILITY] == 1].index.values)
    dates_unstable = matplotlib.dates.date2num(df.loc[df[ProcessingFeatures.SOURCE_STABILITY] == 0].index.values)

    fig = plt.figure()
    ax = fig.add_subplot('111')
    ax.set_title("{}".format(filename))
    ax.plot_date(dates_unstable, df.loc[df[ProcessingFeatures.SOURCE_STABILITY] == 0, SourceFeatures.BCT25_CURRENT].values, fmt='.', c='blue')
    ax.plot_date(dates_stable, df.loc[df[ProcessingFeatures.SOURCE_STABILITY] == 1, SourceFeatures.BCT25_CURRENT].values, fmt='.', c='orange')
    ax.set_ylim(-0.01, 0.08)

    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.93, wspace=None, hspace=0.4)
    plt.show()

if __name__ == "__main__":
    main() 