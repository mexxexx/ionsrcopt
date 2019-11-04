
#%%
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters
import seaborn as sns
from numba import jit

register_matplotlib_converters()

# matplotlib.use('module://backend_interagg')
matplotlib.use('Qt5Agg')


def main():
    # read data and fill gaps
    data = pd.read_csv("Data_Raw/Nov2018.csv").fillna(method='ffill').sample(100)

    # write everything in a file
    # data.to_csv("Nov2018-reformatted.csv")

    print(data)
    description = data.describe()
    print(description)

    data_labels = list(data)
    print(data_labels)

    # count number lines to remove
    for prop in data_labels:
        if prop == 'ITF.BCT15:CURRENT':
            print(prop + ': ' + str(len(data[data['ITF.BCT15:CURRENT'] < 0].index)))
        elif prop == 'ITF.BCT25:CURRENT':
            print(prop + ': ' + str(len(data[data['ITF.BCT25:CURRENT'] < 0].index)))
        elif prop == 'ITH.BCT41:CURRENT':
            print(prop + ': ' + str(len(data[data['ITH.BCT41:CURRENT'] < 0].index)))
        elif prop == 'ITL.BCT05:CURRENT':
            print(prop + ': ' + str(len(data[data['ITL.BCT05:CURRENT'] < 0].index)))
        elif prop == 'IP.NSRCGEN:OVEN1AQNP':
            print(prop + ': ' + str(len(data[data['IP.NSRCGEN:OVEN1AQNP'] < 4.5].index)))
        elif prop == 'IP.SOLEXT.ACQUISITION:CURRENT':
            print(prop + ': ' + str(len(data[data['IP.SOLEXT.ACQUISITION:CURRENT'] < 1200].index)))
        elif prop == 'IP.NSRCGEN:BIASDISCAQNV':
            print(prop + ': ' + str(len(data[data['IP.NSRCGEN:BIASDISCAQNV'] == 0].index)))
        elif prop == 'IP.SAIREM2:FORWARDPOWER':
            print(prop + ': ' + str(len(data[data['IP.SAIREM2:FORWARDPOWER'] < 500].index)))
        elif prop == 'IP.NSRCGEN:SOURCEHTAQNI':
            print(prop + ': ' + str(len(data[data['IP.NSRCGEN:SOURCEHTAQNI'] > 2.5].index)))
            print(prop + ': ' + str(len(data[data['IP.NSRCGEN:SOURCEHTAQNI'] < 0.5].index)))
        else:
            print(prop + ': nothing to do')

    # clean data
    data.dropna(inplace=True)
    for prop in data_labels:
        if prop == 'ITF.BCT15:CURRENT':
            data.drop(data[data['ITF.BCT15:CURRENT'] < 0].index, inplace=True)
        elif prop == 'ITF.BCT25:CURRENT':
            data.drop(data[data['ITF.BCT25:CURRENT'] < 0].index, inplace=True)
        elif prop == 'ITH.BCT41:CURRENT':
            data.drop(data[data['ITH.BCT41:CURRENT'] < 0].index, inplace=True)
        elif prop == 'ITL.BCT05:CURRENT':
            data.drop(data[data['ITL.BCT05:CURRENT'] < 0].index, inplace=True)
        elif prop == 'IP.NSRCGEN:OVEN1AQNP':
            data.drop(data[data['IP.NSRCGEN:OVEN1AQNP'] < 4.5].index, inplace=True)
        elif prop == 'IP.SOLEXT.ACQUISITION:CURRENT':
            data.drop(data[data['IP.SOLEXT.ACQUISITION:CURRENT'] < 1200].index, inplace=True)
        elif prop == 'IP.NSRCGEN:BIASDISCAQNV':
            data.drop(data[data['IP.NSRCGEN:BIASDISCAQNV'] == 0].index, inplace=True)
        elif prop == 'IP.SAIREM2:FORWARDPOWER':
            data.drop(data[data['IP.SAIREM2:FORWARDPOWER'] < 500].index, inplace=True)
        elif prop == 'IP.NSRCGEN:SOURCEHTAQNI':
            data.drop(data[data['IP.NSRCGEN:SOURCEHTAQNI'] > 2.5].index, inplace=True)
            data.drop(data[data['IP.NSRCGEN:SOURCEHTAQNI'] < 0.5].index, inplace=True)
        else:
            print(prop + ': nothing to do')

    def plotpairs(datas):
        sns.set(style="white")
        g = sns.PairGrid(datas, diag_sharey=False)
        g.map_lower(sns.kdeplot)
        g.map_upper(sns.scatterplot)
        g.map_diag(sns.kdeplot, lw=3)

    plotpairs(data)

main()
# %%
    print("TEST")

    values = data.to_numpy()
    number_of_values = np.shape(values)[0]
    number_of_variables = np.shape(values)[1]

    timestamps = np.array(values[:, 0], dtype='datetime64[ns]')

    x_data = [timestamps[i] for i in range(1, number_of_values)]
    '''for var in range(1,number_of_variables):
        fig_name = f"figures/Timestamp (UTC_TIME)-{data_labels[var]}.png"
        print(f" Save plot as {fig_name}.")
        y_data = [values[i, var] for i in range(1, number_of_values)]
        plt.subplots(figsize=(12, 7))
        plt.scatter(x_data, y_data, s=1)
        plt.xticks(rotation=90)
        plt.ylabel(data_labels[var])
        plt.subplots_adjust(bottom=0.25, top=0.95, right=0.95)
        plt.savefig(fig_name)
        # plt.show(block=False)'''

    nbins = 20

    for var1 in range(1, number_of_variables):
        var1_data = [values[i, var1] for i in range(1, number_of_values)]
        for var2 in range(var1, number_of_variables):
            if var1 == var2:
                fig_name = f"figures/histogram-{data_labels[var1]}.png"
                print(f" Save plot as {fig_name}.")
                plt.subplots(figsize=(12, 7))
                plt.hist(var1_data, bins=30)
                plt.xlabel(data_labels[var1])
                plt.subplots_adjust(bottom=0.25, top=0.95, right=0.95)
                plt.savefig(fig_name)
                plt.close()
            else:
                var2_data = [values[i, var2] for i in range(1, number_of_values)]

                fig_name = f"figures/scatter-{data_labels[var1]}-{data_labels[var2]}.png"
                print(f" Save plot as {fig_name}.")
                plt.subplots(figsize=(12, 7))
                plt.scatter(var1_data, var2_data, s=1)
                plt.xlabel(data_labels[var1])
                plt.ylabel(data_labels[var2])
                plt.subplots_adjust(bottom=0.25, top=0.95, right=0.95)
                plt.savefig(fig_name)
                plt.close()

                fig_name = f"figures/hexbin-{data_labels[var1]}-{data_labels[var2]}.png"
                print(f" Save plot as {fig_name}.")
                plt.subplots(figsize=(12, 7))
                try:
                    plt.hexbin(var1_data, var2_data, bins=nbins, gridsize=(100, 100), cmap=plt.cm.get_cmap('inferno'))
                except ValueError as e:
                    print(e)
                plt.xlabel(data_labels[var1])
                plt.ylabel(data_labels[var2])
                plt.subplots_adjust(bottom=0.25, top=0.95, right=0.95)
                plt.savefig(fig_name)
                plt.close()


if __name__ == "__main__":
    main()
