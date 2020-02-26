import pandas as pd
import numpy as np
from scipy import stats

def main():
    filename = 'Results/JanNov2018.csv'
    df = pd.read_csv(filename, index_col=0, header=[0,1])
    parameter = 'gas'

    values = df.loc[df.index>=0, (parameter, '50%')].values

    min_value, max_value = estimate_min_max(values)
    mean, std = estimate_mean_std(values)
    bins = 10
    print("Min: {}".format(min_value))
    print("Max: {}".format(max_value))
    print("Mean: {}".format(mean))
    print("Std: {}".format(std))
    print("DoF is {}".format(bins-2-1))

    hist, edges = np.histogram(values, range=(min_value, max_value), bins=bins)
    print(hist)
    print(edges)
    #hist_expect = calculate_expectations_uniform(edges, len(values))
    hist_expect = calculate_expectations_normal(edges, hist.sum(), mean, std)
    print(hist_expect)

    statistic_chi_squared = chi_squared(hist, hist_expect)
    statistic_g = g_test(hist, hist_expect)
    print("CHI squared statistic is {}".format(statistic_chi_squared))
    print("G statistic is {}".format(statistic_g))


def chi_squared(histogram, histogram_expected):
    result = 0
    for sample, expected in zip(histogram, histogram_expected):
        result += (sample - expected) ** 2 / expected
    return result

def g_test(histogram, histogram_expected):
    result = 0
    for sample, expected in zip(histogram, histogram_expected):
        result += sample * np.log(sample/expected)
    return 2*result


def calculate_expectations_uniform(edges, num_values):
    min_value = edges[0]
    max_value = edges[-1]

    result = [
        stats.uniform.cdf(edges[i+1], loc=min_value, scale=max_value-min_value) - stats.uniform.cdf(edges[i], loc=min_value, scale=max_value-min_value)
        for i in range(len(edges)-1)
    ]

    return np.array(result) * num_values


def calculate_expectations_normal(edges, num_values, mean, std):
    result = [stats.norm.cdf(edges[1], loc=mean, scale=std)]
    result += [
        stats.norm.cdf(edges[i+1], loc=mean, scale=std) - stats.norm.cdf(edges[i], loc=mean, scale=std)
        for i in range(1, len(edges)-2)
    ]
    result += [1 - stats.norm.cdf(edges[-2], loc=mean, scale=std)]

    return np.array(result) * num_values


def estimate_min_max(values):
    return np.quantile(values, (0.05, 0.95))


def estimate_mean_std(values):
    return values.mean(), values.std()

if __name__ == "__main__":
    main()