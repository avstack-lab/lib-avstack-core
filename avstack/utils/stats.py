# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2021-02-15
# @Last modified by:   spencer
# @Last Modified date: 2021-04-03
# @Description:

import numpy as np
import scipy.stats
from scipy import interpolate


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return m, h  # m-h, m+h


def plot_cdf(data, ax=None, xlabel=None, label=None, title='CDF', grid=True):
    """
    Make a cumulative density function plot out of non-sorted data

    To have multiple CDF lines on the sample plot, create an axis
    and run this command on each set of data

    Arguments:
    - data -- 1x array
    """
    import matplotlib.pyplot as plt
    x_cdf = np.sort(data)
    x_cdf = np.hstack([x_cdf[0], x_cdf])
    y_cdf = np.arange(0, len(data)+1) / len(data)

    if ax is None:
        plt.plot(x_cdf, y_cdf, label=label)
        plt.ylabel('CDF Value')
        if grid:
            plt.grid()
        if xlabel is not None:
            plt.xlabel(xlabel)
        if title is not None:
            plt.title(title)
        plt.show()
    else:
        ax.plot(x_cdf, y_cdf, label=label)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if title is not None:
            ax.set_title(title)
        ax.legend()
        if grid:
            ax.grid(visible=True)


def interp_cdf(data, x_query):
    x_cdf = np.sort(data)
    y_cdf = np.arange(0, len(data)) / len(data)
    f = interpolate.interp1d(x_cdf, y_cdf, assume_sorted=True)
    return f(x_query)


def interp_pdf(data, x_query, bins=100):
    y_pdf, bin_edges = np.histogram(data, bins=bins, density=True)
    bin_centers = [(bin_edges[i] + bin_edges[i+1])/2
                   for i in range(len(bin_edges)-1)]
    f = interpolate.interp1d(bin_centers, y_pdf,
                             assume_sorted=True,
                             fill_value='extrapolate')
    return f(x_query)