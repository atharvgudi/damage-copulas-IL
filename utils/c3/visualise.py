import matplotlib.pyplot as plt
import numpy as np

from .c3utils import *


def plotCorrs(thresh, corr, ax=None, **kwargs):
    # ABOUT
    '''
        FUNCTION
        --------
        Plots the observed correlations
            X axis: Thresholds
            Y axis: Correlations
            Form:   Scatter plot


        PARAMETERS
        ----------
        thresh:     The thresholds for the observed correlations
                    REQ: np.ndarray of shape (n,)

        corr:       The observed correlations
                    REQ: np.ndarray of shape (n,)

        ax:         The axes on which to plot
                    REQ: matplotlib.axes.Axes

        **kwargs:   Additional keyword arguments for the scatter plot
                    Assists in customising the plot (e.g., color, label, etc.)


        RETURNS
        -------
        matplotlib.axes.Axes
        ax:     The axes on which the plot was drawn
    '''
    
    if ax is None:
        _, ax = plt.subplots()
    
    ax.scatter(thresh, corr, **kwargs)
    
    return ax

def plotInterval(thresh, lower, upper, ax=None, **kwargs):
    # ABOUT
    '''
        FUNCTION
        --------
        Plots an interval between two curves (e.g., confidence bounds for a strand)
            X axis: Thresholds
            Y axis: Correlation values
            Form:   Shaded region between the lower and upper curves


        PARAMETERS
        ----------
        thresh:     The thresholds corresponding to the lower and upper bounds
                    REQ: np.ndarray of shape (n,)

        lower:      The lower bound values at the given thresholds
                    REQ: np.ndarray of shape (n,)

        upper:      The upper bound values at the given thresholds
                    REQ: np.ndarray of shape (n,)

        ax:         The axes on which to plot
                    REQ: matplotlib.axes.Axes

        **kwargs:   Additional keyword arguments for the fill_between plot
                    Assists in customising the plot (e.g., color, label, opacity, etc.)


        RETURNS
        -------
        matplotlib.axes.Axes
        ax:     The axes on which the plot was drawn
    '''

    if ax is None:
        _, ax = plt.subplots()
    
    ax.fill_between(thresh, lower, upper, **kwargs)
    
    return ax

def plotStrand(strand, ax=None, **kwargs):
    # ABOUT
    '''
        FUNCTION
        --------
        Plots a strand (piecewise linear or polynomial)
            X axis: Thresholds
            Y axis: Correlation values
            Form:   Line plot


        PARAMETERS
        ----------
        strand:     The strand to plot
                    REQ: Strand object

        ax:         The axes on which to plot
                    REQ: matplotlib.axes.Axes

        **kwargs:   Additional keyword arguments for the plot
                    Assists in customising the plot (e.g., color, label, etc.)


        RETURNS
        -------
        matplotlib.axes.Axes
        ax:     The axes on which the plot was drawn
    '''

    if ax is None:
        _, ax = plt.subplots()
    
    if strand.degree < 2:
        ax.plot(strand.thresh, strand.rhos, **kwargs)
    else:
        taus = np.linspace(0., 1., 1000)
        ax.plot(taus, np.polyval(strand.coeffs, taus), **kwargs)
    
    return ax