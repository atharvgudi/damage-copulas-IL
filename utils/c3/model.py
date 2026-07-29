from .c3utils import *

import warnings
import numpy as np
import time



# ------------------------------------------------------------------------------
# Indicator Correlation Fitting
# ------------------------------------------------------------------------------



def fitPolyline(obsIndCorrs, thresh, nSim, nLev=100, maxKnots=None):
    # ABOUT
    '''
        FUNCTION
        --------
        Fits a piecewise linear correlation model to the observed indicator correlations of a dataset

        PARAMETERS
        ----------
        obsIndCorrs:    The observed indicator correlations at the given thresholds.

        thresh:         The thresholds at which the indicator correlations are observed
                        NOTE:   These may not be the same thresholds in the resulting strand fitted
                                They only represent the points at which correlations are observed
                                    from where potential change points are selected for the strand
                        REQ: np.ndarray of shape (1, ) with floats in [0, 1]
        
        nSim:           Number of samples to simulate for each ensemble in the fitting process
                        REQ: Positive integer
        
        nLev:           Number of levels to simulate for each ensemble in the fitting process
                        REQ: Positive integer
        
        maxKnots:       Maximum number of knots (incl. endpoints) to use in the piecewise linear model
                        REQ: Positive integer

        RETURNS
        -------
        Strand
        strand: Fitted strand with piecewise linear correlation structure

        np.ndarray(float) <-> len(thresh)
        lower:  The lower bounds of the confidence interval 
                    of Gaussian correlations at the given thresholds

        np.ndarray(float) <-> len(thresh)
        upper:  The upper bounds of the confidence interval 
                    of Gaussian correlations at the given thresholds
    '''

    '''
    If the number of knots is not specified, it is taken as the length of threshold array
    NOTE:   This would mean an overconstrained piecewise linear changing correlation model
            If this happens, consider a different structure to model the correlation
    '''
    
    if maxKnots is None: maxKnots = len(thresh)

    '''
    Starting with the Gaussian case
    '''

    mid = len(thresh) // 2
    threshMid = thresh[mid]
    Rho = gaussCorr(obsIndCorrs[mid], threshMid)

    strand = Strand(
        degree=0, 
        rhos=np.array([Rho, Rho], dtype=float),
        thresh=np.array([0., 1.], dtype=float)
        )
    lower, upper = indCorrbounds(strand, thresh, nSim=nSim, nLev=nLev)

    '''
    If the Gaussian case is satisfied,
        return the correlation, the midpoint of the threshold and bounds
        & call it a day
    '''
    if max(obsIndCorrs[1:-1] - upper[1:-1]) <= 0 and max(lower[1:-1] - obsIndCorrs[1:-1]) <= 0:
        print("Gaussian Fitting succeeded.")
        return strand, indCorrToGauss(lower, thresh), indCorrToGauss(upper, thresh)

    '''
    If not, a multicorrelation model is necessary

    Start with a two correlation model differing at the endpoints
    '''

    rho0 = gaussCorr(obsIndCorrs[0], thresh[0])
    rho1 = gaussCorr(obsIndCorrs[-1], thresh[-1])

    rhos = np.array([rho0, rho1], dtype=float)
    taus = np.array([0., 1.], dtype=float)

    strand = Strand(
        degree=1, 
        rhos=rhos, 
        thresh=taus
        )

    lower, upper = indCorrbounds(strand, thresh, nSim=nSim, nLev=nLev)

    upDiff = obsIndCorrs - upper
    lowDiff = lower - obsIndCorrs
    maxUp = max(upDiff)
    maxLow = max(lowDiff)

    maxIter = 0

    while ((maxUp > 0 or maxLow > 0) and maxIter < maxKnots):
        '''
        Edit rhos and taus
        '''
        worstIndex = np.argmax(upDiff) if maxUp > maxLow else np.argmax(lowDiff)

        newTau = thresh[worstIndex]
        
        if newTau not in taus:
            insertAt = np.searchsorted(taus, newTau)
            taus = np.insert(taus, insertAt, newTau)
            newRho = gaussCorr(obsIndCorrs[worstIndex], newTau)
            rhos = np.insert(rhos, insertAt, newRho)

        strand = Strand(
            degree=-1, 
            rhos=rhos, 
            thresh=taus
            )
        lower, upper = indCorrbounds(strand, thresh, nSim=nSim, nLev=nLev)

        upDiff = obsIndCorrs - upper
        lowDiff = lower - obsIndCorrs
        maxUp = max(upDiff)
        maxLow = max(lowDiff)

        maxIter += 1

    return strand, indCorrToGauss(lower, thresh), indCorrToGauss(upper, thresh)

def fitPolynomial(obsIndCorrs, thresh, nSim, nLev=100, maxDegree=None):
    # ABOUT
    '''
        FUNCTION
        --------
        Fits a polynomially changing correlation model to the observed indicator correlations of a dataset

        PARAMETERS
        ----------
        obsIndCorrs:    The observed indicator correlations at the given thresholds.

        thresh:         The thresholds at which the indicator correlations are observed
                        NOTE:   These may not be the same thresholds in the resulting strand fitted
                                They only represent the points at which correlations are observed
                                    from where potential change points are selected for the strand
                        REQ: np.ndarray of shape (1, ) with floats in [0, 1]
        
        nSim:           Number of samples to simulate for each ensemble in the fitting process
                        REQ: Positive integer
        
        nLev:           Number of levels to simulate for each ensemble in the fitting process
                        REQ: Positive integer

        maxDegree:      Maximum degree that the strand of the changing correlation can be
                            Corresponds to the maximum degree of the polynomial
                        REQ: Positive integer
    
        RETURNS
        -------
        Strand
        strand: Fitted strand with polynomial correlation structure

        np.ndarray(float) <-> len(thresh)
        lower:  The lower bounds of the confidence interval 
                    of Gaussian correlations at the given thresholds

        np.ndarray(float) <-> len(thresh)
        upper:  The upper bounds of the confidence interval 
                    of Gaussian correlations at the given thresholds
    '''

    '''
    If no maximum degree is specified, the length of thresholds is taken as maximum
    NOTE:   This would mean an overconstrained polynomially changing correlation model
            If this happens consider a different structure to model the correlation
    '''
    if maxDegree is None: maxDegree = len(thresh)

    '''
    Starting with the Gaussian case
    '''

    mid = len(thresh) // 2
    threshMid = thresh[mid]

    Rho = gaussCorr(obsIndCorrs[mid], threshMid)
    strand = Strand(
        degree=0, 
        rhos=np.array([Rho, Rho], dtype=float),
        thresh=np.array([0., 1.], dtype=float)
        )
    lower, upper = indCorrbounds(strand, thresh, nSim=nSim, nLev=nLev)

    '''
    If the Gaussian case is satisfied,
        return the correlation, the midpoint of the threshold and bounds
        & call it a day
    '''
    if max(obsIndCorrs[1:-1] - upper[1:-1]) <= 0 and max(lower[1:-1] - obsIndCorrs[1:-1]) <= 0:
        print("Gaussian Fitting succeeded.")
        return strand, indCorrToGauss(lower, thresh), indCorrToGauss(upper, thresh)

    '''
    If not, start with a polynomial model with degree 1
    '''

    rhos = np.array([gaussCorr(indC, th) for indC, th in zip(obsIndCorrs, thresh)])
    indDegree = 2
    maxUp = maxLow = np.inf

    while (maxUp > 0 or maxLow > 0) and indDegree <= maxDegree:

        indCoeffs = polyfitConstrained(thresh, obsIndCorrs, indDegree)
        sampledIndCorrs = np.polyval(indCoeffs, thresh)
        sampledGaussCorrs = np.array([gaussCorr(indC, th) for indC, th in zip(sampledIndCorrs, thresh)])

        gaussDegree = 1

        while (maxUp > 0 or maxLow > 0) and gaussDegree <= indDegree:
            coeffs = np.polyfit(thresh, sampledGaussCorrs, deg=gaussDegree)
            strand = Strand(degree=gaussDegree, coeffs=coeffs)

            lower, upper = indCorrbounds(strand, thresh, nSim=nSim, nLev=nLev)
            upDiff = obsIndCorrs[1:-1] - upper[1:-1]
            lowDiff = lower[1:-1] - obsIndCorrs[1:-1]
            maxUp, maxLow = max(upDiff), max(lowDiff)

            if maxUp > 0 or maxLow > 0: gaussDegree += 1

        if maxUp > 0 or maxLow > 0: indDegree += 1

    if maxUp > 0 or maxLow > 0:
        print("Polynomial fitting did not perfectly converge")

    return strand, indCorrToGauss(lower, thresh), indCorrToGauss(upper, thresh)



# ------------------------------------------------------------------------------
# Data Fitting
# ------------------------------------------------------------------------------



def fitData(data, form='lines', max=5, nThresh=None, nSim=100, nLev=100):
    # ABOUT
    '''
        FUNCTION
        --------
        Fits a strand to the given data

        PARAMETERS
        ----------
        data:       The data to fit the strand
                    REQ: np.ndarray of shape (n, 2)

        form:       Form of changing correlations
                        Either piecewise linear or polynomial
                    REQ: String of 'lines' or 'poly'

        max:        IF form == 'lines', maximum number of knots (incl. endpoints) for the piecewise linear strand
                    IF form == 'poly', maximum degree of the polynomial strand
                    REQ: Positive integer
        
        nThresh:    Number of thresholds at which the strand is fitted (excl. 0 & 1)
                    REQ: Positive integer
        
        nSim:       Number of samples to simulate for each ensemble in the fitting process
                    REQ: Positive integer
        
        nLev:       Number of levels to simulate
                    REQ: Positive integer


        RETURNS
        -------
        Strand
        strand: Fitted strand with the specified form and parameters
    '''

    if nThresh is None:
        thresh = np.linspace(0., 1., 101)
    else:
        thresh = np.linspace(0., 1., nThresh + 2)

    obsIndCorrs = indCorrs(data.values, thresh)
    obsIndCorrs[0], obsIndCorrs[-1] = 1.0, 1.0

    if form == 'lines':
        return fitPolyline(obsIndCorrs, thresh, nSim=1000, nLev=nLev, maxKnots=max)
    elif form == 'poly':
        return fitPolynomial(obsIndCorrs, thresh, nSim=1000, nLev=nLev, maxDegree=max)
    else:
        raise ValueError("Only piecewise linear ('lines') or polynomial ('poly') forms are supported.")



# ------------------------------------------------------------------------------
# Strand Simulation
# ------------------------------------------------------------------------------



def simulate(strand, nSim, nLev=100, pts=None):
    # ABOUT
    '''
        FUNCTION
        --------
        Simulates data from a copula whose structure is defined by the given strand

        PARAMETERS
        ----------
        strand: The strand defining the changing correlation structure of the copula
                REQ: Strand
        
        nSim:   Number of samples to simulate
                REQ: Positive integer

        nLev:   Number of levels to simulate
                REQ: Positive integer

        RETURNS
        -------
        np.ndarray <-> (nSim, 2)
        copSimData: Simulated data with uniform marginals
    '''

    thresh, sqrtCorrs = strand.getInfo()

    copSimData = copSim(sqrtCorrs, thresh, nSim, nLev=nLev, pts=pts)

    return copSimData
