'''
This file contains all the helper functions required for the project
'''

import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# ------------------------------------------------------------------------------
# Imputation Functions
# ------------------------------------------------------------------------------
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error as mse



def impute(data, k=None):
    # ABOUT
    '''
        FUNCTION
        --------
        Imputes missing values in a DataFrame using K-Nearest Neighbours (KNN)
        imputation. Non-numeric columns are preserved unchanged.

        PARAMETERS
        ----------
        data:   DataFrame to impute
                REQ: pd.DataFrame

        k:      Number of nearest neighbours to use for imputation.
                If None, the optimal k is determined via bestK().
                OPT: Positive integer, default None

        RETURNS
        -------
        pd.DataFrame
        imputedNumeric: DataFrame with missing values imputed
    '''

    numericData = data.select_dtypes(include=np.number)

    if not np.isnan(numericData).any().any(): return data

    copy = numericData.copy()

    if k is None: k = bestK(copy)
    print(f'{k} nearest neighbours will be used for imputation.')

    imputedNumeric = pd.DataFrame(
        KNNImputer(n_neighbors=k).fit_transform(copy),
        columns=copy.columns
    )

    for col in data.columns:
        if col not in imputedNumeric.columns:
            imputedNumeric[col] = data[col]

    return imputedNumeric

def bestK(data, minK=1, maxK=None, nSplits=10):
    # ABOUT
    '''
        FUNCTION
        --------
        Determines the optimal number of nearest neighbours for KNN imputation
        using 10-(pseudo)fold cross validation. Missingness is simulated at the
        observed missing rate and imputation error is measured via MSE.

        PARAMETERS
        ----------
        data:       DataFrame with missing values to optimise k for
                    REQ: pd.DataFrame

        minK:       Minimum number of nearest neighbours to consider
                    OPT: Positive integer, default 1

        maxK:       Maximum number of nearest neighbours to consider.
                    If None, defaults to sqrt(#samples).
                    OPT: Positive integer, default None

        nSplits:    Number of pseudofolds for cross validation
                    OPT: Positive integer, default 10

        RETURNS
        -------
        int
        optimal k value minimising average MSE across pseudofolds
    '''

    # Cross-validation setup
    if maxK is None: maxK = int(np.sqrt(data.shape[0])) + 1
    kValues = range(minK, maxK + 1)
    avgErrors = []

    # Precompute - these don't change between folds or k-values
    nullMask = np.isnan(data)
    noNull = np.argwhere(~nullMask)
    nullRatio = np.sum(nullMask.values) / nullMask.size

    for k in kValues:
        foldErrors = []
        for _ in range(nSplits):

            # SETUP DATA COPY
            simData = data.copy()

            # CREATE IMPUTER
            imputer = KNNImputer(n_neighbors=k)

            # SIMULATE MISSINGNESS
            flip = np.random.choice(len(noNull),
                                    size=int(len(noNull) * nullRatio) + 1,
                                    replace=False)

            simNull = nullMask.copy()
            simNull[:] = False
            for idx in flip:
                simNull.iat[noNull[idx][0], noNull[idx][1]] = True

            simData[simNull] = np.nan

            # IMPUTE
            imputedSimData = imputer.fit_transform(simData)

            # RECORD ERROR FOR PSEUDOFOLD
            foldErrors.append(mse(data.values[simNull.values],
                                    imputedSimData[simNull.values]))

        # RECORD ERROR FOR k
        avgErrors.append(np.mean(foldErrors))

    # Find the 'k' with the lowest average error
    return kValues[np.argmin(avgErrors)]



# ------------------------------------------------------------------------------
# Data <-> Uniform Transformations
# ------------------------------------------------------------------------------
from sklearn.preprocessing import QuantileTransformer



def PIT(obs, data):
    # ABOUT
    '''
        FUNCTION
        --------
        Applies the Probability Integral Transform (PIT) to data 
            using a quantile transformer fitted on the observed data, 
            mapping values to uniform marginals in [0,1]^d

        PARAMETERS
        ----------
        obs:    Observed data to fit the quantile transformer on
                REQ: pd.DataFrame <-> (n, d)

        data:   Data to apply the PIT to
                REQ: pd.DataFrame <-> (m, d)

        RETURNS
        -------
        pd.DataFrame <-> (m, d)
        df:     Data with uniform marginals in [0,1]^d
    '''

    qt = QuantileTransformer(output_distribution='uniform')
    qt.fit(obs)

    return pd.DataFrame(qt.transform(data), columns=obs.columns) # = df

def rescale(obs, sim):
    # ABOUT
    '''
        FUNCTION
        --------
        Rescales simulated data from the uniform hypercube [0, 1]^d 
            back to the original scale of the observed data.
            Inverse of the PIT

        PARAMETERS
        ----------
        obs:    Observed data to fit the quantile transformer on
                REQ: pd.DataFrame <-> (n, d)

        sim:    Simulated data in [0,1]^d to rescale
                REQ: pd.DataFrame <-> (m, d)

        RETURNS
        -------
        pd.DataFrame <-> (m, d)
        df:     Simulated data rescaled to the original scale of the observed data
    '''
    qt = QuantileTransformer(output_distribution='uniform')
    qt.fit(obs)

    return pd.DataFrame(qt.inverse_transform(sim), columns=obs.columns) # = df



# ------------------------------------------------------------------------------
# Copula Functions
# ------------------------------------------------------------------------------
from pyvinecopulib import Bicop, BicopFamily
import openturns as ot

from scipy.stats import t, multivariate_t



def biCopula(data):
    # ABOUT
    '''
        FUNCTION
        --------
        Fits a bivariate copula to the data 
            by selecting the best fitting family
            Bicop automatically does this using AIC/BIC

        PARAMETERS
        ----------
        data:   Observed data to fit the copula to
                REQ: pd.DataFrame <-> (n, 2)

        RETURNS
        -------
        pyvinecopulib.Bicop
        cop:    Fitted bivariate copula with the optimal family selected
    '''

    cop = Bicop()
    cop.select(data.rank(pct=True).values)

    return cop

def gaussCopula(data):
    # ABOUT
    '''
        FUNCTION
        --------
        Fits a Gaussian copula to the data 
            regardless of whether it is the optimal family

        PARAMETERS
        ----------
        data:   Observed data to fit the copula to
                REQ: pd.DataFrame <-> (n, 2)

        RETURNS
        -------
        pyvinecopulib.Bicop
        cop:    Fitted bivariate Gaussiancopula
    '''

    cop = Bicop(BicopFamily.gaussian)
    cop.fit(data.rank(pct=True).values)

    return cop

def tCopula(data):
    # ABOUT
    '''
        FUNCTION
        --------
        Fits a Student t-copula to the data 
            regardless of whether it is the optimal family

        PARAMETERS
        ----------
        data:   Observed data to fit the copula to
                REQ: pd.DataFrame <-> (n, 2)

        RETURNS
        -------
        pyvinecopulib.Bicop
        cop:    Fitted bivariate Student t-copula
    '''

    cop = Bicop(BicopFamily.student)
    cop.fit(data.rank(pct=True).values)

    return cop

def displayCopula(cop):
    # ABOUT
    '''
        FUNCTION
        --------
        Displays a summary of a fitted bivariate copula, 
            including its family and parameters, 
            a plot of the copula density with uniform marginals, 
            and its AIC and BIC values.

        PARAMETERS
        ----------
        cop:    Fitted bivariate copula to display
                REQ: pyvinecopulib.Bicop

        RETURNS
        -------
        None
    '''

    print(cop)
    cop.plot(margin_type="unif")
    print(f"Akaike Information Criterion (AIC): {cop.aic()}")
    print(f"Bayesian Information Criterion (BIC): {cop.bic()}")

def otTCopula(tCop):
    # ABOUT
    '''
        FUNCTION
        --------
        Converts a pyvinecopulib Student t-copula 
            to an OpenTURNS StudentCopula object

        PARAMETERS
        ----------
        tCop:   Fitted bivariate Student t-copula to convert
                REQ: pyvinecopulib.Bicop with BicopFamily.student

        RETURNS
        -------
        ot.StudentCopula
        otT:    OpenTURNS Student t-copula 
                    with the same correlation and degrees of freedom as the input
    '''

    rho, df = tCop.parameters
    rho, df = rho[0], df[0]
    R = ot.CorrelationMatrix([[1.0, rho], [rho, 1.0]])
    return ot.StudentCopula(df, R) # = otT

def tCopulaCDF(magProb, dmgProb, rho, df):
    # ABOUT
    '''
        FUNCTION
        --------
        Evaluates the CDF of a bivariate Student t-copula at a given point 
            This is done by transforming uniform marginals to t-quantiles 
                and evaluating the bivariate t-CDF.

        PARAMETERS
        ----------
        magProb:    Marginal probability of the magnitude variable
                    REQ: float in [0, 1]

        dmgProb:    Marginal probability of the damage variable
                    REQ: float in [0, 1]

        rho:        Correlation parameter of the t-copula
                    REQ: float in [-1, 1]

        df:         Degrees of freedom of the t-copula
                    REQ: Positive float

        RETURNS
        -------
        float
        cdf:    CDF value of the t-copula at (magProb, dmgProb) in [0, 1]
    '''

    corr = [[1.0, rho], [rho, 1.0]]
    u = [magProb, dmgProb]
    return multivariate_t.cdf(t.ppf(u, df), loc=[0, 0], shape=corr, df=df) # = cdf



# ------------------------------------------------------------------------------
# Goodness-of-Fit
# ------------------------------------------------------------------------------
from utils.KS2D import *
import ndtest



def ksTests(obs, sim, simRescaled):
    # ABOUT
    '''
        FUNCTION
        --------
        Runs two 2D Kolmogorov-Smirnov tests (Gabinou, 2018 and Li, 2019)
            comparing observed data against simulated copula data.

        H0: The observed data and the simulated data come from the same distribution

        HA: The observed data and the simulated data do not come from the same distribution

        PARAMETERS
        ----------
        obs:            Observed data with predictor as the first column
                            and response as the second column
                        REQ: pd.DataFrame <-> (n, 2)

        sim:            Simulated data in [0,1]^2 from the fitted copula
                        REQ: pd.DataFrame <-> (n, 2)

        simRescaled:    Simulated data rescaled to the original scale of obs
                        REQ: pd.DataFrame <-> (n, 2)

        RETURNS
        -------
        pd.DataFrame <-> (2, 2)
        gabinou:    KS test statistics and p-values (Gabinou 2018)
                    for raw and rescaled simulations

        pd.DataFrame <-> (2, 2)
        li:         KS test statistics and p-values (Li 2019)
                    for raw and rescaled simulations
    '''

    ranks = obs.rank(pct=True)

    # Gabinou (2018)
    d1, prob1 = ks2d2s(ranks.values, sim.values)
    d2, prob2 = ks2d2s(obs.values, simRescaled.values)
    gabinou = pd.DataFrame(
        {'Statistic': [d1, d2], 'p-Value': [prob1, prob2]},
        index=['Raw Simulation', 'Rescaled Simulation']
    )

    # Li (2019)
    prob3, d3 = ndtest.ks2d2s(ranks.values[:,0], ranks.values[:,1],
                               sim.values[:,0], sim.values[:,1], extra=True)
    prob4, d4 = ndtest.ks2d2s(obs.values[:,0], obs.values[:,1],
                               simRescaled.values[:,0], simRescaled.values[:,1], extra=True)
    li = pd.DataFrame(
        {'Statistic': [d3, d4], 'p-Value': [prob3, prob4]},
        index=['Raw Simulation', 'Rescaled Simulation']
    )

    return gabinou, li

def ksRejectionRate(obs, simFunc, nRuns=100, alpha=0.05):
    # ABOUT
    '''
        FUNCTION
        --------
        Estimates the rejection rate of two 2D Kolmogorov-Smirnov tests (Gabinou 2018 and Zhaozhou Li 2019)
            The tests are run nRuns times on simulated data generated by simFunc(obs),
                and the proportion of rejections at significance level alpha is returned for both tests.

        H0: The observed data and the simulated data come from the same distribution

        HA: The observed data and the simulated data do not come from the same distribution

        PARAMETERS
        ----------
        obs:        Observed data with predictor as the first column
                        and response as the second column
                    REQ: pd.DataFrame <-> (n, 2)

        simFunc:    Function taking obs and returning (sim, simRescaled),
                        where sim is simulated data in [0,1]^2
                        and simRescaled is the same data rescaled to the original scale of obs.
                    Called once per run to generate fresh simulations.
                    REQ: Callable(pd.DataFrame) -> (pd.DataFrame, pd.DataFrame)

        nRuns:      Number of times to run the 2D KS tests
                    REQ: Positive integer

        alpha:      Significance level for rejection
                    REQ: float in (0, 1)

        RETURNS
        -------
        pd.DataFrame <-> (2, 2)
        rr:     Rejection rates for Gabinou (2018) and Zhaozhou Li (2019)
                    across raw and rescaled simulations
    '''

    gabinouRejects = np.zeros(2)
    liRejects = np.zeros(2)

    for _ in range(nRuns):
        sim, simRescaled = simFunc(obs)
        gabinou, li = ksTests(obs, sim, simRescaled)
        gabinouRejects += (gabinou['p-Value'] < alpha).values
        liRejects += (li['p-Value'] < alpha).values

    gabinouRates = pd.DataFrame(
        {'Rejection Rate': gabinouRejects / nRuns},
        index=['Raw Simulation', 'Rescaled Simulation']
    )
    liRates = pd.DataFrame(
        {'Rejection Rate': liRejects / nRuns},
        index=['Raw Simulation', 'Rescaled Simulation']
    )

    return pd.concat(
        [gabinouRates, liRates],
        keys=['Gabinou (2018)', 'Zhaozhou Li (2019)'],
        axis=1
    ) # = rr



# ------------------------------------------------------------------------------
# Application-Specific Functions
# ------------------------------------------------------------------------------



def confInt(data, x, prob=0.95, rho=None, df=None):
    # ABOUT
    '''
        FUNCTION
        --------
        Computes a two-tailed confidence interval for the response (y)
            conditioned on a given value of the predictor (x)
        
        If no parameters are provided, a fitted Student t-copula is used

        PARAMETERS
        ----------
        data:   Observed data in the form (predictor, response) to fit the copula
                REQ: pd.DataFrame <-> (n, 2)

        x:      Value of the predictor as the condition
                REQ: float

        prob:   Confidence level
                REQ: float in (0, 1)

        rho:    Correlation parameter of the t-copula to use for the confidence interval.
                    If None, the correlation from a fitted t-copula is used.
                REQ: float in [-1, 1] or None

        df:     Degrees of freedom of the t-copula to use for the confidence interval.
                    If None, the degrees of freedom from a fitted t-copula is used.
                REQ: Positive float or None

        RETURNS
        -------
        float
        yLow:   Lower bound of the confidence interval in the original scale

        float
        yUp:    Upper bound of the confidence interval in the original scale
    '''

    xCol = data.columns[0]
    yCol = data.columns[1]

    if rho is None or df is None:
        cop = tCopula(data)
        params = cop.parameters
        if rho is None: rho = params[0][0]
        if df is None: df = params[1][0]

    R = ot.CorrelationMatrix([[1.0, rho], [rho, 1.0]])
    tcopula = ot.StudentCopula(df, R)

    end = (1 - prob) / 2
    xProb = PIT(data[[xCol]], pd.DataFrame([[x]], columns=[xCol])).values[0][0]

    lowProb = tcopula.computeConditionalQuantile(end, ot.Point([xProb]))
    yLow = rescale(data[[yCol]], pd.DataFrame([[lowProb]], columns=[yCol])).values[0][0]

    upProb = tcopula.computeConditionalQuantile(1 - end, ot.Point([xProb]))
    yUp = rescale(data[[yCol]], pd.DataFrame([[upProb]], columns=[yCol])).values[0][0]

    return yLow, yUp

def cumCondProb(data, yCond, xMin=None, xMax=None, df=None, rho=None):
    # ABOUT
    '''
        FUNCTION
        --------
        Computes the conditional cumulative probability P(Y <= y | xMin <= X <= xMax)
            using a fitted Student t-copula
            At least one of xMin or xMax must be provided

        PARAMETERS
        ----------
        data:   Observed data with X as the first column and Y as the second column
                REQ: pd.DataFrame <-> (n, 2)

        yCond:  Value of Y to evaluate the conditional CDF at
                REQ: float

        xMin:   Lower bound of the X conditioning interval.
                    If None, conditions on X <= xMax only.
                REQ: float

        xMax:   Upper bound of the X conditioning interval.
                    If None, conditions on X >= xMin only.
                REQ: float
        
        df:     Fixed degrees of freedom of the t-copula
                    If None, t-copula will be fitted using Bicop 
                        and respective parameter is used
                REQ: float

        rho:    Fixed correlation of the t-copula
                    If None, t-copula will be fitted using Bicop 
                        and respective parameter is used
                REQ: float in (-1, 1)

        RETURNS
        -------
        float
        prob:   Conditional cumulative probability P(Y <= y | xMin <= X <= xMax) in [0, 1]
    '''

    if xMin is None and xMax is None:
        raise ValueError("At least one of xMin or xMax must be provided.")

    cop = tCopula(data)
    rho = cop.parameters[0][0]
    if df is None: df = cop.parameters[1][0]

    xCol = data.columns[0]
    yCol = data.columns[1]

    yProb = PIT(data[[yCol]], pd.DataFrame([[yCond]], columns=[yCol])).values[0][0]

    if xMax is None:
        xMinProb = PIT(data[[xCol]], pd.DataFrame([[xMin]], columns=[xCol])).values[0][0]
        copCDF = tCopulaCDF(xMinProb, yProb, rho, df)
        return (yProb - copCDF) / (1 - xMinProb)

    if xMin is None:
        xMaxProb = PIT(data[[xCol]], pd.DataFrame([[xMax]], columns=[xCol])).values[0][0]
        copCDF = tCopulaCDF(xMaxProb, yProb, rho, df)
        return copCDF / xMaxProb

    xMinProb = PIT(data[[xCol]], pd.DataFrame([[xMin]], columns=[xCol])).values[0][0]
    xMaxProb = PIT(data[[xCol]], pd.DataFrame([[xMax]], columns=[xCol])).values[0][0]

    maxCopCDF = tCopulaCDF(xMaxProb, yProb, rho, df)
    minCopCDF = tCopulaCDF(xMinProb, yProb, rho, df)

    return (maxCopCDF - minCopCDF) / (xMaxProb - xMinProb) # = prob

