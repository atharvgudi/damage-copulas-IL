from .strand import Strand

import numpy as np
from scipy.stats import rankdata, norm, multivariate_normal
import scipy.stats as stats
from scipy.optimize import brentq


# GLOBALS
T_MIN = 1e-5



# ------------------------------------------------------------------------------
# MATHEMATICAL FUNCTIONS
# ------------------------------------------------------------------------------



def indCorr(rho, t):
    # ABOUT
    '''
        FUNCTION
        --------
        Computes the indicator correlation for a given Gaussian correlation and threshold quantile


        PARAMETERS    
        ----------
        rho:    Gaussian correlation between the two variables
                REQ: Value in [-1, 1]

        t:      Threshold quantile of the indicator correlation
                REQ: Value in (0, 1)


        RETURNS
        -------
        float
        indCorr:    Indicator correlation corresponding to the given Gaussian correlation and threshold quantile
    '''

    z = norm.ppf(t)
    mean = [0.0, 0.0]
    cov  = [[1.0, rho], [rho, 1.0]]

    phi = multivariate_normal.cdf(np.column_stack([z, z]), mean=mean, cov=cov)

    return (phi - t**2) / (t * (1 - t))

def gaussCorr(indR, t, *, tol=1e-5):
    # ABOUT
    '''
        FUNCTION
        --------
        Computes the Gaussian correlation corresponding to a given indicator correlation and threshold quantile


        PARAMETERS
        ----------
        indR:   Indicator correlation between the two variables
                REQ: Value in [-1, 1]

        t:      Threshold quantile of the indicator correlation
                REQ: Value in (0, 1)


        RETURNS
        -------
        float
        gaussCorr: Gaussian correlation corresponding to the given indicator correlation and threshold quantile
    '''

    if t < T_MIN or t > 1 - T_MIN: t = T_MIN
    if indR == 1: return 1
    if indR == -1: return 1 / (1 - (1 / min(t, 1 - t)))
    if np.isclose(t, 0.5): return np.sin(0.5 * np.pi * indR)

    def f(rho): return indCorr(rho, t) - indR

    # Clamp indR to the achievable range of the Gaussian copula at this threshold
    lo = indCorr(-1 + T_MIN, t)
    hi = indCorr(1 - T_MIN, t)
    indR = np.clip(indR, lo, hi)

    return brentq(f, -1 + T_MIN, 1 - T_MIN, xtol=tol)

def sqrt(corr):
    # ABOUT
    '''
        FUNCTION
        --------    
        Computes the square root of a correlation matrix using Schur's decomposition
            (https://en.wikipedia.org/wiki/Square_root_of_a_matrix)


        PARAMETERS
        ----------
        corr:   Correlation matrix whose square root is to be computed
                REQ: 2-dimensional non-empty numpy.ndarray
        

        RETURNS
        -------
        np.ndarray(float)
        sqrtCorr:  Square root of the correlation matrix
                        Shape: Same as corr
    '''

    lambdas, vecs = np.linalg.eigh(corr)
    lambdas[lambdas < 0] = 0

    return vecs @ np.diag(np.sqrt(lambdas)) @ vecs.T # = sqrtCorr

    '''
    NOTE:
    sqrtCorr need not be of the form np.array([[1, sqRho], [sqRho, 1]])
        This is expected to ensure sqrtCorr @ sqrtCorr.T = corr
    '''

def polyfitConstrained(x, y, degree):
    # ABOUT
    '''
        FUNCTION
        --------    
        Fits the dataset x, y to a polynomial p(x) of given degree, enforcing p(0) = p(1) = 1
            using the least squares method

        By factor theorem,
            p(x) = 1 + x(x-1)q(x)
                where q(x) is a polynomial 2 degrees lesser than p(x)

        PARAMETERS
        ----------
        x:      Independent variable values (Predictor)
                REQ: 1-dimensional non-empty numpy.ndarray

        y:      Dependent variable values (Response)
                REQ: 1-dimensional non-empty numpy.ndarray

        degree: Degree of the polynomial p(x) to fit
                REQ: Positive integer > 1
        

        RETURNS
        -------
        np.ndarray(float)
        p.coeffs:  Coefficients of the fitted polynomial p(x)
                        Shape: (degree + 1,)

    '''
    
    qDeg = degree - 2

    '''
    Vandermonde matrix construction for polynomial q(x)
        basis = [ x^qDeg, x^(qDeg-1), ..., x^1, x^0 ]
                            <- qDeg - 1 ->
    '''
    basis = np.vander(x, qDeg + 1)

    '''
    Weighting makes the polynomial x(x-1)q(x)
    '''
    weighted = basis * (x * (x - 1))[:, None]

    '''
    Least squares is used to solve
        p(x) - 1 = x(x-1)q(x)
            <=> p(x) = 1 + x(x-1)q(x)
    '''
    qCoeffs, *_ = np.linalg.lstsq(weighted, y - 1, rcond=None)
    q = np.poly1d(qCoeffs)

    '''
    Converting from q -> p
        [1, -1, 0]*q <=> (1*x^2 - 1*x + 0)*q(x) = x(x-1)q(x)
        np.poly1d([1]) <=> 1

        Hence p(x) = x(x-1)q(x) + 1
                <=> p <- [1, -1, 0]*q + [1]
    '''
    p = np.poly1d([1, -1, 0])*q + np.poly1d([1])

    return p.coeffs



# ----------------------------------------------------------------------------
# INDICATOR CORRELATION FUNCTIONS
# ----------------------------------------------------------------------------



def indCorrs(data, thresh):
    # ABOUT
    '''
        FUNCTION
        --------
        Measures indicator correlations between two variables
        

        PARAMETERS
        ----------
        data:   Data containing the two variables
                REQ: 2-dimensional non-empty numpy.ndarray

        thresh: List of indicator thresholds when measuring the change in correlation
                REQ: Non-empty list of values in [0, 1]


        RETURNS
        -------
        numpy.ndarray(float) <-> len(thresh)
        rankCorr:   Indicator correlation values between the variables corresponding to thresholds
    '''

    n = data.shape[0]
    x, y = data[:, 0], data[:, 1]

    err = 1 + np.random.random(n) * 1e-2

    xProb = rankdata(x * err, method='min') / (n + 1)
    yProb = rankdata(y * err, method='min') / (n + 1)

    rankCorr = []

    for th in thresh:

        xLow = (xProb <= th).astype(float)
        yLow = (yProb <= th).astype(float)
        pxLow = np.mean(xLow)
        pyLow = np.mean(yLow)

        if (pxLow == 0 and pyLow == 0) or (pxLow == 1 and pyLow == 1):
            indCorr = 1
        elif (pxLow == 0 and pyLow == 1) or (pxLow == 1 and pyLow == 0):
            indCorr = -1
        else:
            indCorr = np.corrcoef(xLow, yLow)[0, 1]

        rankCorr += [indCorr]

    return np.array(rankCorr)

def indCorrToGauss(bounds, thresh):
    # ABOUT
    '''
        FUNCTION
        --------
        Converts interval bounds of indicator correlations to bounds of Gaussian correlations 
            using the gaussCorr function
        

        PARAMETERS
        ----------
        bounds: Bounds of indicator correlations to be converted
                REQ: Non-empty np.ndarray(float)

        thresh: List of indicator thresholds at which copula values are to be measured
                REQ: Non-empty list of length len(bounds) of values in [0, 1]


        RETURNS
        -------
        np.ndarray(float) <-> len(thresh)
        gaussBound: Bound of Gaussian correlations corresponding to the given bounds 
                        of indicator correlations and thresholds
    '''

    return np.array([gaussCorr(b, t) for b, t in zip(bounds, thresh)]) # gaussBound

def gaussianBand(rho, thresh, nSim, nEns=1000, alpha=0.90):
    # ABOUT
    '''
        FUNCTION
        --------
        Creates a confidence interval of simulated Gaussian copula values of a given correlation
            These are referred to in literature as "Gaussian bands"
        

        PARAMETERS
        ----------
        rho:    Correlation of the Gaussian copula to be simulated
                REQ: Float in (0, 1)

        thresh: List of indicator thresholds at which copula values are to be measured
                REQ: Non-empty list of values in [0, 1]
        
        nSim:   No. of simulated copula datapoints per ensemble
                REQ: Positive integer
        
        nEns:   No. of ensembles to determine the confidence intervals
                REQ: Positive integer

        alpha:  Confidence level of the interval
                REQ: Float in (0, 1)


        RETURNS
        -------
        np.ndarray(float) <-> len(thresh)
        lower:  The lower bounds of the confidence interval 
                    of indicator correlations for the Gaussian copula values

        np.ndarray(float) <-> len(thresh)
        upper:  The upper bounds of the confidence interval 
                    of indicator correlations for the Gaussian copula values
    '''

    nInd = thresh.shape[0]

    # Creating the correlation matrix to create simulations
    corr = np.array([[1., rho], [rho, 1.]], dtype=float)
    sqrtCorr = sqrt(corr)

    indCorrs = np.zeros([nEns, nInd])

    for ens in range(nEns):
        # Simulate independent normal data and transform it
        simData_indep = np.random.normal(0, 1, size=(nSim, 2))
        simData = simData_indep @ sqrtCorr

        simProb = rankdata(simData, method='max', axis=0) / (nSim + 1)

        simIndCorrs = []

        for th in thresh:
            xLow = (simProb[:, 0] <= th).astype(float)
            yLow = (simProb[:, 1] <= th).astype(float)
            pxLow = np.mean(xLow)
            pyLow = np.mean(yLow)

            if (pxLow == 0 and pyLow == 0) or (pxLow == 1 and pyLow == 1):
                indCorr = 1
            elif (pxLow == 0 and pyLow == 1) or (pxLow == 1 and pyLow == 0):
                indCorr = -1
            else:
                indCorr = np.corrcoef(xLow, yLow)[0, 1]

            simIndCorrs += [indCorr]

        indCorrs[ens] = np.array(simIndCorrs)

    indCorrs.sort(axis=0) # Sort each column independently

    '''
    Each column corresponds to the indicator correlation for a given threshold
    So each column gets sorted in ascending order of correlation

    Now, the confidence interval composes of rows (edge, nSim - edge)
    '''
    # Number of simulations that would be on either extreme of the simulations
    edge = int(nEns * (1 - alpha) / 2)

    return indCorrToGauss(indCorrs[edge], thresh), indCorrToGauss(indCorrs[-edge], thresh) # = lower, upper

def indCorrbounds(strand, obsThresh, nSim, nLev, nEns=None, alpha=0.90):
    # ABOUT
    '''
        FUNCTION
        --------
        Computes the bounds of 90% confidence interval bounds 
            for indicator correlations of a given strand
            at given thresholds of observation


        PARAMETERS
        ----------
        strand:     Strand defining the copula whose indicator correlations are to be observed
                    REQ: Strand

        obsThresh:  Thresholds at which indicator correlations are observed
                    REQ: 1-dimensional numpy.ndarray ranging in [0, 1]
        
        nSim:       No. of simulations of the copula to be generated
                    REQ: Positive integer
        
        nLev:       No. of levels to use in the copula simulation
                    REQ: Positive integer
        
        nEns:       No. of ensembles to run the copula simulations, from which the interval is chosen
                    REQ: Positive integer

        alpha:      Confidence level of the interval
                    REQ: Float in (0, 1)


        RETURNS
        -------
        np.ndarray(float) <-> len(obsThresh)
        lower:  Lower bounds of the confidence interval of indicator correlations 
                    at the respective thresholds of observation

        np.ndarray(float) <-> len(obsThresh)
        upper:  Upper bounds of the confidence interval of indicator correlations 
                    at the respective thresholds of observation
    '''

    if nEns is None: nEns = 500 if nSim < 100 else 200 # no. of ensembles

    strandThresh, sqrtCorrs = strand.getInfo()

    ensSimCorrs = np.zeros([nEns, len(obsThresh)])

    for ens in range(nEns):
        ensSimData = copSim(sqrtCorrs, strandThresh, nSim, nLev)
        ensSimCorrs[ens] = indCorrs(ensSimData, thresh=obsThresh)

    lim = 100 * (1.00 - alpha) / 2
    upper = np.percentile(ensSimCorrs, 100 - lim, axis=0)
    lower = np.percentile(ensSimCorrs, lim, axis=0)
    
    lower[0] = lower[-1] = 1.0
    upper[0] = upper[-1] = 1.0

    return lower, upper



# ------------------------------------------------------------------------------
# SIMULATION FUNCTIONS
# ------------------------------------------------------------------------------



def copSim(sqrtCorrs, taus, nSim, nLev, pts=None):
    # ABOUT
    '''
        FUNCTION
        --------
        Simulates data from a copula with given indicator correlations' root matrices and taus
            Taus are the points at which indicator correlations change value
        

        PARAMETERS
        ----------
        sqrtCorrs:  List of square root correlation matrices corresponding to the taus
                    REQ: List of 2-dimensional non-empty numpy.ndarray
        
        taus:       List of Gaussian correlation change points corresponding to the sqrtCorrs
                    REQ: List of values in (0, 1) of same length as sqrt
        
        nSim:       No. of points to simulate
                    REQ: Positive integer
        
        nLev:       No. of levels to use in the copula simulation (more levels = more accuracy but more time)
                    REQ: Positive integer


        RETURNS
        -------
        np.ndarray(float) <-> (nSim, 2)
        copSimData: Simulated data from the copula
    '''


    # In the bivariate case
    nDim = 2

    '''
    Randomly generate points from a multivariate standard normal distribution
    These exact same points will run through each level and be simulated
    '''
    if pts is None:
        pts = np.random.multivariate_normal(
            np.zeros(nDim), 
            np.eye(nDim), 
            size=nSim
            )

    xPath = np.zeros([nSim, nLev, nDim])

    '''
    Probability levels of the standard normal
    '''
    alfie = np.array([stats.norm.ppf((j + 0.5) / nLev) for j in range(nLev)])

    for lev in range(nLev):
    # For each level

        # Create a t in the corresponding level
        t = (lev + 0.5) / nLev

        '''
        Find the interval for t within the given taus
            In this case, t ∈ [taus[it]], taus[it + 1])
        '''
        it = np.searchsorted(taus, t, side="right") - 1
        
        '''
        Create the coefficient for interpolation, alp
        Using this, interpolate the correlation matrices
            on either side of the interval (taus[it], taus[it + 1])
        '''
        beta = (taus[it + 1] - t) / (taus[it + 1] - taus[it])
        sqrtCorr = beta * sqrtCorrs[it] + (1 - beta) * sqrtCorrs[it + 1]

        '''
        Multiply square root of correlation matrix with independent normal values
        Each level corresponds to the given generated values
        interpolated with the correlation matrix
        '''
        xPath[:, lev, :] = pts @ sqrtCorr

    copSimData = np.zeros([nSim, nDim])   
    
    # ITERATIVE VERSION
    '''
    crossed = np.zeros([nSim, nDim], dtype=bool) 
    
    for sim in range(nSim):
    # For each simulation

        for var in range(nDim):
        # For each variable

            for lev in range(nLev):
            # For each level
                if (xPath[sim, lev, var] < alfie[lev]) and (not crossed[sim, var]):
                    if lev > 0:
                        beta = (xPath[sim, lev, var] - alfie[lev-1]) / (alfie[lev] - alfie[lev-1])
                        copSimData[sim, var] = beta * alfie[lev] + (1 - beta) * alfie[lev-1]
                    else:
                        copSimData[sim, var] = xPath[sim, lev, var]
                    crossed[sim, var] = True

                if (lev == nLev - 1 and not crossed[sim, var]):
                    copSimData[sim, var] = xPath[sim, lev, var]
    '''

    # VECTORISED VERSION

    # Instead of loops, vectorised code is used to cut down runtime
    
    '''
    crossingMask: np.ndarray(bool) <-> (nSim, nLev, nDim)
    It indicates whether or not each dimension of the individual simulated datapoint 
        crosses the probability level at each level of the simulation
        (ergo, whether the moving datapoint's curve crosses the standard normal diagonal)
    
    alfie[np.newaxis, :, np.newaxis]: np.ndarray(float) <-> (1, nLev, 1)
    It basically reshapes the array to ensure each variable at each simulation can be compared
    '''
    crossingMask = xPath < alfie[np.newaxis, :, np.newaxis]

    '''
    simIndices: np.ndarray(int) <-> (nSim, 1)
    varIndices: np.ndarray(int) <-> (1, nDim)

    These arrays just enumerate the indices to use in the vectorised code
    It eliminates the need for loops to iterate through simulations and variables

    simIndices looks like [[0], [1], [2], ..., [nSim-1]]
    varIndices is [[0, 1]]
    '''
    simIndices = np.arange(nSim)[:, np.newaxis]
    varIndices = np.arange(nDim)[np.newaxis, :]

    '''
    HOW DOES THE FILLING IN WORK?
        The filling follows the following code pattern:
            copSimData[cond] <- xPath[simIndices, level, varIndices][cond]
        
        The cond filters the copSimData locations to fill out the xPath which satisfy the condition,
            which is why both have the [cond]
        
        The [simIndices, __, varIndices] replace the need for the loops

        The [__, level, __] inserts the associated probability level
            that is determined by the condition
    '''
    
    '''
    anyCrossing: np.ndarray(bool) <-> (nSim, nDim)
    It indicates whether or not each variable of the individual simulated datapoint
        crosses the probability level at any level of the simulation at all
    

    The axis of nLev is collapsed
    Any crossing at any level at all for a variable of a simulation
        results in a True value at the given variable and simulation in anyCrossing
    '''
    anyCrossing = crossingMask.any(axis=1)

    '''
    CASE 1: 
    None of the levels are crossed by a certain variable from a certain simulated datapoint
    The very last probability level is used

    No crossing <-> cond:        ~anyCrossing
    Last level  <-> level index: nLev - 1
    '''
    copSimData[~anyCrossing] = xPath[simIndices, nLev - 1, varIndices][~anyCrossing]
    
    '''
    crossLev: np.ndarray(int) <-> (nSim, nDim)
    It indicates the index of the first probability level that is higher than the path
        for each variable of the individual simulated datapoint
    Graphically, this is the level just after the crossing
    
    The axis of nLev is collapsed
    The index of the first crossing level is taken as the crossing level for the given variable and simulation
        If it does not cross, argmax defaults to 0,
            but this case has already been handled first to avoid the 0 value ambiguity
    '''
    crossLev = np.argmax(crossingMask, axis=1)
    
    
    '''
    CASE 2:
    The very first level was crossed by a certain variable from a certain simulated datapoint
    The very first probability level is used

    Crossing occured at the first level index 
                <-> cond:        anyCrossing & (crossLev == 0)
    First level <-> level index: 0
    '''
    cond0 = anyCrossing & (crossLev == 0)
    copSimData[cond0] = xPath[simIndices, 0, varIndices][cond0]
    
    '''
    CASE 3:
    Interpolation is necessary

    There is a crossing at some point and it is beyond the first level
                <-> cond:           anyCrossing & (crossLev > 0)

    Level requires interpolation between crossed level and the one prior to cross
                <-> level indices:  lev, lev-1
    '''
    cond = anyCrossing & (crossLev > 0)
    lev = crossLev[cond]

    # beta = How far along the point is after the level prior to crossing
    #               / Difference between the probability levels containing crossing
    beta = (xPath[simIndices, crossLev, varIndices][cond] - alfie[lev-1]) / (alfie[lev] - alfie[lev-1])

    copSimData[cond] = beta * alfie[lev] + (1-beta) * alfie[lev-1]

    
    # Rank normalize the columns
    copSimData = np.array([rankdata(colT, method='max') for colT in copSimData.T]).T / (nSim + 1)

    return copSimData

def gaussianIndCorrInterval(corr, n, thresh, alpha=0.05, nSim=2000):
    # ABOUT
    '''
        FUNCTION
        --------
        Defines a confidence interval to determine possible indicator correlation values
            given the thresholds and a corresponding Gaussian correlation
        
        PARAMETERS
        ----------
        corr:   Gaussian correlation between the two variables
                REQ: Value in [-1, 1]
        
        n:      Number of datapoints per simulation
                REQ: Positive integer

        alpha:  Significance level for the confidence interval
                REQ: Value in (0, 1)
        
        thresh: List of indicator thresholds when measuring the change in correlation
                REQ: Non-empty list of values in (0, 1)

        nSim:   Number of simulations, within which the confidence interval is defined
                REQ: Positive integer
        

        RETURNS
        -------
        np.ndarray(float) <-> len(thresh)
        indCorrsLow:    Lower bound of the confidence interval for the indicator correlation
        
        np.ndarray(float) <-> len(thresh)
        indCorrsHigh:   Upper bound of the confidence interval for the indicator correlation
    '''

    nInd = thresh.shape[0]

    # Creating the square root correlation matrix to create simulations
    corrMatrix = np.ones([2, 2])
    corrMatrix[0, 1] = corr
    corrMatrix[1, 0] = corr

    sqrtCorr = sqrt(corrMatrix)

    indCorrs = np.zeros([nSim, nInd])

    for sim in range(nSim):
    # Within each simulation

        # Simulate independent normal data and transform it
        simData_indep = np.random.normal(0, 1, size=(n, 2))
        simData = simData_indep @ sqrtCorr

        simProb = rankdata(simData, method='max', axis=0) / (n + 1)

        simIndCorrs = []

        for th in thresh:
            xLow = (simProb[:, 0] <= th).astype(float)
            yLow = (simProb[:, 1] <= th).astype(float)
            pxLow = np.mean(xLow)
            pyLow = np.mean(yLow)

            xHigh = (simProb[:, 0] > th).astype(float)
            yHigh = (simProb[:, 1] > th).astype(float)
            pxHigh = np.mean(xHigh)
            pyHigh = np.mean(yHigh)

            if (pxLow == 0 and pyLow == 1) or (pxLow == 1 and pyLow == 0):
                indCorr = -1
            elif (pxLow == 0 or pyLow == 0):
                indCorr = np.corrcoef(xHigh, yHigh)[0, 1]
            else:
                indCorr = np.corrcoef(xLow, yLow)[0, 1]

            simIndCorrs += [indCorr]

        indCorrs[sim] = np.array(simIndCorrs)

    indCorrs.sort(axis=0) # Sort each column independently

    '''
    Each column corresponds to the indicator correlation for a given threshold
    So each column gets sorted in ascending order of correlation

    Now, the confidence interval composes of rows (edge, nSim - edge)
    '''
    # Number of simulations that would be on either extreme of the simulations
    edge = int(nSim * alpha / 2)

    return indCorrs[edge], indCorrs[-edge]



