import numpy as np
import pandas as pd
from dataclasses import dataclass
import scipy.stats as stats


'''
DATA STRUCTURES
--------------
'''

@dataclass
class Strand:
    degree: int = -2
    thresh: np.ndarray = None
    rhos: np.ndarray = None
    coeffs: np.ndarray = None

    def __post_init__(self):
        '''
        VALIDATION
        ----------
        '''

        if self.degree == 1:
            if self.coeffs is None:
                assert self.thresh is not None and self.rhos is not None, "Provide thresholds and rhos for a linear strand"
                self.coeffs = np.polyfit(self.thresh, self.rhos, 1)
            if self.rhos is None:
                assert self.coeffs is not None, "Provide coefficients for a linear strand"
                self.thresh = np.array([0., 1.])
                self.rhos = np.polyval(self.coeffs, self.thresh)

        if self.degree > 1:
            # validate coeffs
            assert self.coeffs is not None, "Polynomial strand requires coefficients"
            assert self.coeffs.ndim == 1, "Coefficients must be a 1-dimensional array"
            assert len(self.coeffs) == self.degree + 1, "Coefficients inconsistent with degree"
        else:
            assert self.thresh.ndim == 1, "Provide a 1-dimensional array of thresholds"
            assert self.rhos.ndim == 1, "Provide a 1-dimensional array of rhos"
            assert len(self.thresh) == len(self.rhos), "Thresholds and rhos must be of the same length"
            assert np.all((self.thresh >= 0) & (self.thresh <= 1)), "Thresholds must lie in the interval [0, 1]"
            assert np.all(np.diff(self.thresh) != 0), "Ensure that there are no duplicate thresholds (i.e. no repeated points on the strand)"
            assert np.all(np.diff(self.thresh) > 0), "Thresholds must be in increasing order"
            assert self.thresh[0] == 0 and self.thresh[-1] == 1, "Thresholds must start at 0 and end at 1"
            assert np.all(np.abs(self.rhos) <= 1), "Correlation must be in the interval [-1, 1]"
    

    def getInfo(self, nInd=100):
        # ABOUT
        '''
            FUNCTION
            --------
            Gets the thresholds and square root correlation matrices of a given strand

            If the strand is a piecewise linear strand, 
                the thresholds are the change points of the indicator correlation structure
                    and the square root correlation matrices are computed for the rhos in the strand
            
            If the strand is a polynomial strand with degree >= 2,
                a series of (nInd) thresholds are generated 
                    and the square root correlation matrices are computed for the rhos in the strand at those thresholds


            PARAMETERS
            ----------            
            nInd:   Number of thresholds to generate if the strand is a polynomial strand with degree >= 2
                    REQ: Positive integer

            RETURNS
            -------
            np.ndarray(float)
            thresh: The thresholds of the given strand

            np.ndarray(float)
            sqrtCorrs: List of square root correlation matrices corresponding to the rhos in the given strand
            
        '''

        # LOCAL HELPER
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
        
        '''
        Piecewise linear, Gaussian strands and strands with linearly changing correlations are obvious
        '''
        if self.degree < 2:

            sqrtCorrs = [sqrt(np.array([[1, rho], [rho, 1]])) for rho in self.rhos]
            return self.thresh, sqrtCorrs
        
        '''
        Polynomial strands with degree >= 2 require 
            the creation of thresholds and square root correlation matrices at those thresholds
        '''
        thresh = np.linspace(0, 1, nInd)
        rhos = np.clip(np.polyval(self.coeffs, thresh), -1., 1.)

        sqrtCorrs = [sqrt(np.array([[1., rho], [rho, 1.]])) for rho in rhos]
        return thresh, sqrtCorrs

    def getIndex(self, t):
        # ABOUT
        '''
            FUNCTION
            --------
            Get the index of a given threshold value in the strand's thresholds
                If the passed value is not among the strand's thresholds, 
                    the index of the highest threshold not exceeding is returned


            PARAMETERS
            ----------
            t:  The threshold value to find the index of
                REQ: Scalar value in [0, 1]
            

            RETURNS
            -------
            int
            idx:    The index of the given t value in the strand's thresholds
                        OR the index of the highest threshold not exceeding t
        '''
        assert self.degree == -1, "Getting index is only defined for piecewise linear strands"
        assert 0 <= t <= 1, "Threshold must be in the interval [0, 1]"

        return np.searchsorted(self.thresh, t, side="right") - 1 # = idx
    
    def getCorr(self, t):
        # ABOUT
        '''
            FUNCTION
            --------
            Get the correlation corresponding to a given threshold value
                If the passed value is not among the strand's thresholds, 
                    the interval containing the value is found 
                        and the correlation matrices of the bounding thresholds are linearly interpolated


            PARAMETERS
            ----------
            t:  The threshold value to find the correlation of
                REQ: Scalar value in [0, 1]
            

            RETURNS
            -------
            np.ndarray(float)
            corr:   The correlation corresponding to the given threshold value in the strand's thresholds
        '''

        assert 0 <= t <= 1, "Threshold must be in the interval [0, 1]"

        if self.degree == 0: 
            '''
            Gaussian case: Return the single correlation matrix of the underlying Gaussian copula
            '''
            return np.array([[1, self.rhos[0]], [self.rhos[0], 1]]) # = corr

        elif self.degree >= 2:
            '''
            Polynomial case: Interpolate the correlation using the polynomial coefficients
            '''
            rho = np.clip(np.polyval(self.coeffs, t), -1., 1.)
            return np.array([[1, rho], [rho, 1]]) # = corr
    
        else:
            '''
            Piecewise linear case: Interpolate the correlation using the bounding correlations of the interval containing t
            This is also the same as the linear polynomial case,
                where correlations on either end of [0., 1.] are used to interpolate
            '''
            idx = self.getIndex(t)

            if np.isclose(self.thresh[idx], t): 
                return np.array([[1, self.rhos[idx]], [self.rhos[idx], 1]]) # = corr

            tLow, tHigh = self.thresh[idx], self.thresh[idx + 1]
            rhoLow, rhoHigh = self.rhos[idx], self.rhos[idx + 1]

            beta = (t - tLow) / (tHigh - tLow)
            rho = beta * rhoHigh + (1 - beta) * rhoLow

            return np.array([[1, rho], [rho, 1]]) # = corr

    def sample(self, nSam, seed=None):
        # ABOUT
        '''
            FUNCTION
            --------
            Sample independent datapoints from the Strand's changing correlation structure

            PARAMETERS
            ----------
            nSam:   Number of datapoints to sample
                    REQ: Positive integer

            seed:   Random seed for reproducibility
                    Default: None

            RETURNS
            -------
            np.ndarray <-> (nSam, 2)
            samples:    Sampled datapoints with uniform marginals
        '''
        assert isinstance(nSam, int) and nSam > 0, "No. of samples must be a positive integer"

        rng = np.random.default_rng(seed)

        u1   = rng.uniform(0, 1, nSam)
        z    = rng.standard_normal(nSam)

        rhos = np.array([self.getCorr(t)[0, 1] for t in u1])
        
        u2   = stats.norm.cdf(rhos * stats.norm.ppf(u1) + np.sqrt(1 - rhos**2) * z)

        return np.column_stack([u1, u2])

