from .utils import *

# Matrix functions

def rankCorrMatrix(data):
    # ABOUT
    '''
        FUNCTION
        --------
        Measures pairwise rank correlations between variables in a dataset
        Rank correlation is achieved when the ranks of the variables move together
        

        PARAMETERS
        ----------
        data:   Array of all of the data with observations as rows and variables as columns
                Pandas dataframes can be passed in through their values (df.values)
                REQ: 2D non-empty numpy.ndarray
            

        RETURNS
        -------
        2D numpy.ndarray
        rankCorrMatrix:   Pairwise Spearman's rank correlation matrix
    '''
    
    # How many variables are there?
    nDim = data.shape[1]
    
    '''  
    A variable has perfect Spearman's rank correlation to itself
        => begin correlation matrix with all ones
    '''
    rankCorrMatrix = np.ones([nDim, nDim])
    
    # Iterate through the columns
    '''
    To avoid going over the same pair of columns multiple times,
        the following iteration is used
    It exploits symmetry of the matrix
    '''
    
    for var1 in range(nDim - 1):
    # The last variable is automatically filled in
    
        for var2 in range(var1 + 1, nDim):
        # Fill for all variables after it (which is how the last variable gets filled in)
            rankCorrMatrix[var1, var2] = rankCorrMatrix[var2, var1] = rankCorr(data[:, var1], data[:, var2])
    
    return rankCorrMatrix
    
def normRankCorrMatrix(data):
    # ABOUT
    '''
        FUNCTION
        --------
        Measures pairwise normal rank correlations between variables in a dataset
        Normal rank correlation is achieved when the normal ranks of the variables move together
        

        PARAMETERS
        ----------
        data:   Array of all of the data with observations as rows and variables as columns
                Pandas dataframes can be passed in through their values (pandas.DataFrame.values)
                REQ: 2D non-empty numpy.ndarray
            

        RETURNS
        -------
        2D numpy.ndarray
        normRankCorrMatrix:   Pairwise normal rank correlation matrix
    '''
    
    # How many variables are there?
    nDim = data.shape[1]
    
    '''  
    A variable has perfect normal rank correlation to itself
        => begin correlation matrix with all ones
    '''
    normRankCorrMatrix = np.ones([nDim, nDim])
    
    # Iterate through the columns
    '''
    To avoid going over the same pair of columns multiple times,
        the following iteration is used
    It exploits symmetry of the matrix
    '''
    
    for var1 in range(nDim - 1):
    # The last variable is automatically filled in
    
        for var2 in range(var1 + 1, nDim):
        # Fill for all variables after it (which is how the last variable gets filled in)
                normRankCorrMatrix[var1, var2] = normRankCorrMatrix[var2, var1] = normRankCorr(data[:, var1], data[:, var2])
    
    return normRankCorrMatrix

def dirAsyMatrix(data):
    # ABOUT
    '''
        FUNCTION
        --------
        Measures pairwise direct asymmetries between variables in a dataset
        

        PARAMETERS
        ----------
        data:   Array of all of the data with observations as rows and variables as columns
                Pandas dataframes can be passed in through their values (pandas.DataFrame.values)
                REQ: 2D non-empty numpy.ndarray
            

        RETURNS
        -------
        2D numpy.ndarray
        dirAsyMatrix:   Pairwise direct asymmetry matrix
    '''
    
    # How many variables are there?
    nDim = data.shape[1]
    
    '''  
    A variable is perfectly symmetrical to itself
        => begin asymmetry matrix with all ones
    '''
    dirAsyMatrix = np.zeros([nDim, nDim])
    
    # Iterate through the columns
    '''
    To avoid going over the same pair of columns multiple times,
        the following iteration is used
    It exploits symmetry of the matrix
    '''
    
    for var1 in range(nDim - 1):
    # The last variable is automatically filled in
    
        for var2 in range(var1 + 1, nDim):
        # Fill for all variables after it (which is how the last variable gets filled in)
                dirAsyMatrix[var1, var2] = dirAsyMatrix[var2, var1] = directAsymmetry(data[:, var1], data[:, var2])
    
    return dirAsyMatrix

def concAsyMatrix(data):
    # ABOUT
    '''
        FUNCTION
        --------
        Measures pairwise concordance asymmetries between variables in a dataset
        

        PARAMETERS
        ----------
        data:   Array of all of the data with observations as rows and variables as columns
                Pandas dataframes can be passed in through their values (pandas.DataFrame.values)
                REQ: 2D non-empty numpy.ndarray
            

        RETURNS
        -------
        2D numpy.ndarray
        concAsyMatrix:   Pairwise concordance asymmetry matrix
    '''
    
    # How many variables are there?
    nDim = data.shape[1]
    
    '''  
    A variable is perfectly symmetrical to itself
        => begin asymmetry matrix with all ones
    '''
    concAsyMatrix = np.zeros([nDim, nDim])
    
    # Iterate through the columns
    '''
    To avoid going over the same pair of columns multiple times,
        the following iteration is used
    It exploits symmetry of the matrix
    '''
    
    for var1 in range(nDim - 1):
    # The last variable is automatically filled in
    
        for var2 in range(var1 + 1, nDim):
        # Fill for all variables after it (which is how the last variable gets filled in)
                concAsyMatrix[var1, var2] = concAsyMatrix[var2, var1] = concordAsymmetry(data[:, var1], data[:, var2])
    
    return concAsyMatrix

def concAsyConfInt(corr, nPts, alpha=0.05, nEns=2000):
    # ABOUT
    '''
        FUNCTION
        --------
        Measures (alpha, 1 - alpha) confidence intervals of pairwise concordance asymmetries
        

        PARAMETERS
        ----------
        corr:   Correlation matrix to be replicated by simulated points
                REQ: 2D positive definite non-empty numpy.ndarray

        nPts:   Number of points simulated in each ensemble
                Among these points, the concordance asymmetry matrix will be measured
                    and stored per ensemble
                REQ: int

        alpha:  Confidence level at which the intervals are made
                REQ: float in (0, 1)

        nEns:   Number of ensembles in which the given number of points are replicated
                These many pairwise concordance asymmetries will be measured, 
                    out of which confidence intervals will be made
                REQ: int
            

        RETURNS
        -------
        2D numpy.ndarray
        concAsySims[edge]:  Lower bounds of confidence intervals of respective pairwise concordance asymmetries

        2D numpy.ndarray
        concAsySims[-edge]: Upper bounds of confidence intervals of respective pairwise concordance asymmetries
    '''

    # Asymmetry values for each simulation begin with an empty list
    concAsySims = []

    for i in range(nEns):
        '''
        For each simulation, simulate a multivariate normal dataset
            centered around the origin with the above correlation matrix
        
        Using this dataset of points, measure the pairwise concordance asymmetries as a matrix
        '''
        pts = np.random.multivariate_normal(np.zeros(len(corr)), corr, size=nPts)
        concAsySims += [concAsyMatrix(pts)]

    concAsySims = np.sort(concAsySims, axis=0)

    # Number of simulations that would be on either extreme of the simulations
    edge = int(nEns * alpha / 2)
    '''
    The idea is to create and order simulations
    and choose the bottom [edge] & top [-edge] simulations
    to compare asymmetry to the original data

    Essentially, the values of asymmetry from [edge, nsim - edge]
    form a two-tailed confidence interval for symmetric data
    '''

    return concAsySims[edge], concAsySims[-edge]

def dirAsyConfInt(corr, nPts, alpha=0.05, nEns=2000):
    # ABOUT
    '''
        FUNCTION
        --------
        Measures (alpha, 1 - alpha) confidence intervals of pairwise direct asymmetries
        

        PARAMETERS
        ----------
        corr:   Correlation matrix to be replicated by simulated points
                REQ: 2D positive definite non-empty numpy.ndarray

        nPts:   Number of points simulated in each ensemble
                Among these points, the direct asymmetry matrix will be measured
                    and stored per ensemble
                REQ: int

        alpha:  Confidence level at which the intervals are made
                REQ: float in (0, 1)

        nEns:   Number of ensembles in which the given number of points are replicated
                These many pairwise direct asymmetries will be measured, 
                    out of which confidence intervals will be made
                REQ: int
            

        RETURNS
        -------
        2D numpy.ndarray
        dirAsySims[edge]:  Lower bounds of confidence intervals of respective pairwise direct asymmetries

        2D numpy.ndarray
        dirAsySims[-edge]: Upper bounds of confidence intervals of respective pairwise direct asymmetries
    '''

    # Asymmetry values for each simulation begin with an empty list
    dirAsySims = []

    for i in range(nEns):
        '''
        For each simulation, simulate a multivariate normal dataset
            centered around the origin with the above correlation matrix
        
        Using this dataset of points, measure the pairwise direct asymmetries as a matrix
        '''
        pts = np.random.multivariate_normal(np.zeros(len(corr)), corr, size=nPts)
        dirAsySims += [dirAsyMatrix(pts)]

    dirAsySims = np.sort(dirAsySims, axis=0)

    # Number of simulations that would be on either extreme of the simulations
    edge = int(nEns * alpha / 2)
    '''
    The idea is to create and order simulations
    and choose the bottom [edge] & top [-edge] simulations
    to compare asymmetry to the original data

    Essentially, the values of asymmetry from [edge, nsim - edge]
    form a two-tailed confidence interval for symmetric data
    '''

    return dirAsySims[edge], dirAsySims[-edge]
