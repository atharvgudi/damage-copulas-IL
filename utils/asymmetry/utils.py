import numpy as np
from scipy.stats import rankdata, norm

# ----------------------------------------------------------------------------
# PAIRWISE VARIABLE FUNCTIONS
# ----------------------------------------------------------------------------


def rankCorr(x, y):
    # ABOUT
    '''
    FUNCTION
    --------
    Measures rank correlation between two variables
    Rank correlation is achieved when the ranks of the variables move together
    

    PARAMETERS
    ----------
    x:  Array of values from the first variable
        REQ: 1-dimensional non-empty numpy.ndarray

    y:  Array of values from the second variable
        REQ: 1-dimensional non-empty numpy.ndarray of same length as x
        

    RETURNS
    -------
    float
    rankCorr:   Pearson's rank correlation value between the variables
    '''
    
    n = x.shape[0]
    dist = np.random.random(n)

    # Avoid equal values
    x = x + dist*1e-7
    y = y + dist*1e-7

    # Rank Normalization
    xProb = rankdata(x, method='max') / (n + 1)
    yProb = rankdata(y, method='max') / (n + 1)

    # Rank Correlation
    '''
    numpy.corrcoef returns the correlation matrix,
    so the actual correlation is gotten by taking the [0, 1] value
    '''
    rankCorr = np.corrcoef(xProb, yProb)[0, 1]
    
    return rankCorr

def normRankCorr(x, y):
    # ABOUT
    '''
    FUNCTION
    --------
    Measures normal rank correlation between two variables
    Normal rank correlation is achieved when the normal ranks of the variables move together
    

    PARAMETERS
    ----------
    x:  Array of values from the first variable
        REQ: 1-dimensional non-empty numpy.ndarray

    y:  Array of values from the second variable
        REQ: 1-dimensional non-empty numpy.ndarray of same length as x
        

    RETURNS
    -------
    float
    normRankCorr:   Normal rank correlation value between the variables
    '''
    
    n = x.shape[0]
    dist = np.random.random(n)

    # Avoid equal values
    x = x + dist*1e-7
    y = y + dist*1e-7

    # Rank Normalization
    xProb = rankdata(x, method='max') / (n + 1)
    yProb = rankdata(y, method='max') / (n + 1)

    # Normal Rank Correlation
    '''
    Consider Pearson's rank correlation to be equivalent to unif.ppf(xProb), unif.ppf(yProb)
    It works well with monotonic variables
        but fails to capture tail dependencies & asymmetry
        
    In these cases, normal rank correlation is better to capture the correlation
    '''
    normRankCorr = np.corrcoef(norm.ppf(xProb), norm.ppf(yProb))[0, 1]

    return normRankCorr

def directAsymmetry(x, y):
    # ABOUT
    '''
    FUNCTION
    --------
    Measures direct asymmetry between two variables
    

    PARAMETERS
    ----------
    x:  Array of values from the first variable
        REQ: 1-dimensional non-empty numpy.ndarray

    y:  Array of values from the second variable
        REQ: 1-dimensional non-empty numpy.ndarray of same length as x
        

    RETURNS
    -------
    float
    directAsymmetry:   How jointly asymmetrical the data is using the values themselves
    '''
    
    n = x.shape[0]
    dist = np.random.random(n)

    # Avoid equal values
    x = x + dist*1e-7
    y = y + dist*1e-7

    # Rank Normalization
    xProb = rankdata(x, method='max') / (n + 1)
    yProb = rankdata(y, method='max') / (n + 1)
    
    # Probabilities are centered around 0
    '''
    The median value of the ranks is 0.5
    so respective the sample medians end up close to 0
    & extreme values close to 0.5 or -0.5

    This is used to measure concordance & discordance

    xProbC + yProbC almost 1
    => both observations are jointly larger than the median
    => Positive concordance

    xProbC + yProbC almost -1
    => both observations are jointly smaller than the median
    => Negative concordance

    xProbC + yProbC almost 0
    => both observations are on equidistantly opposing sides of the median
    => Discordance
    '''
    xProbC = xProb - 0.5
    yProbC = yProb - 0.5

    # Direct asymmetry is another measure of asymmetry
    directAsymmetry = np.mean(xProbC * yProbC * (xProbC + yProbC))
    '''
    (xProbC + yProbC) eliminates discordant values
    AND values jointly close to the medians
    => only concordant values away from the medians are considered

    (xProbC * yProbC) individually assesses the values' positions compared to the median
    When jointly above median, they are weighted highly positively
    When jointly below median, they are weighted highly negatively
        The negatives cancel out, but the (xProbC + yProbC) negates the sign

    Similar to concordAsymmetry, directAsymmetry shows skewness
    directAsymmetry > 0 => more highly positive concordant values
                        => asymmetric due to higher positive concordant values

    directAsymmetry < 0 => more highly negative concordant values
                        => asymmetric due to higher negative concordant values

    directAsymmetry = 0 => perfectly symmetric
                        => either the data is very discordant
                            or clustered around the median
    '''
    return directAsymmetry

def concordAsymmetry(x, y):
    # ABOUT
    '''
    FUNCTION
    --------
    Measures concordance asymmetry between two variables
    

    PARAMETERS
    ----------
    x:  Array of values from the first variable
        REQ: 1-dimensional non-empty numpy.ndarray

    y:  Array of values from the second variable
        REQ: 1-dimensional non-empty numpy.ndarray of same length as x
        

    RETURNS
    -------
    float
    concordAsymmetry:   How jointly asymmetrical the data is using concordance
    '''
    
    n = x.shape[0]
    dist = np.random.random(n)

    # Avoid equal values
    x = x + dist*1e-7
    y = y + dist*1e-7

    # Rank Normalization
    xProb = rankdata(x, method='max') / (n + 1)
    yProb = rankdata(y, method='max') / (n + 1)
    
    # Probabilities are centered around 0
    '''
    The median value of the ranks is 0.5
    so respective the sample medians end up close to 0
    & extreme values close to 0.5 or -0.5

    This is used to measure concordance & discordance

    xProbC + yProbC almost 1
    => both observations are jointly larger than the median
    => Positive concordance

    xProbC + yProbC almost -1
    => both observations are jointly smaller than the median
    => Negative concordance

    xProbC + yProbC almost 0
    => both observations are on equidistantly opposing sides of the median
    => Discordance
    '''
    xProbC = xProb - 0.5
    yProbC = yProb - 0.5

    # Concordance asymmetry is a measure of asymmetry
    concordAsymmetry = np.mean((xProbC + yProbC)**3)
    '''
    The cubing is basically doing (xProbC + yProbC)**2 * (xProbC + yProbC)

    (xProbC + yProbC) eliminates discordant values
        AND values jointly close to the medians
    => only concordant values away from the medians are considered

    (xProbC + yProbC)**2 amplifies the concordance & discards sign
    Values jointly larger AND smaller than the median are weighted highly
        while those cancelling out are almost 0

    So in a way, concordAsymmetry shows skewness as an amplified concordance
    concordAsymmetry > 0    => more highly positive concordant values
                            => asymmetric due to higher positive concordant values

    concordAsymmetry < 0    => more highly negative concordant values
                            => asymmetric due to higher negative concordant values

    concordAsymmetry = 0    => perfectly symmetric
                            => either the data is very discordant
                                or clustered around the median
    '''
    return concordAsymmetry

# fin