import numpy as np
import pandas as pd
from .format_data import format_data

def data_aids(all_age_groups = False):
    
    """
    
    Organises UK aids data into 'format_data' output. 
    
    The data set is taken from table 1 of De Angelis and Gilks (1994). The data
    are also analysed by Davison and Hinkley (1998, Example 7.4). The data are
    reporting delays for AIDS counting the number of cases by the date of
    diagnosis and length of reporting delay, measured by quarter.

    The data set is in "trapezoid"-format. The original data set is unbalanced
    in various ways: the first column covers a reporting delay of less than one
    month (or should it be less than one quarter?); last column covers a
    reporting delay of at least 14 quarters; last diagonal include incomplete
    counts. The default data set excludes the incomplete counts in the last
    diagonal, but includes the unbalanced first and last columns.
    
    
    Parameters
    ----------
    
    all_age_groups : bool, optional
                     Determines if data for the last, unbalanced, period should
                     be included. (Default is 'False'.)

    
    Notes
    -----
    
    The data description is taken from the R package apc.


    References
    ----------
    
    De Angelis, D. and Gilks, W.R. (1994) Estimating acquired immune deficiency
    syndrome incidence accounting for reporting delay. Journal of the Royal
    Statistical Sociey A 157, 31-40.

    Davison, A.C. and Hinkley, D.V. (1998) Bootstrap methods and their
    application. Cambridge: Cambridge University Press.

    """
    
    aids_response = pd.DataFrame(np.array([[   2.,    6.,    0.,    1.,    1.,    0.,    0.,    1.,    0.,
           0.,    0.,    0.,    0.,    0.,    1.],
       [   2.,    7.,    1.,    1.,    1.,    0.,    0.,    0.,    0.,
           0.,    0.,    0.,    0.,    0.,    0.],
       [   4.,    4.,    0.,    1.,    0.,    2.,    0.,    0.,    0.,
           0.,    2.,    1.,    0.,    0.,    0.],
       [   0.,   10.,    0.,    1.,    1.,    0.,    0.,    0.,    1.,
           1.,    1.,    0.,    0.,    0.,    0.],
       [   6.,   17.,    3.,    1.,    1.,    0.,    0.,    0.,    0.,
           0.,    0.,    1.,    0.,    0.,    1.],
       [   5.,   22.,    1.,    5.,    2.,    1.,    0.,    2.,    1.,
           0.,    0.,    0.,    0.,    0.,    0.],
       [   4.,   23.,    4.,    5.,    2.,    1.,    3.,    0.,    1.,
           2.,    0.,    0.,    0.,    0.,    2.],
       [  11.,   11.,    6.,    1.,    1.,    5.,    0.,    1.,    1.,
           1.,    1.,    0.,    0.,    0.,    1.],
       [   9.,   22.,    6.,    2.,    4.,    3.,    3.,    4.,    7.,
           1.,    2.,    0.,    0.,    0.,    0.],
       [   2.,   28.,    8.,    8.,    5.,    2.,    2.,    4.,    3.,
           0.,    1.,    1.,    0.,    0.,    1.],
       [   5.,   26.,   14.,    6.,    9.,    2.,    5.,    5.,    5.,
           1.,    2.,    0.,    0.,    0.,    2.],
       [   7.,   49.,   17.,   11.,    4.,    7.,    5.,    7.,    3.,
           1.,    2.,    2.,    0.,    1.,    4.],
       [  13.,   37.,   21.,    9.,    3.,    5.,    7.,    3.,    1.,
           3.,    1.,    0.,    0.,    0.,    6.],
       [  12.,   53.,   16.,   21.,    2.,    7.,    0.,    7.,    0.,
           0.,    0.,    0.,    0.,    1.,    1.],
       [  21.,   44.,   29.,   11.,    6.,    4.,    2.,    2.,    1.,
           0.,    2.,    0.,    2.,    2.,    8.],
       [  17.,   74.,   13.,   13.,    3.,    5.,    3.,    1.,    2.,
           2.,    0.,    0.,    0.,    3.,    5.],
       [  36.,   58.,   23.,   14.,    7.,    4.,    1.,    2.,    1.,
           3.,    0.,    0.,    0.,    3.,    1.],
       [  28.,   74.,   23.,   11.,    8.,    3.,    3.,    6.,    2.,
           5.,    4.,    1.,    1.,    1.,    3.],
       [  31.,   80.,   16.,    9.,    3.,    2.,    8.,    3.,    1.,
           4.,    6.,    2.,    1.,    2.,    6.],
       [  26.,   99.,   27.,    9.,    8.,   11.,    3.,    4.,    6.,
           3.,    5.,    5.,    1.,    1.,    3.],
       [  31.,   95.,   35.,   13.,   18.,    4.,    6.,    4.,    4.,
           3.,    3.,    2.,    0.,    3.,    3.],
       [  36.,   77.,   20.,   26.,   11.,    3.,    8.,    4.,    8.,
           7.,    1.,    0.,    0.,    2.,    2.],
       [  32.,   92.,   32.,   10.,   12.,   19.,   12.,    4.,    3.,
           2.,    0.,    2.,    2.,    0.,    2.],
       [  15.,   92.,   14.,   27.,   22.,   21.,   12.,    5.,    3.,
           0.,    3.,    3.,    0.,    1.,    1.],
       [  34.,  104.,   29.,   31.,   18.,    8.,    6.,    7.,    3.,
           8.,    0.,    2.,    1.,    2.,   np.nan],
       [  38.,  101.,   34.,   18.,    9.,   15.,    6.,    1.,    2.,
           2.,    2.,    3.,    2.,   np.nan,   np.nan],
       [  31.,  124.,   47.,   24.,   11.,   15.,    8.,    6.,    5.,
           3.,    3.,    4.,   np.nan,   np.nan,   np.nan],
       [  32.,  132.,   36.,   10.,    9.,    7.,    6.,    4.,    4.,
           5.,    0.,   np.nan,   np.nan,   np.nan,   np.nan],
       [  49.,  107.,   51.,   17.,   15.,    8.,    9.,    2.,    1.,
           1.,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan],
       [  44.,  153.,   41.,   16.,   11.,    6.,    5.,    7.,    2.,
          np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan],
       [  41.,  137.,   29.,   33.,    7.,   11.,    6.,    4.,    3.,
          np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan],
       [  56.,  124.,   39.,   14.,   12.,    7.,   10.,    1.,   np.nan,
          np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan],
       [  53.,  175.,   35.,   17.,   13.,   11.,    2.,   np.nan,   np.nan,
          np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan],
       [  63.,  135.,   24.,   23.,   12.,    1.,   np.nan,   np.nan,   np.nan,
          np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan],
       [  71.,  161.,   48.,   25.,    5.,   np.nan,   np.nan,   np.nan,   np.nan,
          np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan],
       [  95.,  178.,   39.,    6.,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,
          np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan],
       [  76.,  181.,   16.,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,
          np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan],
       [  67.,   66.,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,
          np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan]]))
    
    aids_response.columns = ['0*', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14+']
    
    if all_age_groups is False:
        for i,row in enumerate(reversed(range(30,38))):
            aids_response.iloc[row,i+1] = np.nan
            
        return format_data(response = aids_response.T, data_format = 'trapezoid',
                           age1 = 0, coh1 = 1983.5, 
                           unit = 1/4, label = "UK AIDS -clean")
    
    if all_age_groups is True:
        return format_data(response = aids_response.T, data_format = 'trapezoid',
                           age1 = 0, coh1 = 1983.5,
                           unit = 1/4, label = "UK AIDS - all: last column reporting delay >= 14, last diagonal: incomplete count")