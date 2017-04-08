import numpy as np
import pandas as pd
from .format_data import format_data

def data_Belgian_lung_cancer():
    
    """
    Organises Belgian lung cancer data into 'format_data' output. 
    
    The data set is taken from table VIII of Clayton and Schifflers (1987),
    which contains age-specific incidence rates (per 100,000 person-years
    observation) of lung cancer in Belgian females during the period 1955-1978.
    Numerators are also available. The original source was the WHO mortality
    database.

    The 'data_format' is "AP". The original data set is unbalanced since the
    first four period groups cover 5 years, while the last covers 4 years. The
    unbalanced period group is not included in this data set.
        
       
    Notes
    -----
    
    The data description is largely taken from the R package apc.


    References
    ----------
    
    Clayton, D. and Schifflers, E. (1987) Models for temperoral variation in
    cancer rates. I: age-period and age-cohort models. Statistics in Medicine 6,
    449-467.
    
    """
    
    index = ['25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64',
       '65-69', '70-74', '75-79']
    
    columns = ['1955-1959', '1960-1964', '1965-1969', '1970-1974']
    
    
    lung_cases = pd.DataFrame(np.array(([  3,   2,   7,   3],
                                        [ 11,  16,  11,  10],
                                        [ 11,  22,  24,  25],
                                        [ 36,  44,  42,  53],
                                        [ 77,  74,  68,  99],
                                        [106, 131,  99, 142],
                                        [157, 184, 189, 180],
                                        [193, 232, 262, 249],
                                        [219, 267, 323, 325],
                                        [223, 250, 308, 412],
                                        [198, 214, 253, 338])).reshape((11, 4)),
                             index = index, columns = columns)
    
    lung_rates = pd.DataFrame(np.array(([  0.19,   0.13,   0.5 ,   0.19],
                                        [  0.66,   0.98,   0.72,   0.71],
                                        [  0.78,   1.32,   1.47,   1.64],
                                        [  2.67,   3.16,   2.53,   3.38],
                                        [  4.84,   5.6 ,   4.93,   6.05],
                                        [  6.6 ,   8.5 ,   7.65,  10.59],
                                        [ 10.36,  12.  ,  12.68,  14.34],
                                        [ 14.76,  16.37,  18.  ,  17.6 ],
                                        [ 20.53,  22.6 ,  24.9 ,  24.33],
                                        [ 26.24,  27.7 ,  30.47,  36.94],
                                        [ 33.47,  33.61,  36.77,  43.69])).reshape((11, 4)),
                             index = index, columns = columns)
    
    return format_data(lung_cases, rate = lung_rates, data_format = 'AP')