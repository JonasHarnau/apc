import numpy as np
import pandas as pd
from .format_data import format_data

def data_Italian_bladder_cancer():
    
    """
    Organises Italian bladder cancer data into 'format_data' output. 
    
    The data set is taken from table IV of Clayton and Schifflers (1987), which
    contains age-specific incidence rates (per 100,000 person-years observation)
    of bladder cancer in Italian males during the period 1955-1979. Numerators
    are also available. The original source was the WHO mortality database.

    The 'data_format' is "AP". 
    
       
    Notes
    -----
    
    The data description is largely taken from the R package apc.


    References
    ----------
    
    Clayton, D. and Schifflers, E. (1987) Models for temperoral variation in
    cancer rates. I: age-period and age-cohort models. Statistics in Medicine 6,
    449-467.
    
    """
    bladder_cases = pd.DataFrame(np.array([[  3,    3,    1,    4,   12],
                                              [ 16,   17,   11,    8,    8],
                                              [ 24,   29,   33,   39,   30],
                                              [ 79,   76,   82,   95,  115],
                                              [234,  185,  183,  267,  285],
                                              [458,  552,  450,  431,  723],
                                              [720,  867, 1069,  974, 1004],
                                              [890, 1230, 1550, 1840, 1811],
                                              [891, 1266, 1829, 2395, 3028],
                                              [920, 1243, 1584, 2292, 3176],
                                              [831,  937, 1285, 1787, 2659]]))
    
    
    bladder_rates = pd.DataFrame(np.array((0.03, 0.03, 0.01, 0.04, 0.12, 0.17, 0.18, 0.12, 
        0.08, 0.09, 0.32, 0.31, 0.35, 0.42, 0.32, 1.04, 1.05, 
        0.91, 1.04, 1.27, 2.86, 2.52, 2.61, 3.04, 3.16, 6.64, 
        7.03, 6.43, 6.46, 8.47, 12.71, 13.39, 14.59, 14.64, 16.38, 
        20.11, 23.98, 26.69, 27.55, 28.53, 24.4, 33.16, 42.12, 
        47.77, 50.37, 32.81, 42.31, 52.87, 66.01, 74.64, 45.54, 
        47.94, 62.05, 84.65, 104.21)).reshape((11,5)))
    
    bladder_dose = bladder_cases.divide(bladder_rates)
                                          
                                          
    return format_data(response = bladder_cases, dose = bladder_dose, data_format = 'AP',
                       age1 = 25, per1 = 1955, unit = 5, 
                       label = "Italian bladder cancer")