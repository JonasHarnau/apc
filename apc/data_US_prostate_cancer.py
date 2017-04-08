import numpy as np
import pandas as pd
from .format_data import format_data

def data_US_prostate_cancer():
    
    """
    Organises US prostate cancer data into 'format_data' output. 
    
    The data set is taken from table 2 of Holford (1983), which contains age-
    specific counts of deaths and midperiod population measured in 1000s, during
    the period 1935-1969. Reported in 5 year age groups and 5 year period
    groups.

    The original source was Cancer deaths: National Center for Health
    Statistics, 1937-1973 Population 1935-60: Grove and Hetzel, 1968 Population
    1960-69: Bureau of the Census, 1974

    The 'data_format' is "AP".
    
       
    Notes
    -----
    
    The data description is largely taken from the R package apc.


    References
    ----------
    
    Holford, T.R. (1983) The estimation of age, period and cohort effects for
    vital rates. Biometrics 39, 311-324.
    
    """
    
    col_names = ['{} - {}'.format(i,i + 4) for i in range(1935,1966,5)]
    row_names = ['{} - {}'.format(i,i + 4) for i in range(50,81,5)]

    prostate_deaths = pd.DataFrame(
        np.array((177, 271, 312, 382, 321, 305, 308, 262, 350, 
        552, 620, 714, 649, 738, 360, 479, 644, 949, 932, 1292, 
        1327, 409, 544, 812, 1150, 1668, 1958, 2153, 328, 509, 
        763, 1097, 1593, 2039, 2433, 222, 359, 584, 845, 1192, 
        1638, 2068, 108, 178, 285, 475, 742, 992, 1374)).reshape((7,7)),
                                    columns = col_names, index = row_names)
    prostate_population = pd.DataFrame(
        np.array((301, 317, 353, 395, 426, 473, 498, 212, 
        248, 279, 301, 358, 411, 443, 159, 194, 222, 222, 258, 
        304, 341, 132, 144, 169, 210, 230, 264, 297, 76, 94, 
        110, 125, 149, 180, 197, 37, 47, 59, 71, 91, 108, 118, 
        19, 22, 32, 39, 44, 56, 66)).reshape((7,7)),
                                    columns = col_names, index = row_names)
    
    return format_data(response = prostate_deaths, dose = prostate_population,
                         data_format = 'AP', label = "US prostate cancer")