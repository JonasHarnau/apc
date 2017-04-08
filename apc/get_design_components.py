import numpy as np
import pandas as pd
from ._ReturnValue import _ReturnValue

def get_design_components(formatted_data):
    
    """
        
    Creates the components of the design matrix.
    
    This function creates the 'level', 'slopes' and 'double_diffs'. Combined
    this would yield a collinear design. The theory is described in Nielsen
    (2014, 2015) which generalises introduced by Kuang, Nielsen and Nielsen
    (2008). In normal use this function is needed for internal use by
    'get_design'.

    
    Parameters
    ----------
    
    formatted_data : output of 'apc.format_data'
    
    Returns
    -------
    
    Class
    The class contains the following.
    
    anchor_index : int
                   This is what Nielsen (2014, 2015) defines as 'U'. (U,U) is 
                   the array index in age-cohort space that contains only the
                   level, starting to count from 0. 
    
    per_odd : bool
              This is 'True' if the period before the first observed period has
              an odd index, starting to count from 1 for age, period and cohort.
    
    slopes : dict
             Keys are 'Age', 'Period' and 'Cohort', values are the designs for
             the accompanying slopes.
    
    double_diffs : dict
                   Keys are 'Age', 'Period' and 'Cohort', values are the design
                   for the accompanying slopes.
        
    
    Notes
    -----
    
    The description is largely taken from the R package apc.
    
    
    See also
    --------
    
    apc.get_design : This picks out the relevant components from
                     'apc.get_design_components' and puts them together into a
                     design.
    
    
    Examples
    --------
    
    >>> import apc
    >>> data = apc.data_Italian_bladder_cancer()
    >>> AC_design = apc.get_design_components(data)
    >>> AC_design
    
    
    References
    ----------
    
    Kuang, D., Nielsen, B. and Nielsen, J.P. (2008a) Identification of the age-
    period-cohort model and the extended chain ladder model. Biometrika 95, 979-
    986. 
    
    Nielsen, B. (2014) Deviance analysis of age-period-cohort models.
    
    Nielsen, B. (2015) apc: An R package for age-period-cohort analysis. R
    Journal 7, 52-64.
    
    """
    
    index = formatted_data.data_as_vector.index
    
    #check the index ordering is as intended, if not rebuild the index
    if index.names != ['Age','Period','Cohort']:
        formatted_data.data_as_vector.reset_index().set_index(['Age', 'Period', 'Cohort'])
        index = formatted_data.data_as_vector.index
        
    index_levels_age = index.levels[0]
    index_levels_per = index.levels[1]
    index_levels_coh = index.levels[2]
    index_trap = pd.DataFrame(index.labels, index = index.names).T.loc[:,['Age','Cohort']]
    
    n_age = formatted_data.n_age
    n_coh = formatted_data.n_coh
    n_per = formatted_data.n_per
    n_missing_lower_per = formatted_data.n_missing_lower_per
    n_obs = formatted_data.n_obs
    
    anchor_index = int((n_missing_lower_per + 3)/2 -1)
    per_odd = False if n_missing_lower_per % 2 == 0 else True 
    
    level = pd.Series(1, index = range(n_obs), name = 'level')
    
    slope_age = index_trap['Age'] - anchor_index
    slope_age.rename('slope_age', inplace = True)
    slope_coh = index_trap['Cohort'] - anchor_index
    slope_coh.rename('slope_coh', inplace = True)
    slope_per = slope_age + slope_coh
    slope_per.rename('slope_per', inplace = True)
    
    dd_age_col = ['dd_age_{}'.format(age) for age in index_levels_age[2:]]
    dd_age = pd.DataFrame(0, index = range(n_obs), columns = dd_age_col)
    
    for i in range(anchor_index):
        dd_age.loc[slope_age == slope_age[0] + i, i:anchor_index] = np.arange(1, anchor_index - i + 1)
    
    for i in range(1, n_age - anchor_index):
        dd_age.loc[slope_age == i + 1, anchor_index:anchor_index + i] = np.arange(i, 0, -1)
    
    dd_per_col = ['dd_per_{}'.format(per) for per in index_levels_per[2:]]
    dd_per = pd.DataFrame(0, index = range(n_obs), columns = dd_per_col)
    
    if per_odd:
        dd_per.loc[slope_per == -1,0:1] = 1
    
    for j in range(n_per - 2 - per_odd):
        dd_per.loc[slope_per == j + 2, int(per_odd):j + 1 + int(per_odd)] = np.arange(j + 1, 0, -1)
    
    dd_coh_col = ['dd_coh_{}'.format(coh) for coh in index_levels_coh[2:]]
    dd_coh = pd.DataFrame(0, index = range(n_obs), columns = dd_coh_col)
    
    
    for k in range(anchor_index):
        dd_coh.loc[slope_coh == - anchor_index + k, k:anchor_index] = np.arange(1, anchor_index - k + 1)
    
    for k in range(1, n_coh - anchor_index):
        dd_coh.loc[slope_coh == k + 1, anchor_index:anchor_index + k] = np.arange(k, 0, -1)
    
    return _ReturnValue(anchor_index = anchor_index, per_odd = per_odd,
                        level = level,
                        slopes = {'Age' : slope_age, 'Period' : slope_per, 'Cohort' : slope_coh},
                        double_diffs = {'Age' : dd_age, 'Period' : dd_per, 'Cohort' : dd_coh})