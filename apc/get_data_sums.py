import pandas as pd
from ._ReturnValue import _ReturnValue

def get_data_sums(formatted_data, data_type = 'all', average = False):
    
    """
    
    Computes sums or averages over age, period and cohort.
    
    
    Parameters
    ----------
    
    formatted_data : output of 'apc.format_data'
    
    data_type : {'response', 'dose', 'rate', 'mortality', 'all'}, optional
                Determines what the sums are computed for. 'mortlity' maps to
                'rate'. If 'all' is chosen and 'dose' is 'None', output is for
                'response'.
                
    average : bool, optional
              Determines whether sums or averages are computed. (Default is 'False')
                    
    
    Returns
    -------
    
    Class
    
    by_age : sums (or averages) by age
    
    by_period : sums (or averages) by period
    
    by_cohort : sums (or averages) by cohort
    
    
    Examples
    --------
    
    >>> import apc
    >>> data = apc.data_Italian_bladder_cancer()
    >>> data_sums = apc.get_data_sums(data)
    >>> print(data_sums.by_age)
    >>> print('\n')
    >>> print(data_sums.by_period)
    >>> print('\n')
    >>> print(data_sums.by_cohort)
    
    
    """
    
    if data_type.lower() not in ['response', 'dose', 'rate', 'mortality',
                                 'all']:
        
        raise ValueError('\'data_type\' not recognized.')
        
    if data_type.lower() in ['dose', 'rate', 'mortality'] and formatted_data.dose is None:
        
        raise ValueError('{} not available.'.format(data_type.title()))
    
    data_as_vector = formatted_data.data_as_vector
    
    if data_type.lower() == 'all':
        col_selector = data_as_vector.columns
    elif data_type.lower() == 'mortality':
        col_selector = 'Rate'
    else:
        col_selector = data_type.title()
        
    if average is False:
        by_age = data_as_vector[col_selector].sum(level = 'Age')
        by_period = data_as_vector[col_selector].sum(level = 'Period')
        by_cohort = data_as_vector[col_selector].sum(level = 'Cohort')
    elif average is True:
        by_age = data_as_vector[col_selector].mean(level = 'Age')
        by_period = data_as_vector[col_selector].mean(level = 'Period')
        by_cohort = data_as_vector[col_selector].mean(level = 'Cohort')
    else:
        raise ValueError('\'average\' has to be either \'True\' or \'False\'')
    
    
    return _ReturnValue(by_age = by_age, by_period = by_period, 
                       by_cohort = by_cohort, average = average)
