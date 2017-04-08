import numpy as np
import pandas as pd
from ._ReturnValue import _ReturnValue

    
def format_data(response, dose = None, rate = None, data_format = None, 
                age1 = None, per1 = None, coh1 = None, unit = 1, 
                time_adjust = None, label = None):

    """
    
    Arranges data into a format useable by the other functions in the package.
    
    This is the first step of the analysis. While usually formatted in a 
    two-way table, age-period-cohort data come in many different formats. Any
    two of age, period and cohort can be on the axes. Further, data may be
    available for responses (death counts, paid amounts) alone, or for both dose
    (exposure, cases) and response. The main job of this function is to keep
    track of what age, period and cohort goes with which entry. Since generally
    only two of the three time scales are available, the third is generated
    internally.  
    
    Parameters
    ----------
    
    response : pandas.DataFrame
               Responses formatted in a two-way table. The time scales on the 
               axes are expected to increase from the top left. For example, 
               in an age-period ("AP") array, ages increases from top to bottom 
               and period from left to right and similarly for other arrays
               types.
               
    dose : pandas.DataFrame, optional
           Numbers of doses. Should have the same format as 'response'. If
           'dose' is supplied, 'rate' are generated internally as 
           'response'/'dose'. (Default is 'None')
    
    rate : pandas.DataFrame, optional
           Rates (or mortality). Should have the same format as 'response'. If
           'rate' is supplied, 'dose' are generated internally as 
           'response'/'rate'. (Default is 'None')
    
    data_format : {"AP", "AC", "CA", "CL", "CP", "PA", "PC", "trapezoid"},
                  optional
                  This input is necessary if 'response.index.name' and 
                  'response.column.name' are not supplied. If those are any two
                  of 'Age', 'Period' and 'Cohort', the 'data_format' is inferred
                  and need not be supplied.
                  The following options are implemented:
                  "AC"
                  has age/cohort as increasing row/column index.
                  "AP"
                  has age/period as increasing row/column index.
                  "CA"
                  has cohort/age as increasing row/column index.
                  "CL"
                  has cohort/age as increasing row/column index, triangular.
                  "CP"
                  has cohort/period as increasing row/column index.
                  "PA"
                  has period/age as increasing row/column index.
                  "PC"
                  has period/cohort as increasing row/column index.
                  "trapezoid"
                  has age/period as increasing row/column index. Empty periods
                  in triangular shape at the top left and bottom right are
                  allowed.
                  
    age1 : float, optional
            Time label for the youngest age group. If this is supplied at least
            one of 'per1' and 'coh1' also has to be specified. Has to be used if
            the axis labels do not allow the time scales to be inferred. Used if
            data_format is "AC", "AP", "CA", "CL", "PA", "trapezoid".
            
    per1 : float, optional
           Time label for the oldest period group. If this is supplied at least
            one of 'age1' and 'coh1' also has to be specified. Has to be used if
            the axis labels do not allow the time scales to be inferred. Used if 
            data.format is "AP", "CP", "PA", "PC".
           
    coh1 : float, optional
           Time label for the oldest cohort group. If this is supplied at least
            one of 'age1' and 'per1' also has to be specified. Has to be used if
            the axis labels do not allow the time scales to be inferred. Used if
            data.format is "AC", "CA", "CL", "CP", "PC", "trapezoid".
           
    unit : float, optional
           Common time steps for age, period and cohort. This is only used if 
           labels are generated from 'age1', 'per1' and 'coh1' inputs. For
           quarterly data use 1/4. For monthly data use 1/12. (Default is 1.)
              
    time_adjust : int, optional
                  Specifies the relation between age, period and cohort through
                  age + cohort = period + 'time_adjust'. This is used to compute
                  the missing time labels internally. If this is 'None' this is
                  set to 0 internally with one exception: if 'data_type' is
                  any of "AC", "CA", "CL" or "trapezoid" and the minimum age and
                  cohort labels are 1 so that the minimum period is also 1.
                   
    label : str, optional
            Label for the data set. Useful when working with multiple data sets.
            
    
    Returns
    -------
    
    Class
    The Class outputs all the input arguments, possible after adjustment. If one
    of  dose' and 'rate' was supplied, the missing one is generated internally
    and reported. If the time labels were generated from axes labels, 'unit' is 
    set to the implied value.
    
    Other supplied outputs are the following.
    
    data_as_vector : pandas.DataFrame
                     A dataframe of responses and, if applicable, doses and 
                     rates in the columns. The dataframe has MultiIndex with
                     levels 'Age', 'Period' and 'Cohort', sorted by 'Age'.
                     
    n_age : Int
            the number of ages with available data
    
    n_per : Int
            the number of periods with available data
    
    n_coh : Int
        the number of cohorts with available data
    
    n_obs : Int 
            the number of data points
    
    n_missing_lower_per : Int
                          the number of missing earlier periods in age-cohort 
                          space, that is missing diagonals in the top-left.
                          
    
    
    
    Notes 
    -----
    
    If no two of 'age1', 'per1' and 'coh1' are supplied, the time labels are 
    inferred from the axes labels. Those may be numeric, strings that can be
    coerced to numeric, or strings that represent timespans, such as 
    '1955 - 1959'. For the latter case, the label used for the remaining 
    analysis is then 1955.
 
    Examples 
    --------
    
    This example makes use of the Belgian lung cancer data spreadsheet that 
    comes with the package. 'formatted_data1' generates labels from the
    axes labels. 'formatted_data2' generates them from supplied values.
    
    >>> import pandas as pd
    >>> import apc
    >>> data = pd.read_excel('data_Belgian_lung_cancer.xlsx', 
    >>>                  sheetname = ['response', 'rates'], index_col = 0)
    >>> formatted_data1 = apc.format_data(response = data['response'], 
    >>>                               rate = data['rates'], data_format = 'AP')
    >>> formatted_data2 = apc.format_data(response = data['response'], 
    >>>                                   rate = data['rates'], 
    >>>                                   data_format = 'AP', age1 = 25, 
    >>>                                   per1 = 1955, unit = 5)
    >>> print(formatted_data1.response)
    >>> print(formatted_data2.data_as_vector)
    >>> formatted_data1.data_as_vector.equals(formatted_data2.data_as_vector)
    
    """
   
    # If data_format is supplied this overrides index and column names.
    # If infer_from_dataframe is false, we also build the index and column indices
    # from the remaining inputs. If these are supplied, infer_from_dataframe is set to False.
    
    if (age1 is None) and (per1 is None) and (coh1 is None):
        
        infer_from_dataframe = True
    else:
        infer_from_dataframe = False
    
    if (dose is not None and rate is not None) and not response.equals(rate.divide(dose)):
        
        raise ValueError('\'rate\'  is not \'response\'/\'dose\'. Choose either \'dose\' or \'rate\'') 
    
    if rate is not None and dose is None:
        
        dose = response.divide(rate)
        
    if dose is not None and rate is None:
        
        rate = response.divide(dose)
    
    if data_format is not None:
        
        if data_format not in ["AP", "AC", "CA", "CL", "CP", "PA", "PC", "trapezoid"]:
            
            raise ValueError('\'data_format\' is not permisible.')
        
        nrow, ncol = response.shape
        
        if data_format.startswith('A') or data_format is 'trapezoid':
            
            row_name = 'Age'
            
            if not infer_from_dataframe:
                row_labels = np.arange(age1, age1 + nrow * unit, unit)
            
        if data_format.startswith('P'):
        
            row_name = 'Period'
            
            if not infer_from_dataframe:
                row_labels = np.arange(per1, per1 + nrow * unit, unit)
            
            
        if data_format.startswith('C'):
            
            row_name = 'Cohort'
            
            if not infer_from_dataframe:
                row_labels = np.arange(coh1, coh1 + nrow * unit, unit)
            
        
        if data_format.endswith('A') or data_format is 'CL':
            
            col_name = 'Age'
            
            if not infer_from_dataframe:
                col_labels = np.arange(age1, age1 + ncol * unit, unit)
            
        if data_format.endswith('P'):
        
            col_name = 'Period'
            
            if not infer_from_dataframe:
                col_labels = np.arange(per1, per1 + ncol * unit, unit)
            
            
        if data_format.endswith('C') or data_format is 'trapezoid':
            
            col_name = 'Cohort'
            
            if not infer_from_dataframe:
                col_labels = np.arange(coh1, coh1 + ncol * unit, unit)
            
        if not infer_from_dataframe:
            response.index = row_labels
            response.columns = col_labels
        
            if dose is not None:
                dose.index = row_labels
                dose.columns = col_labels
                rate.index = row_labels
                rate.columns = col_labels
            
            if age1 is not None and per1 is not None and coh1 is not None:
                print('Warning: used only {}1 and {}1 to generate axes labels.'.format(row_name[:3].lower(),
                                                                             col_name[:3].lower()))
            
        response.index.rename(row_name, inplace = True)
        response.columns.rename(col_name, inplace = True)
        
        if dose is not None:
            dose.index.rename(row_name, inplace = True)
            dose.columns.rename(col_name, inplace = True)
            rate.index.rename(row_name, inplace = True)
            rate.columns.rename(col_name, inplace = True)
        
            
    ## With this, we can proceed as if we had a data frame with well formatted index and columns
    
    # Infer the data_format if not supplied. This uses only response df information and adjust the
    # dose and rate df, if not None.
    
    if data_format is None:
        
        if response.index.name is None or response.columns.name is None:
            
            raise ValueError('Need index and column names if no' + 
                             '\'data_format\' is supplied')
        
        index_labels = {'Age' : ['A', 'a', 'age', 'Age'], 
                        'Period' : ['P', 'p', 'period', 'Period'], 
                        'Cohort' : ['C', 'c', 'cohort', 'Cohort']}
        
                
        for used_labels, permisible_labels in index_labels.items():
                        
            if response.index.name in permisible_labels:
                
                data_format_row = used_labels[0]
                response.index.name = used_labels
                
                if dose is not None:
                
                    dose.index.name = used_labels
                    rate.index.name = used_labels
            
            if response.columns.name in permisible_labels:
                
                data_format_col = used_labels[0]
                response.columns.name = used_labels
                
                if dose is not None:
                
                    dose.columns.name = used_labels
                    rate.columns.name = used_labels
        
        if data_format_row is data_format_col:
            raise ValueError('Index name is not permisible. It seems both indices are the same!')
            
        data_format = data_format_row + data_format_col
            
    # Now we can build a df that contains information about the three time scales and their
    # correspondance to response and dose/ rate, if applicable.
    
    response_vector = response.unstack().rename('Response', inplace = True)
    
    if dose is not None:
        
        dose_vector = dose.unstack().rename('Dose', inplace = True)
        
        rate_vector = rate.unstack().rename('Rate', inplace = True)
        
        data_as_vector = pd.concat((response_vector, dose_vector, rate_vector), axis = 1)
        
        #rate_as_vector = response_vector.loc[:,'Response'].divide(dose_vector.loc[:,'Dose'])
        
        #data_as_vector.insert(data_as_vector.shape[1], 'Rate', rate_as_vector, allow_duplicates = True)
    
    else:
        data_as_vector = pd.DataFrame(response_vector)
    
    data_as_vector = data_as_vector.reset_index()
    
    # Potentially reformat the dates now to be able to infer cohorts
    format_labels = lambda x: pd.to_numeric(x.split('-')[0].strip()) if isinstance(x,str) else x
    
    data_as_vector.iloc[:,:2] = data_as_vector.iloc[:,:2].applymap(format_labels)
    
    
    # Need the unit by which the indices increase to set the time adjust. 
    # This should be the same for all indices. Since the third index is a linear combination of the other two, 
    # it is sufficient to check equality of two units.
    
    # We need these indices even if we dont have to get the units
    indices_min = data_as_vector.iloc[:,0:2].min()
    
    if infer_from_dataframe:    
        indices_units = data_as_vector.iloc[:,0:2][data_as_vector.iloc[:,0:2]> indices_min].min().astype(indices_min.dtype) - indices_min
    
        if indices_units[0] == indices_units[1]:
            unit = indices_units[0]
        else:
            raise ValueError('Units are not the same for all time indices')
        
    if time_adjust is None:
        
        if data_format in ['AC', 'CA', 'trapezoid', 'CL'] and min(data_as_vector.loc[:,'Age']) == 1 and min(data_as_vector.loc[:,'Cohort']) == 1:
            
            time_adjust = 1
        
        else:
            
            time_adjust = 0
            
    if data_format in ['AC', 'CA', 'trapezoid', 'CL']:
        
        period_index = data_as_vector.loc[:,'Age'] + data_as_vector.loc[:,'Cohort'] - time_adjust
        
        data_as_vector.insert(1, 'Period', period_index, allow_duplicates=True)
        
    elif data_format in ['AP', 'PA']:
        
        cohort_index = data_as_vector.loc[:,'Period'] - data_as_vector.loc[:,'Age'] + time_adjust
        
        data_as_vector.insert(2, 'Cohort', cohort_index, allow_duplicates=True)
    
    elif data_format in ['CP', 'PC']:
        
        age_index = data_as_vector.loc[:,'Period'] - data_as_vector.loc[:,'Cohort'] + time_adjust
        
        data_as_vector.insert(0, 'Age', age_index, allow_duplicates=True)
    
    else:
        
        raise ValueError('{} is not a permisible \'data_format\'.'.format(data_format))
        
    if data_format in ['AP', 'PA', 'CP', 'PC'] and data_as_vector.isnull().any().any():
        
        raise ValueError('Cannot handle missing data except missing periods in age-cohort space.')
    
    data_as_vector.dropna(inplace = True)
    
    #Set the age-period-cohort columns as MultiIndex and sort by age
    data_as_vector.set_index(['Age','Period','Cohort'], inplace = True)
    data_as_vector.sortlevel('Age', inplace = True)
    
    # Some information about the array, this is needed for the design construction
    
    n_age = len(data_as_vector.index.levels[0])
    n_per = len(data_as_vector.index.levels[1])
    n_coh = len(data_as_vector.index.levels[2])
    n_obs = data_as_vector.shape[0]
    # The next is 'L' in the Kuang et al. (2008a) paper, 
    # this works because we sorted by age. labels[2] are the cohort labels
    n_missing_lower_per = data_as_vector.index.labels[2][0]
    
    if data_format in ['AC', 'CA', 'CL', 'trapezoid']:
        
        if n_missing_lower_per > 0 or (n_age + n_coh > n_missing_lower_per + n_per + 1):
            
            data_format = 'trapezoid'
        
        if (n_missing_lower_per == 0) and (n_age == n_coh == n_per):
            
            data_format = 'CL'

    return _ReturnValue(response = response, dose = dose, rate = rate,  data_format = data_format,
                        data_as_vector = data_as_vector, time_adjust = time_adjust,
                        unit = unit, n_age = n_age, n_per = n_per, n_coh = n_coh, 
                        n_obs = n_obs, n_missing_lower_per = n_missing_lower_per,
                        label = label)