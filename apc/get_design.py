import pandas as pd
from .get_design_components import get_design_components

def get_design(formatted_data, model_design, design_components = None):
    
    """
        
    Creates a design matrix.
    
    This function creates the design for the canonical parameters as described
    in Nielsen (2014, 2015) which generalises introduced by Kuang, Nielsen and
    Nielsen (2008). In normal use this function are needed for internal use by
    'fit_model' and 'fit_table'.

    
    Parameters
    ----------
    
    formatted_data : output of 'apc.format_data'
    
    model_design : {'APC', 'AP', 'AC', 'PC', 'Ad', 'Pd', 'Cd', 'A', 'P', 'C',
                    't', 'tA', 'tP', 'tC', '1'}
                   Indicates the design choice. The following options are
                   available. These are discussed in Nielsen (2014).
                   "APC"
                       Age-period-cohort model.
                   "AP"
                       Age-period model. Nested in "APC"
                   "AC"
                       Age-cohort model. Nested in "APC"
                   "PC"
                       Period-cohort model. Nested in "APC"
                   "Ad"
                       Age-trend model, including age effect and two linear
                       trends. Nested in "AP", "AC".
                   "Pd"
                       Period-trend model, including period effect and two
                       linear trends. Nested in "AP", "PC".
                   "Cd"
                       Cohort-trend model, including cohort effect and two
                       linear trends. Nested in "AC", "PC".
                   "A"
                       Age model. Nested in "Ad".
                   "P"
                       Period model. Nested in "Pd".
                   "C"
                       Cohort model. Nested in "Cd".
                   "t"
                       Trend model, with two linear trends. Nested in "Ad",
                       "Pd", "Cd".
                   "tA"
                       Single trend model in age index. Nested in "A", "t".
                   "tP"
                       Single trend model in period index. Nested in "P", "t".
                   "tC"
                       Single trend model in cohort index. Nested in "C", "t".
                   "1"
                       Constant model. Nested in "tA", "tP", "tC".
                
    design_components : output from 'apc.get_design_components', optional
                        If this is not supplied it is computed internally. For
                        simulations it may be useful to supply this so it does
                        not have to be computed every time.
                    
    
    Returns
    -------
    
    design : pandas.DataFrame
    
    
    Notes
    -----
    
    The description is largely taken from the R package apc.
    
    
    See also
    --------
    
    apc.get_design_components : Generates level, slopes and double difference
                                components. 'apc.get_design' selects those
                                appropriate for the specific model at hand.
    
    
    Examples
    --------
    
    >>> import apc
    >>> data = apc.data_Italian_bladder_cancer()
    >>> AC_design = apc.get_design(data, 'AC')
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
    
    model_design_list = ['APC', 'AP', 'AC', 'PC', 'Ad', 'Pd', 
        'Cd', 'A', 'P', 'C', 't', 'tA', 'tP', 'tC', '1']
    
    if model_design not in model_design_list:
        raise ValueError('\'model_design\' has argument \'' + 
                         model_design + '\' which is not allowed')
    
    if design_components is None:
        design_components = get_design_components(formatted_data)
    
    level = design_components.level
    slopes = design_components.slopes
    double_diffs = design_components.double_diffs
    
    design = pd.concat((level,
                        slopes['Age'] if model_design in 
                        ['APC', 'AP', 'AC', 'PC', 'Ad', 'Pd', 'Cd', 'A', 't', 'tA'] 
                        else None,
                        slopes['Period'] if model_design in 
                        ['P','tP'] 
                        else None,
                        slopes['Cohort'] if model_design in 
                        ['APC', 'AP', 'AC', 'PC', 'Ad', 'Pd', 'Cd', 'C', 't', 'tC'] 
                        else None,
                        double_diffs['Age'] if model_design in 
                        ['APC', 'AP', 'AC', 'Ad', 'A'] 
                        else None,
                        double_diffs['Period'] if model_design in 
                        ['APC', 'AP', 'PC', 'Pd', 'P'] 
                        else None,
                        double_diffs['Cohort'] if model_design in 
                        ['APC', 'AC', 'PC', 'Cd', 'C'] 
                        else None),
                      axis = 1)
    
    design.index = formatted_data.data_as_vector.index
    
    return design