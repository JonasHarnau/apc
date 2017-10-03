import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
import collections

class Model:
    """
    This class is the central object for age-period-cohort modelling.
    
    This is the central object for age-period-cohort modelling. Currently it
    can only hold data, but will be expanded to fit models, plot, etc. Check
    out the help for the methods to see all the things it can do!
    
    """
    def __init__(self, **kwargs):
        # The kwargs can be used for example to attach a label to the model.
        # They should be used with care to prevent overlap with arguments 
        # used in the modelling process.
        if kwargs:
            print('These arguments are attached, but not used in modeling:')
            print(kwargs.keys())
            for key, value in kwargs.items():
                setattr(self, key, value)

    def data_from_df(self, response, dose=None, rate=None, 
                     data_format=None, time_adjust=0):
        """    
        Arranges data into a format useable by the other functions in the package.
        
        This is the first step of the analysis. While usually formatted in a 
        two-way table, age-period-cohort data come in many different formats. Any
        two of age, period and cohort can be on the axes. Further, data may be
        available for responses (death counts, paid amounts) alone, or for both 
        dose (exposure, cases) and response. The main job of this function is to
        keep track of what age, period and cohort goes with which entry. Since
        generally only two of the three time scales are available, the third is
        generated internally.  
        
        Parameters
        ----------
        
        response : pandas.DataFrame
                   Responses formatted in a two-way table. The time scales on 
                   the axes are expected to increase from the top left. For 
                   example, in an age-period ("AP") array, ages increases from
                   top to bottom and period from left to right and similarly for
                   other arrays types.
                   
        dose : pandas.DataFrame, optional
               Numbers of doses. Should have the same format as 'response'. If
               'dose' is supplied, 'rate' are generated internally as 
               'response'/'dose'. (Default is 'None')
        
        rate : pandas.DataFrame, optional
               Rates (or mortality). Should have the same format as 'response'. 
               If 'rate' is supplied, 'dose' are generated internally as 
               'response'/'rate'. (Default is 'None')
        
        data_format : {"AP", "AC", "CA", "CL", "CP", "PA", "PC", "trapezoid"},
                      optional
                      This input is necessary if 'response.index.name' and 
                      'response.column.name' are not supplied. If those are 
                      any two of 'Age', 'Period' and 'Cohort', the 
                      'data_format' is inferred and need not be supplied.
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
                      has age/period as increasing row/column index. Empty
                      periods in triangular shape at the top left and bottom
                      right are allowed.
                             
        time_adjust : int, optional
                      Specifies the relation between age, period and cohort
                      through age + cohort = period + 'time_adjust'. This is
                      used to compute the missing time labels internally. If
                      this is 'None' this is set to 0 internally with one
                      exception: if 'data_type' is any of "AC", "CA", "CL" or
                      "trapezoid" and the minimum age and cohort labels are 1
                      so that the minimum period is also 1.


        Returns
        -------
        
      
        Attaches the following output to the apc.Model().
        
        data_vector : pandas.DataFrame
                      A dataframe of responses and, if applicable, doses and 
                      rates in the columns. The dataframe has MultiIndex with
                      levels 'Age', 'Period' and 'Cohort'.
                         
        I : Int
            the number of ages with available data
        
        J : Int
            the number of periods with available data
        
        K : Int
            the number of cohorts with available data
        
        L : Int
            the number of missing lower periods in age-cohort  space, 
            that is missing diagonals in the top-left.
        
        n : Int 
            the number of data points       
                              
     
        Examples 
        --------
        
        This example makes use of the Belgian lung cancer data spreadsheet that 
        comes with the package. 
        
        >>> import pandas as pd
        >>> data = pd.read_excel('./apc/data/Belgian_lung_cancer.xlsx', 
        ...                      sheetname = ['response', 'rates'], index_col = 0)
        >>> model = Model()
        >>> model.data_from_df(data['response'], rate=data['rates'], 
        ...                    data_format='AP')
        >>> model.I # We have data for I=11 ages,
        11 
        >>> model.K # K=14 cohorts
        14 
        >>> model.J # and 4 periods
        4 
        >>> model.L # with the 10 lowest periods missing in AC space
        10
        >>> print(model.data_vector.head())
                                   response       dose  rate
        Period    Age   Cohort                              
        1955-1959 25-29 1926-1934         3  15.789474  0.19
                  30-34 1921-1929        11  16.666667  0.66
                  35-39 1916-1924        11  14.102564  0.78
                  40-44 1911-1919        36  13.483146  2.67
                  45-49 1906-1914        77  15.909091  4.84
                  
        References
        ----------
        
        Kuang, D., Nielsen, B. and Nielsen, J.P. (2008) Identification of the 
        age-period-cohort model and the extended chain ladder model. 
        Biometrika 95, 979-986. 
        
        """
        
        ## Checking inputs
        supported_data_formats = ("AP", "AC", "CA", "CL", "CP", "PA", "PC", 
                                  "trapezoid", None)
        if data_format not in supported_data_formats:
            raise ValueError("\'data_format\' not understood. Check the help.")
        if (response is not None) and not isinstance(response, pd.DataFrame):
            raise ValueError("\'response\' must be pandas DataFrame.")
        if (dose is not None) and not isinstance(dose, pd.DataFrame):
            raise ValueError("\'dose\' must be pandas DataFrame.")
        if (rate is not None) and not isinstance(rate, pd.DataFrame):
            raise ValueError("\'rate\' must be pandas DataFrame.")
        if ((rate is not None) and (dose is not None) 
            and not response.equals(rate.divide(dose))):
            raise ValueError('\'rate\' must be \'response\'/\'dose\'.')
        
        ## Inferring from inputs
        def _infer_data_format(response):
            if not (response.index.name and response.columns.name):
                raise ValueError('Need index and column label names if '
                                 '\'data_format\' is not supplied.')
            
            supported_idx_names = {
                ('a', 'age', 'development year', 'underwriting year'): 'A', 
                ('p', 'period', 'calendar year'): 'P', 
                ('c', 'cohort', 'accident year'): 'C'
            }
            for idx_name, short in supported_idx_names.items():
                if response.index.name.lower() in idx_name:
                    data_format_row = short
                if response.columns.name.lower() in idx_name:
                    data_format_col = short
            data_format = data_format_row + data_format_col
            if data_format in ("AP", "AC", "CA", "CL", "CP", "PA", "PC"):
                return data_format
            else:
                raise ValueError('Couldn\'t infer valid \'data_format\'.')

        if not data_format:
            data_format = _infer_data_format(response)
            print("Inferred \'data_format\' from response: {}".format(
                data_format))
        self.data_format = data_format
        
        if rate is None and dose is None:
            data_vector = pd.DataFrame(response.unstack().rename('response'))
        else: 
            if dose is None:
                dose = response.divide(rate)
            elif rate is None:
                rate = response.divide(dose)
            else:
                if not response.equals(rate.divide(dose)):
                    raise ValueError('\'rate\' must be \'response\'/\'dose\'.')
            data_vector = pd.concat([response.unstack().rename('response'), 
                                     dose.unstack().rename('dose'), 
                                     rate.unstack().rename('rate')], 
                                    axis=1)

        if not len(data_vector) == response.size:
            raise ValueError('Label mismatch between response and dose/rate.')
        
        def _get_index_names(data_format):
            if data_format == 'CL':
                data_format = 'CA'
            elif data_format == 'trapezoid':
                data_format = 'AC'
            relation_dict = {'A': 'Age', 'P': 'Period', 'C': 'Cohort'}
            row_index_name = relation_dict[data_format[0]]
            col_index_name = relation_dict[data_format[1]]
            return (col_index_name, row_index_name)
        
        data_vector.index.names = _get_index_names(self.data_format)
        
        def _append_third_index(data_vector, data_format, time_adjust):
            """
            Adds the third time-scale to the data_vector. Can currently handle
            integer valued pure dates (e.g. 1 or 2017) or ranges (2010-2015)
            """
            index = data_vector.index
            
            if data_format in ('AC', 'CA', 'trapezoid', 'CL'):
                age_idx = (pd.Series(index.get_level_values(level = 'Age')).
                           # The next row helps to deal with ranges
                           astype(str).str.split('-', expand=True).astype(int))
                coh_idx = (pd.Series(index.get_level_values(level = 'Cohort')).
                           astype(str).str.split('-', expand=True).astype(int))
                per_idx = ((age_idx + coh_idx.values - time_adjust).astype(str).
                           # Converts back to ranges if needed
                           apply(lambda x: '-'.join(x), axis=1))
                per_idx.name = 'Period'
                data_vector = data_vector.set_index(per_idx, append=True)
            elif data_format in ('AP', 'PA'):
                age_idx = (pd.Series(index.get_level_values(level = 'Age')).
                           astype(str).str.split('-', expand=True).astype(int))
                per_idx = (pd.Series(index.get_level_values(level = 'Period')).
                           astype(str).str.split('-', expand=True).astype(int))
                coh_idx = ((per_idx - age_idx.loc[:,::-1].values + time_adjust).
                           astype(str).apply(lambda x: '-'.join(x), axis=1))
                coh_idx.name = 'Cohort'
                data_vector = data_vector.set_index(coh_idx, append=True)
            else:
                per_idx = (pd.Series(index.get_level_values(level = 'Period')).
                           astype(str).str.split('-', expand=True).astype(int))
                coh_idx = (pd.Series(index.get_level_values(level = 'Cohort')).
                           astype(str).str.split('-', expand=True).astype(int))
                age_idx = ((per_idx - coh_idx.values + time_adjust).astype(str).
                          apply(lambda x: '-'.join(x), axis=1))
                age_idx.name = 'Age'
                data_vector = data_vector.set_index(age_idx, append=True)
            
            return data_vector
        
        data_vector = _append_third_index(data_vector, data_format, time_adjust)
        
        def _set_trapezoid_information(self, data_vector):
            """
            Check if data_vector fits generalized trapezoid structure and get the
            defining trapezoid characteristics.
            """
            data_vector = data_vector.dropna()
            
            ac_array = data_vector.reset_index().pivot(
                index='Age', columns='Cohort', values='Cohort').isnull()
            ap_array = data_vector.reset_index().pivot(
                index='Age', columns='Period', values='Cohort').isnull()
            
            # Assumes we have no fully empty columns or rows.
            I, K = ac_array.shape
            J = ap_array.shape[1]
            # 'L' is # missing lower period in age-cohort space (Kuang et al. 2008). 
            L = ac_array.iloc[0,:].sum()
            n = len(data_vector)
            
            # Check if the data are a generalized trapezoid.
            
            # Get the equivalent of L for the missing periods at the bottom right.
            bottom_L = ac_array.iloc[-1,:].sum()
            
            # Generate a mask that we match to the generalized trapezoid structure
            # of the data, holding True for cells that should contain data and false
            # otherwise.
            equiv_trapezoid = ac_array.copy()
            equiv_trapezoid[:] = False
            
            for i in range(I):
                equiv_trapezoid.iloc[i,:max(0,L-i)] = True
                equiv_trapezoid.iloc[I-i-1,K-bottom_L+i:] = True
            
            if not ac_array.equals(equiv_trapezoid):
                raise ValueError('Data are not in generalized trapezoid form.')
            
            self.I = I
            self.K = K
            self.J = J
            self.L = L
            self.n = n
            self.data_vector = data_vector          
                
        _set_trapezoid_information(self, data_vector)