import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
import collections
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

class Model:
    """
    This class is the central object for age-period-cohort modelling.
    
    This is the central object for age-period-cohort modelling. It can format
    and plot data, fit models and deviance tables, and simulate from a fitted
    model (currently only Poisson, over-dispersed Poisson, and log-normal). 
    We can also obtain sub-samples from the data (this is not yet well 
    implemented.) For more information, check out the README!
    
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
        self.plot_fit = self.plot_parameters

    def data_from_df(self, response, dose=None, rate=None, 
                     data_format=None, time_adjust=None):
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
                      exception: if the minimum of both supplied labels is 1, 
                      'time_adjust' is set to 1 so that that all three labels
                      start at 1.


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
        ...                      sheet_name = ['response', 'rates'], index_col = 0)
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

        try: # try to convert to integers, useful if integers loaded as strings
            response.columns = response.columns.astype(int)
            response.index = response.index.astype(int)
        except TypeError:
            pass
        
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

        if time_adjust is None:
            if data_vector.index.min() == (1,1):
                time_adjust = 1
            else:
                time_adjust = 0

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
            integer valued pure dates (e.g. 1 or 2017) or ranges (2010-2015).
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
                try:
                    per_idx = per_idx.astype(int)
                except ValueError:
                    pass
                data_vector = data_vector.set_index(per_idx, append=True)
            elif data_format in ('AP', 'PA'):
                age_idx = (pd.Series(index.get_level_values(level = 'Age')).
                           astype(str).str.split('-', expand=True).astype(int))
                per_idx = (pd.Series(index.get_level_values(level = 'Period')).
                           astype(str).str.split('-', expand=True).astype(int))
                coh_idx = ((per_idx - age_idx.loc[:,::-1].values + time_adjust).
                           astype(str).apply(lambda x: '-'.join(x), axis=1))
                coh_idx.name = 'Cohort'
                try:
                    coh_idx = coh_idx.astype(int)
                except ValueError:
                    pass
                data_vector = data_vector.set_index(coh_idx, append=True)
            else:
                per_idx = (pd.Series(index.get_level_values(level = 'Period')).
                           astype(str).str.split('-', expand=True).astype(int))
                coh_idx = (pd.Series(index.get_level_values(level = 'Cohort')).
                           astype(str).str.split('-', expand=True).astype(int))
                age_idx = ((per_idx - coh_idx.values + time_adjust).astype(str).
                          apply(lambda x: '-'.join(x), axis=1))
                age_idx.name = 'Age'
                try:
                    age_idx = age_idx.astype(int)
                except ValueError:
                    pass
                data_vector = data_vector.set_index(age_idx, append=True)
            
            return data_vector
        
        data_vector = _append_third_index(data_vector, data_format, time_adjust)
        
        def _set_trapezoid_information(self, data_vector):
            """
            Check if data_vector fits generalized trapezoid structure and get the
            defining trapezoid characteristics.
            """
            data_vector = data_vector.dropna()
            # Because dropna does not remove old index labels we have to execute this:
            data_vector = data_vector.reset_index().set_index(data_vector.index.names)
            
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
            # of the data, holding True for cells that should not contain data and 
            # false otherwise.
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
            self.time_adjust = time_adjust
                
        _set_trapezoid_information(self, data_vector)
        
    
    def _get_design_components(self):
        """
        Internal function  that creates the design components.
        
        This function creates the 'level', 'slopes' and 'double_diffs'. Combined
        this would yield a collinear design. The theory is described in Nielsen
        (2014, 2015) which generalises introduced by Kuang, Nielsen and Nielsen
        (2008). In normal use this function is needed for internal use by
        'get_design'.
        
        
        Parameters
        ----------
        
        Extracts necessary information from apc.Model().data_as_df(). Specifically, 
        it uses the trapezoid information I, K, J, L, n and 'data_vector'.
        
        Returns
        -------
        
        Dictionary attached to self
        The dictionary contains the following.
        
        anchor_index : int
                       This is what Nielsen (2014, 2015) defines as 'U'. (U,U)
                       is the array index in age-cohort space that contains only
                       the level, starting to count from 0. 
        
        per_odd : bool
                  This is 'True' if the period before the first observed period
                  has an odd index, starting to count from 1 for age, period and
                  cohort.
        
        level : pandas.Series
                Unit vector of length equal to the number of obeservations.
                
        slopes : dict
                 Keys are 'Age', 'Period' and 'Cohort', values are the designs
                 for the accompanying slopes.
        
        double_diffs : dict
                       Keys are 'Age', 'Period' and 'Cohort', values are the 
                       design for the accompanying slopes.
            
        
        Notes
        -----
        
        The description is largely taken from the R package apc.
        
        
        See also
        --------
        
        apc.Model().fit : This fits a model and either calls or, if specified
                          as input, uses _get_design_components.
                          
        
        Examples
        --------
        
        >>> import pandas as pd
        >>> data = pd.read_excel('./data/Belgian_lung_cancer.xlsx', 
        ...                      sheet_name = ['response', 'rates'], index_col = 0)
        >>> import apc
        >>> model = apc.Model()
        >>> model.data_from_df(data['response'], rate=data['rates'], 
        ...                    data_format='AP')
        
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
        
        # Get the trapezoid indices with internal counting (0, 1, 2, ...), 
        # rather than counting by labels (e.g. 1955-1959, 1960-1964, ...).
        try: 
            index = self.data_vector.reorder_levels(['Age', 'Period', 'Cohort']).index
        except AttributeError:
            raise AttributeError("'data_vector' not found, run 'data_from_df' first.")
        index_levels_age = index.levels[0]
        index_levels_per = index.levels[1]
        index_levels_coh = index.levels[2]
        index_trap = (pd.DataFrame(index.labels, index = index.names).T.
                      loc[:,['Age','Cohort']])
        
        I = self.I
        K = self.K
        J = self.J
        L = self.L
        n = self.n
        
        anchor_index = int((L + 3)/2 -1)
        per_odd = False if L % 2 == 0 else True
        
        level = pd.Series(1, index = range(n), name = 'level')
        
        slope_age = index_trap['Age'] - anchor_index
        slope_age.rename('slope_age', inplace = True)
        slope_coh = index_trap['Cohort'] - anchor_index
        slope_coh.rename('slope_coh', inplace = True)
        slope_per = slope_age + slope_coh
        slope_per.rename('slope_per', inplace = True)
        
        dd_age_col = ['dd_age_{}'.format(age) for age in index_levels_age[2:]]
        dd_age = pd.DataFrame(0, index = range(n), columns = dd_age_col)
        
        for i in range(anchor_index):
            dd_age.loc[slope_age == - anchor_index + i, i:anchor_index] = (
                np.arange(1, anchor_index - i + 1))
        
        for i in range(1, I - anchor_index):
            dd_age.loc[slope_age == i + 1, anchor_index:anchor_index + i] = (
                np.arange(i, 0, -1))
        
        dd_per_col = ['dd_per_{}'.format(per) for per in index_levels_per[2:]]
        dd_per = pd.DataFrame(0, index = range(n), columns = dd_per_col)
        
        if per_odd:
            dd_per.loc[slope_per == -1,0:1] = 1
        
        for j in range(J - 2 - per_odd):
            dd_per.loc[slope_per == j + 2, int(per_odd):j + 1 + int(per_odd)] = (
                np.arange(j + 1, 0, -1))
        
        dd_coh_col = ['dd_coh_{}'.format(coh) for coh in index_levels_coh[2:]]
        dd_coh = pd.DataFrame(0, index = range(n), columns = dd_coh_col)
        
        
        for k in range(anchor_index):
            dd_coh.loc[slope_coh == - anchor_index + k, k:anchor_index] = (
                np.arange(1, anchor_index - k + 1))
        
        for k in range(1, K - anchor_index):
            dd_coh.loc[slope_coh == k + 1, anchor_index:anchor_index + k] = (
                np.arange(k, 0, -1))
        
        design_components = {
            'index': self.data_vector.index,
            'anchor_index': anchor_index, 
            'per_odd': per_odd,
            'level': level,
            'slopes': {'Age' : slope_age, 
                       'Period' : slope_per, 
                       'Cohort' : slope_coh},
            'double_diffs': {'Age' : dd_age, 
                             'Period' : dd_per, 
                             'Cohort' : dd_coh}
               }
        
        self._design_components = design_components
    
    def _get_design(self, predictor, design_components=None):
        """
        Takes _design_components and builds design matrix for predictor.
        """
    
        if design_components is None:
            self._get_design_components()
            design_components = self._design_components
        
        level = design_components['level']
        slopes = design_components['slopes']
        double_diffs = design_components['double_diffs']
        
        design = pd.concat(
            (
                level,
                slopes['Age'] if predictor in 
                ('APC', 'AP', 'AC', 'PC', 'Ad', 'Pd', 'Cd', 'A', 't', 'tA')
                else None,
                slopes['Period'] if predictor in 
                ('P','tP')
                else None,
                slopes['Cohort'] if predictor in 
                ('APC', 'AP', 'AC', 'PC', 'Ad', 'Pd', 'Cd', 'C', 't', 'tC')
                else None,
                double_diffs['Age'] if predictor in 
                ('APC', 'AP', 'AC', 'Ad', 'A') 
                else None,
                double_diffs['Period'] if predictor in 
                ('APC', 'AP', 'PC', 'Pd', 'P')
                else None,
                double_diffs['Cohort'] if predictor in 
                ('APC', 'AC', 'PC', 'Cd', 'C') 
                else None
            ),
            axis = 1)
        
        design.index = design_components['index']
        
        return design
        
    def fit(self, family, predictor, design_components=None, R=None, attach_to_self=True):
        """
        Fits an age-period-cohort model to the data from Model().data_from_df().
    
        The model is parametrised in terms of the canonical parameter introduced by
        Kuang, Nielsen and Nielsen (2008), see also the implementation in Martinez
        Miranda, Nielsen and Nielsen (2015), and Nielsen (2014, 2015). This
        parametrisation has a number of advantages: it is freely varying, it is the
        canonical parameter of a regular exponential family, and it is invariant to
        extentions of the data matrix.
    
        'fit' can be be used for all three age period cohort factors, or for
        submodels with fewer of these factors. It can be used in a pure response 
        setting or in a dose-response setting. It can handle binomial, Gaussian, 
        log-normal, over-dispersed Poisson and Poisson models.
        
        
        Parameters
        ----------
    
        family : {"binomial_dose_response", "poisson_dose_response", 
                  "poisson_response", "od_poisson_response", 
                  "gaussian_rates", "gaussian_response", 
                  "log_normal_rates", "log_normal_response",
                  "gen_log_normal_response"}
                  
                  "poisson_response"
                      Poisson family with log link. Only responses are used. Inference is done
                      in a multinomial model, conditioning on the overall level as documented
                      in Martinez Miranda et al. (2015).
                  "poisson_dose_response"
                      Poisson family with log link and doses as offset. Limiting distributions
                      are normal and chi2.
                  "od_poisson_response"
                      Poisson family with log link. Only responses are used. Inference is done
                      in an over-dispersed Poisson model as documented in Harnau and Nielsen
                      (2017). Limiting distributions are t and F.
                  "binomial_dose_response"
                      Binomial family with logit link. Gives a logistic regression. Limiting 
                      distributions are normal and chi2.
                  "gaussian_rates"
                      Gaussian family with identity link. The dependent variable are rates.
                      Limiting distributions are t and F.
                  "gaussian_response"
                      Gaussian family with identity link. Gives a regression on the responses.
                      Limiting distributions are t and F.
                  "log_normal_response"
                      Gaussian family with identity link. Dependent variable are log responses.
                      Limiting distributions are t and F.
                  "log_normal_rates"
                      Gaussian family with identity link. Dependent variable are log rates.
                      Limiting distributions are t and F.
                  "gen_log_normal_response"
                      Gaussian family with identity link. Dependent variable are log responses.
                      Limiting distributions are t and F as shown in Kuang and Nielsen (2018).
        
        predictor : {'APC', 'AP', 'AC', 'PC', 'Ad', 'Pd', 'Cd', 'A', 'P', 'C','t', 'tA', 'tP',
                     'tC', '1'}
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
        
        R : pandas.DataFrame, optional
            Restriction on the design matrix implicitly specified by 'predictor'. Internally calls
            design = design.dot(R). Thus, R has to have as many rows as Model.design. Only intended
            to be used with Model.fit. Use breaks many other functions. Then, further analysis such 
            as parameter plots, would have to be done by hand.
        
        design_components : pandas.DataFrame, optional
                            Output of Model()._get_design_components(). Can speed up computations 
                            if 'fit' is called repeatedly. If not provided computed internally.
                            
        attach_to_self : bool, optional
                         Default True. If this is True the results are attached self.If False 
                         the results are returned as a dictionary. (Default is True.)
                         
        Returns
        -------
        
        The following variables attached to self.
        
        aic : float (only for Gaussian and log-normal models)
              The Akaike information criterion.
        
        cov_canonical : pandas.DataFrame
                        Normalized covariance matrix. See note below.
        
        design : pandas.DataFrame
                 The design used to fit the model. Generated by apc.Model()._get_design().
        
        deviance : float
                   Corresponds to the deviance of 'fit.deviance', except for Gaussian
                   and log-normal models where it is - 2 * log-likelihood, rather 
                   than RSS.
        
        df_resid : int
                   The degrees of freedom.
        
        family : str
                 The specified model family.
        
        fitted_values : pandas.Series
                        Fitted values for the response.
        
        parameters : pandas.DataFrame
                     Dataframe with four columns: coefficients, standard errors,
                     z-stats/t-stats (ratio of coefficients to standard errors) and 
                     p-values. 
        
        predictor : str
                    The specified predictor.
        
        residuals : dict
                    Dictionary containing four forms of residuals generated by fitting
                    with statsmodels: 'anscombe', 'pearson', 'deviance' and 'response'.
        
        rss : float (only for Gaussian and log-normal models)
              Sum of squared residuals, on the log-scale for log-nromal models.
        
        s2 : float (only for Gaussian, log-normal, and over-dispersed Poisson models)
             For Gaussian and log-normal models this is 'rss/df_resid'. For 
             over-dispersed Poisson it is 'deviance/df_resid'.
        
        sigma2 : float (only for Gaussian and log-normal models)
                 Maximum likelihood normal variance estimator 'rss / n'.
        
        
        Notes
        -----
        
        'cov_canonical' generally equals 'fit.normalized_cov_params', except for Poisson 
        and over-dispersed Poisson response models when it is adjusted to a multinomial 
        covariance; see Martinez Miranda et al. (2015) and Harnau and Nielsen (2017). For
        over-dispersed Poisson models, the adjustment also includes multiplication by s2.
        
        'deviance' for Gaussian and log-normal models equals - 2 * log-likelihood, 
        not RSS.
            
        p-values for 'coefs_canonical_table' are generally computed from a normal
        distribution. The exception is an over-dispersed Poisson model for which these
        come from a t distribution; see Harnau and Nielsen (2017).
        
        The description is largely taken from the R package apc.
        
        
        References
        ----------
        
        Harnau, J. and Nielsen, B. (2017) Asymptotic theory for over-dispersed 
        age-period-cohort and extended chain ladder models. To appear in Journal
        of the American Statistical Association.
        
        Kuang, D., Nielsen, B. and Nielsen, J.P. (2008) Identification of the 
        age-period-cohort model and the extended chain ladder model. Biometrika 
        95, 979-986. 
        
        Kuang, D., & Nielsen, B. (2018). Generalized Log-Normal Chain-Ladder. 
        ArXiv E-Prints, 1806.05939.
        
        Martinez Miranda, M.D., Nielsen, B. and Nielsen, J.P. (2015) Inference 
        and forecasting in the age-period-cohort model with unknown exposure 
        with an application to mesothelioma mortality. Journal of the Royal 
        Statistical Society A 178, 29-55.
        
        Nielsen, B. (2014) Deviance analysis of age-period-cohort models. 
        Nuffield Discussion Paper 2014-W03
        
        Nielsen, B. (2015) apc: An R package for age-period-cohort analysis. 
        R Journal 7, 52-64.
        
        
        Examples
        --------
        
        >>> model = apc.Model()
        >>> model.data_from_df(**apc.Belgian_lung_cancer())
        >>> model.fit('poisson_dose_response', 'Ad')
        
        >>> model = apc.Model()
        >>> model.data_from_df(apc.loss_TA(), data_format='CL')
        >>> model.fit('od_poisson_response', 'AC')
        
        """
        
        ## Model family
        supported_families = ('binomial_dose_response', 'poisson_dose_response',
                              'poisson_response', 'od_poisson_response',
                              'gaussian_rates', 'gaussian_response', 
                              'log_normal_rates', 'log_normal_response',
                              'gen_log_normal_response')
        if family not in supported_families:
            raise ValueError('"family" not understood. Check the help.')
        
        ## Model predictor
        supported_predictors = ('APC', 'AP', 'AC', 'PC', 'Ad', 'Pd', 'Cd', 'A', 'P', 'C', 
                                't', 'tA', 'tP', 'tC', '1')
        if predictor not in supported_predictors:
            raise ValueError('"predictor" not understood. Check the help.')
            
        # Get the design matrix
        design = self._get_design(predictor, design_components)
        if R is not None:
            design = design.dot(R)
            
        # Get the data
        response = self.data_vector['response']        
        if family is 'binomial_dose_response':
            dose = self.data_vector['dose']
        elif family in ('gaussian_rates', 'log_normal_rates'):
            rate = self.data_vector['rate']
        elif family in ('poisson_dose_response'):
            offset = np.log(self.data_vector['dose'])
        
        if family is 'binomial_dose_response':
            glm = sm.GLM(pd.concat((response, dose-response), axis=1), design, 
                         sm.families.Binomial())
        elif family == 'poisson_dose_response':
            glm = sm.GLM(response, design, sm.families.Poisson(), offset)
        elif family in ('poisson_response', 'od_poisson_response'):
            glm = sm.GLM(response, design, sm.families.Poisson())        
        elif family is 'gaussian_response':
            glm = sm.GLM(response, design, sm.families.Gaussian())
        elif family is 'gaussian_rates':
            glm = sm.GLM(rate, design, sm.families.Gaussian())
        elif family in ('log_normal_response', 'gen_log_normal_response'):
            glm = sm.GLM(np.log(response), design, 
                         sm.families.Gaussian())
        elif family is 'log_normal_rates':
            glm = sm.GLM(np.log(rate), design, sm.families.Gaussian())

        fit = glm.fit()

        df_resid = fit.df_resid
        fitted_values = fit.fittedvalues
        if family in ('log_normal_rates', 'log_normal_response', 'gen_log_normal_response'):
            fitted_values = np.exp(fitted_values)
        deviance = fit.deviance
        if not family in ('binomial_dose_response', 'poisson_response', 
                          'poisson_dose_response'):
            s2 = fit.deviance / df_resid
        if family in ('gaussian_rates', 'gaussian_response', 
                      'log_normal_rates', 'log_normal_response', 'gen_log_normal_response'):
            rss = fit.deviance
            sigma2 = rss / self.n
            deviance = self.n * (1 + np.log(2 * np.pi) + np.log(sigma2))
            aic = fit.aic
        residuals = {'anscombe': fit.resid_anscombe.rename('Anscombe Residuals'),
                     'pearson': fit.resid_pearson.rename('Pearson Residuals'),
                     'deviance': fit.resid_deviance.rename('Deviance Residuals'),
                     'response': fit.resid_response.rename('Response Residuals')}
        
        cov_canonical = fit.cov_params()
        if family == 'od_poisson_response':
            cov_canonical *= s2
        if family in ('poisson_response', 'od_poisson_response'):
            # Adjust covariance matrix
            cov_canonical.iloc[0,:] = 0
            cov_canonical.iloc[:,0] = 0
        
        # Generate parameter table
        coef = fit.params.rename('coef')
        std_err = fit.bse.rename('std_err')
        t_stat = fit.tvalues.rename('t')
        if family in ('poisson_response', 'poisson_dose_response', 'binomial_dose_response'):
            t_stat.rename('z', inplace=True)
        if family == 'od_poisson_response':
            std_err *= np.sqrt(s2)
            t_stat /= np.sqrt(s2)
        if family in ('poisson_response', 'poisson_dose_response', 'binomial_dose_response'):
            # normal p-values for families with fixed scale
            p_value = fit.pvalues.rename('P>|z|')
        else:
            # t p-values for families with estimated scale
            p_value = 2*pd.Series(1-stats.t.cdf(t_stat.abs(), df_resid), 
                                t_stat.index, name='P>|t|')

        table = pd.concat([coef, std_err, t_stat, p_value], axis=1)

        if family in ('poisson_response', 'od_poisson_response'):
            table.iloc[0, 1:] = np.nan

        output = {'parameters': table, 'df_resid': df_resid, 'predictor': predictor,
                  'family': family, 'design': design, 'deviance': deviance,  
                  'fitted_values': fitted_values, 'cov_canonical': cov_canonical, 
                  'residuals': residuals}

        try:
            output['s2'] = s2
            output['rss'] = rss
            output['sigma2'] = sigma2
            output['aic'] = aic
        except NameError:
            pass
        
        import warnings
        warnings.warn('From the next release, attributes "RSS" is going to be available ' +
                      'only as "rss" and "para_table" only as "parameters".', FutureWarning)
        # removed in next version
        try: 
            output['RSS'] = rss 
        except NameError: 
            pass
        output['para_table'] = table.rename(columns={'std_err': 'std err'})
        
        if attach_to_self:
            for key, value in output.items():
                setattr(self, key, value)
        else:
            return output
    
    def fit_table(self, family=None, reference_predictor=None, design_components=None, 
                  attach_to_self=True):
        """
        Produces a deviance table.
        
        'fit_table' produces a deviance table for up to 15 combinations of
        the three factors and linear trends: "APC", "AP", "AC", "PC", "Ad",
        "Pd", "Cd", "A", "P", "C", "t", "tA", "tP", "tC", "1"; see 
        Nielsen (2014) for a discussion. 'fit_table' allows to specify a 
        'reference_predictor'. The deviance table includes all models nested
        from that point onwards. Nielsen (2015) who elaborates on the 
        equivalent function of the R package apc.
        
        Parameters
        ----------
        
        family : {"binomial_dose_response", "poisson_dose_response", 
                  "poisson_response", "od_poisson_response", 
                  "gaussian_rates", "gaussian_response", 
                  "log_normal_rates", "log_normal_response",
                  "gen_log_normal_response"}, optional
                 See help for Model().fit() for a description. If not specified attempts to 
                 read from self.
        
        model_design : see help for Model().fit() for a description (optional)
                       If not specified attempts to read from self.
        
        reference_predictor : {"APC", "AP", "AC", "PC", "Ad", "Pd", "Cd", 
                               "A", "P", "C", "t", "tA", "tP", "tC"}, optional
                              The 'reference_predictor' determines the baseline relative to 
                              which the deviance table is computed. All sub-models nested in 
                              the 'reference_predictor' are considered. If left empty, this is
                              set to 'Model().predictor' so the table would be computed
                              relative to the last fitted model. If this is empty as well, it 
                              is set to 'APC'.

        design_components : pandas.DataFrame, optional
                            Output of Model()._get_design_components(). Can speed up 
                            computations if 'fit_table' is called repeatedly. If not provided 
                            this is computed internally.
                            
        attach_to_self : bool, optional
                         Default True. If this is True the deviance_table is attached to 
                         self.deviance_table. If False the table is returned. (Default is True.)
        
        Returns
        -------
        
        pandas.DataFrame
        Either attached to self.deviance_table or returned.
        
        The dataframe has the following columns.
        
        "-2logL" or "deviance"
            -2 log Likelihood up to some constant. If the model family is 
            Poisson or binomial (logistic) this is the same as the glm 
            deviance: That is the difference in -2 log likelihood value between
            estimated model and the saturated model. If the model family is 
            Gaussian it is different from the traditional glm deviance. Here 
            -2 log likelihood value is measured in a model with unknown variance,
            which is the standard in regression analysis, whereas in the glm 
            package the deviance is the residual sum of squares, which can be 
            interpreted as the -2 log likelihood value in a model with variance 
            set to one.
            
        "df_resid"
            Degrees of freedom of residual: n_obs - len(parameter). 
            
        "P>chi_sq" 
            Not included for Gaussian or log-normal families. P-value of the 
            deviance, -2logL, or the log-likelihood ratio, "LR_vs_{ref}", compared
            to a chi-square. 

        "P_exact" 
            Only included for Gaussian or log-normal families. Exact p-value 
            of the log-likelihood ratio, "LR_vs_{ref}".

        "LR_vs_{ref}"
            The likelihood ratio statistic against the reference model.
            
        "df_vs_{ref}"
            Degrees of freedom against the reference model.
        
        "aic"
            Akaike's "An Information Criterion", minus twice the maximized 
            log-likelihood plus twice the number of parameters upto a constant. 
            It is take directly from the glm function. 
            
        "F"
            Only included for "od_poisson_response". F statistic: Ratio of 
            'LR_vs_{ref} / df_vs_{ref}' to '-2logL / df_resid'
            
        "P>F"
           Only included for "od_poisson_response". p-value of "F" statistic 
           compared to F-distribution. 
        
        
        Examples
        --------
        
        >>> from apc.Model import Model
        >>> from apc.data.pre_formatted import loss_TA
        >>> model = Model()
        >>> model.data_from_df(loss_TA())
        >>> model.fit_table('od_poisson_response', 'APC')
        
        
        References
        ----------
        
        Harnau, J. and Nielsen, B. (2017) Asymptotic theory for over-dispersed 
        age-period-cohort and extended chain ladder models. To appear in Journal
        of the American Statistical Association.
        
        Nielsen, B. (2015) apc: An R package for age-period-cohort analysis. 
        R Journal 7, 52-64.
        
        """
    
        if design_components is None:
            self._get_design_components()
            design_components = self._design_components
        
        if family is None:
            try:
                family = self.family
            except AttributeError:
                raise AttributeError("Could not infer 'family'. Either specify as " +
                                     "input or fit a model first.")
            
        if reference_predictor is None:
            try:
                reference_predictor = self.predictor
            except AttributeError:
                reference_predictor= 'APC'
        
        if reference_predictor is "APC": 
            sub_predictors = ["AP", "AC", "PC", "Ad", "Pd", "Cd", "A", "P", "C", 
                              "t", "tA", "tP", "tC", "1"]
        elif reference_predictor is "AP":
            sub_predictors = ["Ad", "Pd", "A", "P", "t", "tA", "tP", "1"]
        elif reference_predictor is "AC": 
            sub_predictors = ["Ad", "Cd", "A", "C", "t", "tA", "tC", "1"]
        elif reference_predictor is "PC": 
            sub_predictors = ["Pd", "Cd", "P", "C", "t", "tP", "tC", "1"]
        elif reference_predictor is "Ad": 
            sub_predictors = ["A", "t", "tA", "1"]
        elif reference_predictor is "Pd": 
            sub_predictors = ["P", "t", "tP", "1"]
        elif reference_predictor is "Cd": 
            sub_predictors = ["C", "t", "tC", "1"]
        elif reference_predictor is "A":
            sub_predictors = ["tA", "1"]
        elif reference_predictor is "P": 
            sub_predictors = ["tP", "1"]
        elif reference_predictor is "C": 
            sub_predictors = ["tC", "1"]
        elif reference_predictor is "t": 
            sub_predictors = ["tA", "tP", "tC", "1"]
        
        def _fill_row(ref_fit, sub_fit):
            family, reference_predictor = ref_fit['family'], ref_fit['predictor']
            ref_deviance, sub_deviance = ref_fit['deviance'], sub_fit['deviance']
            ref_df, sub_df = ref_fit['df_resid'], sub_fit['df_resid']
            n = self.n
            try:
                sub_aic = sub_fit['aic']
            except KeyError:
                pass
            if ref_fit['predictor'] == sub_fit['predictor']:
                LR, df, p_LR = np.nan, np.nan, np.nan
            else:
                LR, df = sub_deviance - ref_deviance, sub_df - ref_df
            
            if family in ('gaussian_rates', 'gaussian_response', 
                          'log_normal_rates', 'log_normal_response',
                          'gen_log_normal_response'):
                F = (np.exp(LR/n) - 1) * ref_df/df
                p_F = stats.f.sf(F, df, ref_df)

                idx = ('-2logL', 'df_resid', 'LR_vs_{}'.format(reference_predictor), 
                       'df_vs_{}'.format(reference_predictor), 
                       'F_vs_{}'.format(reference_predictor), 'P>F', 'aic')
                values = (sub_deviance, sub_df, LR, df, F, p_F, sub_aic)
            elif family in ('poisson_response', 'poisson_dose_response',
                            'binomial_dose_response'):
                p_deviance = 1 - stats.chi2.cdf(sub_deviance, sub_df)
                p_LR = 1 - stats.chi2.cdf(LR, df)
                
                idx = ('deviance', 'df_resid', 'P>chi_sq',
                       'LR_vs_{}'.format(reference_predictor), 
                       'df_vs_{}'.format(reference_predictor), 'P>chi_sq')
                values = (sub_deviance, sub_df, p_deviance, LR, df, p_LR)
            elif family == 'od_poisson_response':
                if ref_fit['predictor'] == sub_fit['predictor']:
                    F, p_F = np.nan, np.nan
                    p_deviance = 1 - stats.chi2.cdf(sub_deviance, sub_df)
                else:
                    F = (LR/df) / (ref_deviance/ref_df)
                    p_F = 1 - stats.f.cdf(F, df, ref_df)
                    p_deviance = np.nan
                idx = ('deviance', 'df_resid', 'P>chi_sq', 
                       'LR_vs_{}'.format(reference_predictor), 
                       'df_vs_{}'.format(reference_predictor), 
                       'F_vs_{}'.format(reference_predictor), 'P>F')
                values = (sub_deviance, sub_df, p_deviance, LR, df, F, p_F)
            
            return pd.Series(values, idx)
        
        ref_fit = self.fit(family, reference_predictor, design_components,
                           attach_to_self=False)
        
        ref_fit_row = _fill_row(ref_fit, ref_fit)
        
        deviance_table = pd.DataFrame(None, index=[reference_predictor] + sub_predictors,
                                      columns=ref_fit_row.index)
        
        deviance_table.loc[reference_predictor, :] = ref_fit_row
        
        for sub_pred in sub_predictors:
            sub_fit = self.fit(family, sub_pred, design_components, attach_to_self=False)
            deviance_table.loc[sub_pred, :] = _fill_row(ref_fit, sub_fit)
        
        if attach_to_self:
            self.deviance_table = deviance_table
        else:
            return deviance_table
    
    def _simplify_range(self, col_range, simplify_ranges):
        """
        Simplifies ranges such as 1955-1959 as indicated by 'simplify_ranges'.
        
        Useful for plotting to obtain more concise axes labels.
        
        Parameters
        ----------
        
        simplify_ranges : {'start', 'mean', 'end'}
        
        """
        try:
            col_from_to = col_range.str.split('-', expand=True).astype(int)
        except AttributeError: #not a column range            
            return col_range
        if simplify_ranges == 'start':
            col_simple = col_from_to.iloc[:,0].astype(int)
        elif simplify_ranges == 'end':
            col_simple = col_from_to.iloc[:,-1].astype(int)
        elif simplify_ranges == 'mean':
            col_simple = col_from_to.mean(axis=1).round().astype(int)
        else:
            raise ValueError("'simplify_range' must be one of 'start', " +
                             "'mean' or 'end'")
        
        return col_simple
    
    def plot_data_sums(self, simplify_ranges='mean', logy=False,
                       figsize=None):
        """
    
        Plot for data sums by age, period and cohort.
        
        Produces plots showing age, period and cohort sums for responses, doses and rates
        (if available).
        
        
        Parameters
        ----------
        
        simplify_ranges : {'start', 'mean', 'end', False}, optional
                          Default is 'mean'. If the time indices are ranges, such as 
                          1955-1959, this determines if and how those should be 
                          transformed. Allows for prettier axis labels.
        
        
        logy : bool, optional
               Specifies whether the y-axis uses a log-scale.  (Default is 'False')
                    
        figsize : float tuple or list, optional
                  Specifies the figure size. If left empty matplotlib determines this
                  internally.
        
        
        Returns
        -------
        
        Matplotlib figure attached to self.plotted_data_sums
        
        Examples
        --------

        >>> import pandas as pd
        >>> data = pd.read_excel('./data/Belgian_lung_cancer.xlsx', 
        ...                      sheet_name = ['response', 'rates'], index_col = 0)
        >>> import apc
        >>> model = apc.Model()
        >>> model.data_from_df(data['response'], rate=data['rates'], 
        ...                    data_format='AP')
        >>> model.plot_data_sums()
        
        """
        try:
            data_vector = self.data_vector
        except AttributeError:
            raise AttributeError("Could not find 'data_vector', run " + 
                                 "Model().data_from_df() first.")
            
        idx_names = data_vector.index.names
        
        if simplify_ranges:
            _simplify_range = self._simplify_range
            data_vector = data_vector.reset_index()
            data_vector[idx_names] = data_vector[idx_names].apply(
                lambda col: _simplify_range(col, simplify_ranges))
            data_vector.set_index(idx_names, inplace=True)
        
        fig, ax = plt.subplots(nrows=data_vector.shape[1], ncols=3, 
                               sharex='col', figsize=figsize)
        
        for i,idx_name in enumerate(idx_names):
            data_sums = data_vector.groupby(idx_name).sum()
            try:
                data_sums.plot(subplots=True, ax= ax[:,i], logy=logy, legend=False)
            except IndexError:
                data_sums.plot(subplots=True, ax= ax[i], logy=logy, legend=False)
        
        for i, col_name in enumerate(data_sums.columns):
            try:
                ax[i,0].set_ylabel(col_name)
            except IndexError:
                ax[0].set_ylabel(col_name)
            
        fig.tight_layout()
        
        self.plotted_data_sums = fig
    
    def _vector_to_array(self, col_vector, space):
        """
        Maps column vector into two-dimensional array in 'space'
        
        
        Parameters
        ----------
        
        For 'space' see documentation of Model().plot_data_heatmaps().
        
        col_vector : pandas.Series
                     Column vector with age-period-cohort index
        
        
        Returns
        -------
        
        pandas.DataFrame with row and column index corresponding to 'space' and values
        corresponding to the values of col_vector.
        
        """
        row_idx, col_idx = space[0], space[1]
        space_dict = {'A': 'Age', 'P': 'Period', 'C': 'Cohort'}
        
        if col_vector.name is None:
            col_vector.name = 'data'
        array = col_vector.reset_index().pivot(index=space_dict[row_idx],
                                               columns=space_dict[col_idx],
                                               values=col_vector.name)
        
        return array
    
    
    def plot_data_heatmaps(self, simplify_ranges='mean', space=None, figsize=None,
                           **kwargs):
        """
        
        Heatmap plot of data.
        
        Produces heatmaps of the data for responses, doses and rates, if applicable. 
        The user can choose what space to plot the heatmaps in, e.g. 'AC' got age-cohort
        space. 
        
        
        Parameters
        ----------
        
        simplify_ranges : {'start', 'mean', 'end', False}, optional
                          Default is 'mean'. If the time indices are ranges, such as 
                          1955-1959, this determines if and how those should be 
                          transformed. Allows for prettier axis labels.
        
        space : {'AC', 'AP', 'PA', 'PC', 'CA', 'CP'}, optional
                Specifies what goes on the axes (A = Age, P = period, C = cohort). 
                By default this is set to 'self.data_format'.
        
        figsize : float tuple or list, optional
                  Specifies the figure size. If left empty matplotlib determines this
                  internally.
        
        **kwargs : any kwargs that seaborn.heatmap can handle, optional
                   The kwargs are fed through to seaborn.heatmap. Note that these are
                   applied to all heatmap plots symmetrically.
        
        
        Returns
        -------
        
        Matplotlib figure attached to self.plotted_data_heatmaps
        
        
        Examples
        --------

        >>> import apc
        >>> model = apc.Model()
        >>> model.data_from_df(**apc.Belgian_lung_cancer())
        >>> model.plot_data_heatmaps()
        >>> model.plotted_data_heatmaps
        
        """    
        
        try:
            data_vector = self.data_vector
        except AttributeError:
            raise AttributeError("Could not find 'data_vector', run " + 
                                 "Model().data_from_df() first.")
        
        
        self.plotted_data_heatmaps = self._plot_heatmaps(data_vector, simplify_ranges,
                                                         space, figsize, **kwargs)

    def _plot_heatmaps(self, data, simplify_ranges='mean', space=None, figsize=None, **kwargs):
        """
        
        Internal function for heatmap plotting.


        Parameters
        ----------
        
        See plot_data_heatmaps for specifics on the inputs.


        Returns
        -------
        
        Matplotlib figure
        

        See also
        --------
        
        Called by Model().plot_data_heatmaps and Model().plot_residuals.

        """
        
        if space is None:
            space = self.data_format
        if space == 'CL':
            space = 'CA'

        idx_names = data.index.names

        if simplify_ranges:
            _simplify_range = self._simplify_range
            data = data.reset_index()
            data[idx_names] = data[idx_names].apply(
                lambda col: _simplify_range(col, simplify_ranges))
            data.set_index(idx_names, inplace=True)

        fig, ax = plt.subplots(nrows=1, ncols=data.shape[1], sharey=True,
                               figsize=figsize)

        for i, col in enumerate(data.columns):
            try:
                active_ax = ax[i]
            except TypeError:
                active_ax = ax
            _vector_to_array = self._vector_to_array
            col_vector = data[col]
            col_array = _vector_to_array(col_vector, space)
            sns.heatmap(ax=active_ax, data=col_array, **kwargs)   
            active_ax.set_title(col_vector.name)
            if i > 0:
                active_ax.set_ylabel('')

        fig.tight_layout()

        return fig

    def plot_data_within(self, n_groups=5, logy=False, aggregate='mean', 
                         figsize=None, simplify_ranges=False):
        """
    
        Plot each timescale within the others, e.g. cohort groups over age.
        
        Produces a total of six plots for each of response, dose, and rate (if applicable).
        These plots are sometimes used to gauge how many of the age, period, cohort factors
        are needed: If lines are parallel when dropping one index the corresponding factor
        may not be needed. In practice these plots should possibly be used with care.
        
        
        Parameters
        ----------
        
        n_groups : int or 'all', optional
                   The number of groups plotted within each time scale, computed either
                   be summing or aggregating existing groups (determined by 'aggregate').
                   The advantage is that the plots become less cluttered if there are
                   fewer groups to show. Default is 5. 
        
        logy : bool, optional
               Specifies whether the y-axis uses a log-scale. Default is 'False'.
        
        aggregate : {'mean', 'sum'}, optional
                    Determines whether aggregation to reduce the number of groups is done
                    by summings or averaging. Default is 'mean'.        
                    
        figsize : float tuple or list, optional
                  Specifies the figure size. If left empty matplotlib determines this
                  internally.
        
        simplify_ranges : {'start', 'mean', 'end', False}, optional
                          Default is 'mean'. If the time indices are ranges, such as 
                          1955-1959, this determines if and how those should be 
                          transformed. Allows for prettier axis labels. Default is 'False'.
        
        
        Notes
        -----
        
        Parts of the description are taken from the R package apc.
        
        
        Returns
        -------
        
        Matplotlib figure(s) attached to self.plotted_data_within. If dose/rate is available
        this is a dictionary with separate figures for response, dose, and rate as values.
        
        
        Examples
        --------
        
        >>> import apc
        >>> model = apc.Model()
        >>> model.data_from_df(**apc.Belgian_lung_cancer())
        >>> model.plot_data_within(figsize=(10,6))
        >>> model.plotted_data_within
                
        """
                
        try:
            data_vector = self.data_vector
        except AttributeError:
            raise AttributeError("Could not find 'data_vector', call " + 
                                 "Model().data_from_df() first.")
        
        self.plotted_data_within = dict()
        
        def _plot_vector_within(self, col_vector, n_groups, logy, 
                                simplify_ranges, aggregate):
            
            fig, ax = plt.subplots(nrows=2, ncols=3, figsize=figsize, 
                                   sharex='col', sharey=True)
            ax[0,0].set_ylabel(col_vector.name)
            ax[1,0].set_ylabel(col_vector.name)
            ax_flat = ax.T.flatten() # makes it easier to iterate
        
            plot_type_dict = {'awc': ('A', 'C'), 
                              'awp': ('A', 'P'),
                              'pwa': ('P', 'A'),
                              'pwc': ('P', 'C'),
                              'cwa': ('C', 'A'),
                              'cwp': ('C', 'P')}
            
            _vector_to_array = self._vector_to_array

            j = 0
            for x, y in plot_type_dict.values():
                array = _vector_to_array(col_vector, space=x+y) 
                idx_name = array.index.name
                
                if simplify_ranges:
                    _simplify_range = self._simplify_range
                    array = array.reset_index()
                    array[idx_name] = _simplify_range(array[idx_name], 
                                                      simplify_ranges)
                    array.set_index(idx_name, inplace=True)

                #adjust for the number of groups
                if n_groups is 'all' or n_groups >= array.shape[1]:
                    group_size = 1
                else:
                    group_size = max(1,round(array.shape[1]/n_groups))
                
                if group_size > 1:
                    array_new = pd.DataFrame(None, index=array.index)
                    for i in range(0, array.shape[1], group_size):
                        idx_in_group = array.columns[i:i+group_size]
                        try:
                            start = idx_in_group[0].split('-')[0]
                            end = idx_in_group[-1].split('-')[1]
                        except:
                            start = str(idx_in_group[0])
                            end = str(idx_in_group[-1])
                        new_idx = '-'.join([start, end])
                        if aggregate == 'mean':
                            array_new[new_idx] = array[idx_in_group].mean(axis=1)
                        elif aggregate == 'sum':
                            array_new[new_idx] = array[idx_in_group].mean(axis=1)
                        else:
                            raise ValueError("'aggregate' must by one of " + 
                                             "'mean' or 'sum'")
                else:
                    array_new = array
                    
                array_new.plot(ax=ax_flat[j], logy=logy)
                y_dict = {'A': 'Age', 'C': 'Cohort', 'P': 'Period'}
                ax_flat[j].legend(title=y_dict[y])
                j += 1
                
            fig.tight_layout() 
            self.plotted_data_within[col_vector.name] = fig
    
        for i,col in enumerate(data_vector.columns):
                _plot_vector_within(self, data_vector[col], n_groups, logy, 
                                    simplify_ranges, aggregate)
        
        if len(self.plotted_data_within) == 1:
            self.plotted_data_within = self.plotted_data_within[data_vector.columns[0]]
    
    def simulate(self, repetitions, fitted_values=None, dose=None, sigma2=None, 
                 poisson_dgp='poisson', od_poisson_dgp='cpg', seed=None, attach_to_self=True):
        """
        
        Simulates data for the fitted model. 
        
        This function simulatess data for the data generating process implied by the model 
        family. Unless otherwise specified, takes the model estimates as true values for the 
        data generating process.
        
        
        Parameters
        ----------
        
        repetitions : int
                      The number of draws.
                      
        fitted_values : pandas.Series, optional
                        For Gaussian and (over-dispersed) Poisson families this
                        corresponds to the mean. For log-normal families this is
                        the median so log(fitted_values) are the Gaussian means. 
                        For binomial, corresponds to the the probabilities. If left
                        unspecified this is set to apc.Model().fitted_values as 
                        returned from apc.Model().fit().
        
        dose : pandas.Series, optional
               Only needed for binomial. Corresponds to the cell-wise number of trials. 
               If left unspecified this is set to apc.data_vector['dose'] as returned
               from apc.Model().data_from_df().
        
        sigma2 : float > 0 (>1 for 'od_poisson_repsonse'), optional
                 For Gaussian and log-normal families this is the variance of the 
                 (log-)Gaussian distribution. For 'od_poisson_response', this is
                 the over-dispersion. If left unspecified this is set to apc.Model().s2
                 as returned from apc.Model().fit() if needed by the data generating 
                 process. Ignored for Poisson and binomial families. 
        
        poisson_dgp : {'poisson', 'multinomial'}, optional
                      Only relevant for family 'poisson_response'. If set to 'poisson', 
                      the data generating process is Poisson. If set to 'multinomial',
                      the data generating process is multinomial with total number
                      of trials equal to the sum of 'fitted_values' and cell-wise
                      probabilities set to 'fitted_values/sum(fitted_values)'. This
                      corresponds to a simulation condational on the total counts
                      and may be of interest for inference in a multinomial sampling
                      scheme; see for example Martínez Miranda et al. (2015). (Default
                      is 'poisson'.)
                      
        od_poisson_dgp : {'cpg', 'neg_binomial'}, optional
                         Determines the data generating process for over-dispersed 
                         Poisson. 'cpg' is compound Poisson gamma and generates 
                         continuous data, 'neg_binomal' is negative binomial and
                         generates discrete data. Compound Poisson is generated as
                         described, for example, by Harnau et al. (2017). (Default is
                         'cpg'.)
                         
        seed : int, optional
               The random seed used to generate the draws.
        
        attach_to_self: bool, optional
                        Default True. If this is True the output is attached to 
                        self.draws. If False the table is returned. (Default is True.)
        
        
        Returns
        -------
        
        pandas.DataFrame of draws either directly or attached to Model().draws (see
        'attach_to_self'). The index of this dataframe corresponds to the index of 
        Model().fitted_values. The draws are in the columns. 
        
        
        Notes
        -----
        
        Generalized log-normal simulations use a log-normal data generating process.
        
        
        Examples
        --------
        
        Log-normal rates
        >>> model = apc.Model()
        >>> model.data_from_df(**apc.Belgian_lung_cancer())
        >>> model.fit('log_normal_rates', 'APC')
        >>> model.simulate(repetitions=5)
        >>> model.draws
        
        Over-dispersed Poisson with compound Poisson Gamma draws
        >>> import apc
        >>> model = apc.Model()
        >>> model.data_from_df(apc.loss_TA())
        >>> model.fit(family='od_poisson_response', predictor='AC')
        >>> model.simulate(repetitions=10)
        >>> model.draws
        
        
        References
        ----------
        
        Harnau, J., & Nielsen, B. (2017). Over-dispersed age-period-cohort models.
        Journal of the American Statistical Association. 
        https://doi.org/10.1080/01621459.2017.1366908
        
        Martínez Miranda, M. D., Nielsen, B., & Nielsen, J. P. (2015). Inference and
        forecasting in the age-period-cohort model with unknown exposure with an 
        application to mesothelioma mortality. Journal of the Royal Statistical 
        Society: Series A (Statistics in Society), 178(1), 29–55. 
        
        """
        
        def _dgp(self, repetitions, fitted_values, dose, sigma2, 
                 poisson_dgp, od_poisson_dgp, seed):
            """Specifies the data generating process"""
            n = self.n
            family = self.family
            if fitted_values is None:
                fitted_values = self.fitted_values
            if (sigma2 is None) and (family not in ('poisson_response',
                                                    'poisson_dose_response',
                                                    'binomial_dose_response')):
                sigma2 = self.s2
            if (dose is None) and (family == 'binomial_dose_response'):
                dose = self.data_vector['dose']
            np.random.seed(seed)
            if family == 'poisson_response':
                if poisson_dgp == 'poisson':
                    means = fitted_values
                    draws = np.random.poisson(means, size=(repetitions, n))
                elif poisson_dgp == 'multinomial':
                    tau = fitted_values.sum()
                    p = fitted_values/tau                    
                    draws = np.random.multinomial(tau,p, size=repetitions)
            elif family == 'poisson_dose_response':
                means = fitted_values
                draws = np.random.poisson(means, size=(repetitions, n))                
            elif family == 'od_poisson_response':
                if od_poisson_dgp == 'cpg':
                    means = fitted_values
                    scale = sigma2 - 1
                    shape = 1/scale
                    draws = np.random.gamma(
                        shape * np.random.poisson(means, size=(repetitions, n)), 
                        scale)
                elif od_poisson_dgp == 'neg_binomial':
                    # Implementation has parameters n for number of successes including
                    # the last success and p for probability of success.
                    # The mean of this is given by (1-p)/p * n.
                    # Variance is (1-p)/p * n / p = mean/p.
                    means = fitted_values
                    p = 1/sigma2
                    successes = p/(1-p) * means
                    draws = np.random.negative_binomial(successes, p, size=(repetitions, n))
                else:
                    raise ValueError("od_poisson must be either 'cpg' or 'nb'")
            elif family in ('gaussian_response', 'gaussian_rates'):
                means = fitted_values
                sigma = np.sqrt(sigma2)
                draws = np.random.normal(means, sigma, size=(repetitions, n))
            elif family in ('log_normal_response', 'log_normal_rates', 'gen_log_normal_response'):
                lin_pred = np.log(fitted_values)
                sigma = np.sqrt(sigma2)
                draws = np.random.lognormal(lin_pred, sigma, size=(repetitions, n))
            elif family == 'binomial_dose_response':
                p = fitted_values
                dose = dose.astype(int)
                draws = np.random.binomial(dose, p, size=(repetitions, n))
            else:
                raise ValueError("Model family can currently not be simulated.")
            return draws
        
        draws = pd.DataFrame(_dgp(self, repetitions, fitted_values, dose, sigma2, 
                                  poisson_dgp, od_poisson_dgp, seed).T,
                            index = self.fitted_values.index)
        if attach_to_self:
            self.draws = draws
        else:
            return draws
    
    def sub_sample(self, age_from_to=(None,None), 
                   per_from_to=(None,None), coh_from_to=(None,None)):
        """
        Generates a sub-sample from Model().data_vector.
        
        
        Parameters
        ----------
        
        age_from_to : tuple, optional
                      The ages to be included in the sub-sample, including start and finish.
                      For example, the tuple (5,10) includes ages from and including 5 to and
                      including 10. Can handle index ranges if they can be sorted meaningfully.
        
        per_from_to : tuple, optional
                      The periods to be included in the sub-sample, including start and finish.
                      For example, the tuple (5,10) includes periods from and including 5 to
                      and including 10. Can handle index ranges if they can be sorted
                      meaningfully.

        coh_from_to : tuple, optional
                      The cohorts to be included in the sub-sample, including start and finish.
                      For example, the tuple (5,10) includes cohorts from and including 5 to
                      and including 10. Can handle index ranges if they can be sorted
                      meaningfully.


        Returns
        -------
        
        pandas.DataFrame corresponding to Model().data_vector except limited to the sub-sample.
        
        
        See Also
        --------
        
        Called by Model().sub_model().
        
        
        Examples
        --------
        
        >>> model = apc.Model()
        >>> model.data_from_df(apc.loss_TA())
        >>> model.sub_sample(per_from_to=(1,5))
        
        >>> model = apc.Model()
        >>> model.data_from_df(**apc.Belgian_lung_cancer())
        >>> model.sub_sample(age_from_to=('30-34', '65-69'))
        
        """
        
        data = self.data_vector
        if data.index.names != ['Age', 'Cohort', 'Period']:
            data = data.reorder_levels(['Age', 'Cohort', 'Period']).sort_index()

        idx = pd.IndexSlice
        return data.loc[idx[age_from_to[0]:age_from_to[1], 
                            coh_from_to[0]:coh_from_to[1], 
                            per_from_to[0]:per_from_to[1]],:]    
    
    def sub_model(self, age_from_to=(None,None), per_from_to=(None,None), 
                  coh_from_to=(None,None), fit=True):
        """
        
        Generate a model from specified sub-sample.
        
        Generates a model with a sub-sample of data attached to it. If not otherwise specified,
        this sub-model is automatically fitted with the same family and predictor as the 
        original model.
        
        
        Parameters
        ----------
        
        age_from_to : tuple, optional
                      The ages to be included in the sub-sample, including start and finish.
                      For example, the tuple (5,10) includes ages from and including 5 to and
                      including 10. Can handle index ranges if they can be sorted meaningfully.
                      (Default is (None, None) thus no sub-sampling.)
        
        per_from_to : tuple, optional
                      The periods to be included in the sub-sample, including start and finish.
                      For example, the tuple (5,10) includes periods from and including 5 to
                      and including 10. Can handle index ranges if they can be sorted
                      meaningfully. (Default is (None, None) thus no sub-sampling.)

        coh_from_to : tuple, optional
                      The cohorts to be included in the sub-sample, including start and finish.
                      For example, the tuple (5,10) includes cohorts from and including 5 to
                      and including 10. Can handle index ranges if they can be sorted
                      meaningfully. (Default is (None, None) thus no sub-sampling.)
        
        fit : bool, optional
              Whether the sub-model should be fit to the data. If True, this is fit with the 
              same family and predictor as the parent model. (Default is True.)
        
        
        Returns
        -------
        
        Model() Class with Model().data_from_df(sub_sample) and, if so specified by 'fit',
        Model().fit() already called.
        
        
        See also
        --------
        
        This function is also useful for the misspecification tests apc.bartlett_test() and 
        apc.f_test().
        
        
        Examples
        --------
        
        >>> model = apc.Model()
        >>> model.data_from_df(apc.loss_TA())
        >>> model.fit('od_poisson_response', 'AC')
        >>> model.sub_model(per_from_to=(1,5))
        
        >>> model = apc.Model()
        >>> model.fit('log_normal_rates', 'APC')
        >>> model.data_from_df(**apc.Belgian_lung_cancer())
        >>> model.sub_model(age_from_to=('30-34', '65-69'))
        
        """
        
        sub_sample = self.sub_sample(age_from_to, per_from_to, coh_from_to)
        if sub_sample.empty:
            raise ValueError('Sub-sample is empty')
        # Reshape into array so we can use the data_from_df functionality.
        space = self.data_format
        if space == 'CL':
            space = 'CA'
        sub_response = self._vector_to_array(sub_sample['response'], space)
        try:
            sub_dose = self._vector_to_array(sub_sample['dose'], space)
        except KeyError:
            sub_dose = None
        sub_model = Model()
        sub_model.data_from_df(sub_response, sub_dose, data_format=self.data_format, 
                               time_adjust=self.time_adjust)
        if fit:
            try:
                sub_model.fit(self.family, self.predictor)
            except AttributeError: # if no model had been fit before.
                pass
        return sub_model

    def identify(self, style='detrend', attach_to_self=True):
        """
        
        Computes ad hoc identified time effects.
        
        Forms ad hoc identified time effects from the canonical parameter. 
        These are used either indirectly by apc.plot.fit or they are 
        computed directly with this command. The ad hoc identifications
        are based on Nielsen (2014, 2015). 
        
        Parameters
        ----------

        style : {'detrend', 'sum_sum'} (optional)
                "detrend" gives double sums that start in zero and end in
                zero. "sum_sum" gives double sums anchored in the middle 
                of the first period diagonal.
        
        attach_to_self: bool (optional)
                        Default True. If this is True the output 
                        is attached to self.parameters_adhoc. If False the
                        table is returned.
                        
                
        Returns
        -------
        
        pandas.DataFrame
        Either attached to self.parameters_adhoc or returned.
        
        The dataframe has the following four columns.
        
        "coef"
            The parameter estimates.
            
        "std err"
            The standard errors for the parameters. If this is "NaN" 
            standard errors are not available for the corresponding 
            estimate.
            
        "t" or "z"
            The t or z statistic. Label depending on the theory for the
            model. "t" if the distribution is t, "z" if the distribution
            is Gaussian.
            
        "P>|t|" or "P>|z|"
            The p-value for the t or z statistic.
            
        
        Examples
        --------
        
        >>> model = apc.Model()
        >>> model.data_from_df(**apc.Belgian_lung_cancer())
        >>> model.fit('gaussian_rates', 'APC')
        >>> model.identify()
        >>> model.parameters_adhoc
        
        
        Notes
        -----
        
        The description is largely taken from the R package apc.
        
        
        References
        ----------
        
        Nielsen, B. (2014) Deviance analysis of age-period-cohort models.
        Nuffield Discussion Paper 2014-W03
        
        Nielsen, B. (2015) apc: An R package for age-period-cohort analysis. 
        R Journal 7, 52-64.
        
        """
        def f(labels, filt): 
            # filter
            return [l for l in labels if filt in l]

        parameters = self.parameters
        estimates = parameters['coef']
        std_err = parameters['std_err']
        index_labels = parameters.index    
        column_labels = parameters.columns
        family = self.family
        predictor = self.predictor

        I, J, K, L = self.I, self.J, self.K, self.L
        U = self._design_components['anchor_index'] # in the papers notation, this is U-1

        age_design = self.design.groupby('Age').first()
        per_design = self.design.groupby('Period').first()
        coh_design = self.design.groupby('Cohort').first()
        
        # sum_sum
        A_design = age_design.loc[:,f(index_labels, 'dd_age')]
        B_design = per_design.loc[:,f(index_labels, 'dd_per')]
        C_design = coh_design.loc[:,f(index_labels, 'dd_coh')]

        A_design.index = 'A_' + A_design.index.astype(str)
        B_design.index = 'B_' + B_design.index.astype(str)
        C_design.index = 'C_' + C_design.index.astype(str)

        age_cov = self.cov_canonical.loc[
            f(index_labels, 'dd_age'), 
            f(index_labels, 'dd_age')]
        per_cov = self.cov_canonical.loc[
            f(index_labels, 'dd_per'), 
            f(index_labels, 'dd_per')]
        coh_cov = self.cov_canonical.loc[
            f(index_labels, 'dd_coh'), 
            f(index_labels, 'dd_coh')]

        A_coef = A_design.dot(estimates[f(index_labels, 'dd_age')])
        B_coef = B_design.dot(estimates[f(index_labels, 'dd_per')])
        C_coef = C_design.dot(estimates[f(index_labels, 'dd_coh')])
        
        if (A_coef == 0).all():
            A_coef = A_coef.replace(0,np.nan)
        if (B_coef == 0).all():
            B_coef = B_coef.replace(0,np.nan)
        if (C_coef == 0).all():
            C_coef = C_coef.replace(0,np.nan)
            
        A_cov = A_design.dot(age_cov).dot(A_design.T)
        B_cov = B_design.dot(per_cov).dot(B_design.T)
        C_cov = C_design.dot(coh_cov).dot(C_design.T)

        A_stderr = pd.Series(
            np.sqrt(np.diag(A_cov)), index=A_coef.index).replace(0, np.nan)
        B_stderr = pd.Series(
            np.sqrt(np.diag(B_cov)), index=B_coef.index).replace(0, np.nan)
        C_stderr = pd.Series(
            np.sqrt(np.diag(C_cov)), index=C_coef.index).replace(0, np.nan)

        A_t_stat = A_coef/A_stderr
        B_t_stat = B_coef/B_stderr
        C_t_stat = C_coef/C_stderr

        if column_labels[-2] is 't':
            get_p_values = lambda t: t if np.isnan(t) else 2 * (
                1 - stats.t.cdf(np.abs(t), self.df_resid))
        else:
            get_p_values = lambda z: z if np.isnan(z) else 2 * (
                1 - stats.norm.cdf(np.abs(z)))

        A_p_values = A_t_stat.apply(get_p_values)
        B_p_values = B_t_stat.apply(get_p_values)
        C_p_values = C_t_stat.apply(get_p_values)  

        A_rows = pd.DataFrame(
            [A_coef, A_stderr, A_t_stat, A_p_values], 
            index=column_labels).T
        B_rows = pd.DataFrame(
            [B_coef, B_stderr, B_t_stat, B_p_values], 
            index=column_labels).T
        C_rows = pd.DataFrame(
            [C_coef, C_stderr, C_t_stat, C_p_values], 
            index=column_labels).T

        A_rows.columns = column_labels
        B_rows.columns = column_labels
        C_rows.columns = column_labels

        if style == 'sum_sum':
            parameters_adhoc = pd.concat(
                [parameters, A_rows, B_rows, C_rows], axis=0)
        elif style == 'detrend':    
            A_d_design = np.identity(I)
            A_d_design[:,0] += -1 + (np.arange(1,I+1) - 1)/(I-1)
            A_d_design[:,-1] -= (np.arange(1,I+1)-1)/(I-1)
            A_d_design = pd.DataFrame(
                A_d_design, index=A_coef.index, columns=A_coef.index)

            B_d_design = np.identity(J)
            B_d_design[:,0] += -1 + (np.arange(1,J+1) -  1)/(J-1)
            B_d_design[:,-1] -= (np.arange(1,J+1) - 1)/(J-1)
            B_d_design = pd.DataFrame(
                B_d_design, index=B_coef.index, columns=B_coef.index)

            C_d_design = np.identity(K)
            C_d_design[:,0] += -1 + (np.arange(1,K+1) - 1)/(K-1)
            C_d_design[:,-1] -= (np.arange(1,K+1)-1)/(K-1)
            C_d_design = pd.DataFrame(
                C_d_design, index=C_coef.index, columns=C_coef.index)    

            A_d_coef = A_d_design.dot(A_coef)
            B_d_coef = B_d_design.dot(B_coef)
            C_d_coef = C_d_design.dot(C_coef)
            
            if (A_d_coef == 0).all():
                A_d_coef = A_d_coef.replace(0,np.nan)
            if (B_d_coef == 0).all():
                B_d_coef = B_d_coef.replace(0,np.nan)
            if (C_d_coef == 0).all():
                C_d_coef = C_d_coef.replace(0,np.nan)
            
            A_d_cov = A_d_design.dot(A_cov).dot(A_d_design.T)
            B_d_cov = B_d_design.dot(B_cov).dot(B_d_design.T)
            C_d_cov = C_d_design.dot(C_cov).dot(C_d_design.T)

            A_d_stderr = pd.Series(
                np.sqrt(np.diag(A_d_cov)), index=A_coef.index).replace(0, np.nan)
            B_d_stderr = pd.Series(
                np.sqrt(np.diag(B_d_cov)), index=B_coef.index).replace(0, np.nan)
            C_d_stderr = pd.Series(
                np.sqrt(np.diag(C_d_cov)), index=C_coef.index).replace(0, np.nan)

            level_d_design = pd.Series(0, index=index_labels)
            level_d_design.loc['level'] = 1
            level_d_design.loc[f(index_labels,'slope_age')] = -U
            level_d_design.loc[f(index_labels,'slope_coh')] = -U
            level_d_design.loc[f(index_labels,'slope_per')] = -U
            level_d_design.loc[
                f(index_labels, 'dd_age')] = A_design.iloc[0,:]
            level_d_design.loc[
                f(index_labels, 'dd_coh')] = C_design.iloc[0,:]
            level_d_design.loc[
                f(index_labels, 'dd_per')
            ] = B_design.iloc[0,:] - L * (B_design.iloc[-1,:] - B_design.iloc[0,:])/(J-1)
            
            
            if predictor in ('APC', 'AP', 'AC', 'PC', 'Ad', 'Pd', 'Cd', 'A', 't', 'tA'):
                slope_age_d_design = pd.Series(0, index=index_labels)
                slope_age_d_design.loc[f(index_labels,'slope_age')] = 1
                slope_age_d_design.loc[f(index_labels, 'dd_age')
                ] = (A_design.iloc[-1,:] - A_design.iloc[0,:])/(I-1)
                slope_age_d_design.loc[f(index_labels, 'dd_per')
                ] = (B_design.iloc[-1,:] - B_design.iloc[0,:])/(J-1)
                slope_age_d_coef = slope_age_d_design.dot(estimates)
                slope_age_d_stderr = np.sqrt(slope_age_d_design.dot(
                    self.cov_canonical).dot(slope_age_d_design))
                if slope_age_d_coef == 0:
                    slope_age_d_coef = np.nan
                if slope_age_d_stderr == 0:
                    slope_age_d_stderr = np.nan
                slope_age_d_t_stat = slope_age_d_coef/slope_age_d_stderr
                slope_age_d_p_values = get_p_values(slope_age_d_t_stat)
                slope_age_d_row = pd.DataFrame(
                    [slope_age_d_coef, slope_age_d_stderr, 
                     slope_age_d_t_stat, slope_age_d_p_values], 
                    index=column_labels, columns=['slope_age_detrend']).T   
            else:
                slope_age_d_row = None
                
            if predictor in ('APC', 'AP', 'AC', 'PC', 'Ad', 'Pd', 'Cd', 'C', 't', 'tC'):
                slope_coh_d_design = pd.Series(0, index=index_labels)
                slope_coh_d_design.loc[f(index_labels,'slope_coh')] = 1
                slope_coh_d_design.loc[f(index_labels, 'dd_coh')
                ] = (C_design.iloc[-1,:] - C_design.iloc[0,:])/(K-1)
                slope_coh_d_design.loc[f(index_labels, 'dd_per')
                ] = (B_design.iloc[-1,:] - B_design.iloc[0,:])/(J-1)
                slope_coh_d_coef = slope_coh_d_design.dot(estimates)
                slope_coh_d_stderr = np.sqrt(slope_coh_d_design.dot(
                    self.cov_canonical).dot(slope_coh_d_design))          
                if slope_coh_d_coef == 0:
                    slope_coh_d_coef = np.nan
                if slope_coh_d_stderr == 0:
                    slope_coh_d_stderr = np.nan
                slope_coh_d_t_stat = slope_coh_d_coef/slope_coh_d_stderr
                slope_coh_d_p_values = get_p_values(slope_coh_d_t_stat)                
                slope_coh_d_row = pd.DataFrame(
                    [slope_coh_d_coef, slope_coh_d_stderr, 
                     slope_coh_d_t_stat, slope_coh_d_p_values], 
                    index=column_labels, columns=['slope_coh_detrend']).T
            else:
                slope_coh_d_row = None
                
            if predictor in ('P', 'tP'):
                slope_per_d_design = pd.Series(0, index=index_labels)
                slope_per_d_design.loc[f(index_labels,'slope_per')] = 1
                slope_per_d_design.loc[f(index_labels, 'dd_per')
                ] = (B_design.iloc[-1,:] - B_design.iloc[0,:])/(J-1)
                slope_per_d_coef = slope_per_d_design.dot(estimates)
                slope_per_d_stderr = np.sqrt(slope_per_d_design.dot(
                    self.cov_canonical).dot(slope_per_d_design))
                if slope_per_d_coef == 0:
                    slope_per_d_coef = np.nan
                if slope_per_d_stderr == 0:
                    slope_per_d_stderr = np.nan
                slope_per_d_t_stat = slope_per_d_coef/slope_per_d_stderr
                slope_per_d_p_values = get_p_values(slope_per_d_t_stat)                
                slope_per_d_row = pd.DataFrame(
                    [slope_per_d_coef, slope_per_d_stderr, 
                     slope_per_d_t_stat, slope_per_d_p_values], 
                    index=column_labels, columns=['slope_per_detrend']).T
            else:
                slope_per_d_row = None
                
            level_d_coef = level_d_design.dot(estimates)
            level_d_stderr = np.sqrt(level_d_design.dot(
                self.cov_canonical).dot(level_d_design))
            if self.cov_canonical.loc['level', 'level'] == 0:
                level_d_stderr = np.nan
            level_d_t_stat = level_d_coef/level_d_stderr
            level_d_p_values = get_p_values(level_d_t_stat)            
            level_d_row = pd.DataFrame(
                [level_d_coef, level_d_stderr, 
                 level_d_t_stat, level_d_p_values
                ], 
                index=column_labels, columns=['level_detrend']).T
            
            A_d_t_stat = A_d_coef/A_d_stderr
            B_d_t_stat = B_d_coef/B_d_stderr
            C_d_t_stat = C_d_coef/C_d_stderr
            
            A_d_p_values = A_d_t_stat.apply(get_p_values)
            B_d_p_values = B_d_t_stat.apply(get_p_values)
            C_d_p_values = C_d_t_stat.apply(get_p_values)        

            A_d_rows = pd.DataFrame(
                [A_d_coef, A_d_stderr, A_d_t_stat, A_d_p_values],
                index=column_labels).T
            B_d_rows = pd.DataFrame(
                [B_d_coef, B_d_stderr, B_d_t_stat, B_d_p_values],
                index=column_labels).T
            C_d_rows = pd.DataFrame(
                [C_d_coef, C_d_stderr, C_d_t_stat, C_d_p_values],
                index=column_labels).T

            A_d_rows.index = A_d_rows.reset_index()['index'].apply(
                lambda x: 'A_detrend_' + x[2:])
            B_d_rows.index = B_d_rows.reset_index()['index'].apply(
                lambda x: 'B_detrend_' + x[2:])
            C_d_rows.index = C_d_rows.reset_index()['index'].apply(
                lambda x: 'C_detrend_' + x[2:])
            A_d_rows.columns = column_labels
            B_d_rows.columns = column_labels
            C_d_rows.columns = column_labels
            
            parameters_adhoc = pd.concat(
                [level_d_row, slope_age_d_row, slope_coh_d_row, 
                 slope_per_d_row, parameters.loc[f(index_labels, 'dd_'), :],
                 A_d_rows, B_d_rows, C_d_rows], axis=0)
        else:
            raise ValueError('style must be "sum_sum" or "detrend".')
        parameters_adhoc.dropna(how='all', inplace=True)
        
        if attach_to_self:
            self.parameters_adhoc = parameters_adhoc
            import warnings
            warnings.warn('From the next release, attribute "para_table_adhoc" is going to be ' +
                          'available only as "parameters_adhoc"', FutureWarning)
            # removed in next version
            self.para_table_adhoc = parameters_adhoc
        else:
            return parameters_adhoc
        
    def plot_parameters(self, plot_style='detrend', around_coef=True, 
                        simplify_ranges='start', figsize=(10,8)):
        """
    
        Plots for estimates from apc.Model.fit() with standard errors.
        
        Produces up to nine subplots, depending on the predictor. Plotted are the 
        double differences (first row), linear trends and level (second row),
        and adhoc identified transformations of the parameters (see help for 
        apc.Model().identify). One and two standard errors are also plotted, except
        for families '.poisson.response' and 'od.poisson.response' which use a multinomial
        sampling scheme (or akin to that 'od.poisson.response').
        
        
        Parameters
        ----------
        
        plot_style : {'detrend', 'sum_sum'} (optional)
                     "detrend" gives double sums that start in zero and end in
                     zero. "sum_sum" gives double sums anchored in the middle 
                     of the first period diagonal. (Default is 'detrend')
        
        around_coef : bool (optional)
                      Determines whetherr standard errors are plotted arounf the estimates
                      (True) or around zero (False). (Default is False)
        
        simplify_ranges : {'start', 'mean', 'end', False} (optional)
                          If the time indices are ranges, such as 1955-1959, this 
                          determines if and how those should be transformed. Allows for 
                          prettier axis labels. (Default is 'start'.)
        
        figsize : float tuple or list, optional
                  Specifies the figure size. (Default is (10,8).)


        Returns
        -------
        
        Matplotlib figure attached to self.plotted_fit. 


        Notes
        -----
        
        Parts of the description are taken from the R package apc.


        Examples
        --------
        
        >>> model = apc.Model()
        >>> model.data_from_df(**apc.Belgian_lung_cancer())
        >>> model.fit('gaussian_rates', 'APC')
        >>> model.plot_parameters()
        >>> model.plotted_parameters
                
        """
        fig, ax = plt.subplots(ncols=3, nrows=3, figsize=figsize)
        # first column is age, second period, third cohort. 
        # exceptions are P and tP models for which the period trend is
        # in the first column.
        # first row is double diffs, second trend and level, third sum.sum

        def f(labels, filt): # filter
            return [l for l in labels if filt in l]

        def err_plot(coef, stderr, ax, xticklabel, xlabel, 
                     title=None, around_coef=True):
            try:
                coef.index = xticklabel
                stderr.index = xticklabel
                coef.plot(ax=ax, color='black')
                ci1 = (coef*around_coef - stderr, 
                       coef*around_coef + stderr)
                ci2 = (coef*around_coef - 2*stderr, 
                       coef*around_coef + 2*stderr)
                ax.fill_between(
                    xticklabel, ci1[0],ci1[1],alpha=0.2, color='blue')
                ax.fill_between(
                    xticklabel, ci1[0], ci2[0], alpha=0.1, color='green')
                ax.fill_between(
                    xticklabel, ci1[1], ci2[1],alpha=0.1, color='green')
                ax.axhline(0, color='black', linewidth=1)
                ax.set_title(title)
                ax.set_xlabel(xlabel)
            except TypeError: 
                #occurs if inputs is an empty series. 
                # Then turn off all labels
                ax.axis('off')

        I, J, K, L = self.I, self.J, self.K, self.L
        predictor = self.predictor

        parameters_adhoc = self.identify(plot_style, attach_to_self=False)

        get_coefs = lambda x: parameters_adhoc.loc[
            f(parameters_adhoc.index, x), 'coef'] 
        get_stderr = lambda x: parameters_adhoc.loc[
            f(parameters_adhoc.index, x), 'std_err'] 

        def get_xticklabels(series, simplify_to):
            try:
                col_range = series.reset_index()['index'].str.split(
                    '_', expand=True).iloc[:,-1]
                if simplify_to == False:
                    label = col_range.values
                else:
                    label = self._simplify_range(
                        col_range, simplify_to).values
                return label
            except:
                pass
        # double differences


        err_plot(get_coefs('dd_age'), get_stderr('dd_age'), ax[0,0],
                 get_xticklabels(get_coefs('dd_age'), simplify_ranges), 
                 'age', r'$\Delta^2 \alpha$', around_coef)
        err_plot(get_coefs('dd_per'), get_stderr('dd_per'), ax[0,1],
                 get_xticklabels(get_coefs('dd_per'), simplify_ranges),
                 'period', r'$\Delta^2 \beta$', around_coef)
        err_plot(get_coefs('dd_coh'), get_stderr('dd_coh'), ax[0,2], 
                 get_xticklabels(get_coefs('dd_coh'), simplify_ranges),
                 'cohort', r'$\Delta^2 \gamma$', around_coef)

        # level and slopes
        def get_trend(time_scale, coef_or_stderr):
            if coef_or_stderr == 'coef':
                h = get_coefs
            elif coef_or_stderr == 'stderr':
                h = get_stderr
            if time_scale == 'age':
                trend = pd.Series(np.arange(I)).apply(
                    lambda i: i*h('slope_age'))
            elif time_scale == 'per':
                trend = pd.Series(np.arange(J)).apply(
                    lambda j: j*h('slope_per'))
            else:
                trend = pd.Series(np.arange(K)).apply(
                    lambda k: k*h('slope_coh'))
            try:
                trend = trend.iloc[:,0]
            except IndexError:
                pass
            return trend  
        # the labels for the trends are more complicated since we
        # construct those from the slopes to be begin with.
        def get_trend_xticklabel(trend, simplify_ranges):
            if trend == 'age':
                raw_label = self.design.groupby('Age').first().index
            elif trend == 'per':
                raw_label = self.design.groupby('Period').first().index
            elif trend == 'coh':
                raw_label = self.design.groupby('Cohort').first().index
            raw_label = pd.Series(raw_label)
            if simplify_ranges is False:
                labels = raw_label.values
            else:
                labels = self._simplify_range(raw_label, simplify_ranges)
            return labels

        if predictor in ('P', 'tP'):
            err_plot(get_trend('per', 'coef'), get_trend('per', 'stderr'),
                     ax[1,0], get_trend_xticklabel('per', simplify_ranges),
                     'period', 'period linear trend', around_coef)
        else:
            err_plot(get_trend('age', 'coef'), get_trend('age', 'stderr'),
                     ax[1,0], get_trend_xticklabel('age', simplify_ranges),
                     'age', 'first linear trend', around_coef)
        err_plot(get_trend('coh', 'coef'), get_trend('coh', 'stderr'),
                 ax[1,2], get_trend_xticklabel('coh', simplify_ranges),
                 'cohort', 'second linear trend', around_coef)

        # In the following cases the trends in the age and cohort direction
        # can be attributed to age or cohort.
        if predictor in ('A', 'tA'):
            ax[1,0].set_title('age linear trend')
        if predictor in ('C', 'tC'):
            ax[1,2].set_title('cohort linear trend')

        err_plot(pd.Series([get_coefs('level')[0]] * 2),
                 pd.Series([get_stderr('level')[0]] * 2),             
                 ax[1,1], range(2), 
                 'age, period,cohort', 'level', around_coef)
        ax[1,1].set_xticks([])

        A_title = r'$\left.\sum\right.^2 \Delta^2 \alpha$'
        B_title = r'$\left.\sum\right.^2 \Delta^2 \beta$'
        C_title = r'$\left.\sum\right.^2 \Delta^2 \gamma$'
        if plot_style == 'detrend':
            A_title = 'detrended ' + A_title
            B_title = 'detrended ' + B_title
            C_title = 'detrended ' + C_title

        err_plot(get_coefs('A_'), get_stderr('A_'), ax[2,0],
                 get_xticklabels(get_coefs('A_'), simplify_ranges),
                 'age', A_title, around_coef)
        err_plot(get_coefs('B_'), get_stderr('B_'), ax[2,1],
                 get_xticklabels(get_coefs('B_'), simplify_ranges),
                 'period', B_title, around_coef)
        err_plot(get_coefs('C_'), get_stderr('C_'), ax[2,2],
                 get_xticklabels(get_coefs('C_'), simplify_ranges),
                 'cohort', C_title, around_coef)

        fig.tight_layout()

        self.plotted_parameters = fig
        
        import warnings
        warnings.warn('From the next release, "plot_fit" and "plotted_fit" are going to be ' +
                      'available exclusively as "plot_parameters" and "plotted_fit", respectively.',
                      FutureWarning)
        # removed in next version
        self.plotted_fit = fig      
        
    def _get_fc_design(self, predictor):
        """
        Generates design for forecasting array.
        """
        ac_array = self._vector_to_array(
            self.data_vector['response'], space='AC'
        )
        
        # need to fill future period with dummy data
        # need to take care not to mess with past periods - changes anchor index and design
        I = self.I
        K = self.K
        if I <= K:
            for i in range(I):
                ac_array.iloc[i, K-i:] = ac_array.iloc[i, K-i:].fillna(np.inf)
        else:
            for k in range(K):
                ac_array.iloc[I-k:, k] = ac_array.iloc[I-k:, k].fillna(np.inf)

        # generate auxiliary model to get design
        tmp_model = Model()
        tmp_model.data_from_df(ac_array, data_format='AC')
        full_design = tmp_model._get_design(predictor)

        # get last in-sample period
        d = dict(zip(full_design.index.names, range(3)))
        per_labels = full_design.index.levels[d['Period']]
        J = self.J
        max_insmpl_per_label = per_labels[J]

        fc_design = full_design.loc[
            pd.IndexSlice[:,:,max_insmpl_per_label:],:
        ]
        return fc_design

    def _get_point_fc(self, attach_to_self=True):
        """
        Generate point forecasts.
        """
        fc_design = self._get_fc_design(self.predictor)
        fc_linpred = fc_design.dot(self.parameters['coef']).rename(
            'linear_predictor_forecast')

        if self.family in ('poisson_response', 'od_poisson_response'):
            fc_point = np.exp(fc_linpred).rename('point_forecast')
        elif self.family in ('log_normal_response', 'gen_log_normal_response'):
            fc_point = np.exp(fc_linpred + self.s2/2).rename('point_forecast')
        elif self.family in ('gaussian_response'):
            fc_point = fc_linpred.rename('point_forecast')
        else:
            raise ValueError('Currently supports only "poisson_response", ' +
                             '"log_normal_response", "gen_log_normal_response"' +
                             ' "od_poisson_response" and "gaussian_response"')

        if attach_to_self:
            self._fc_design = fc_design
            self._fc_linpred = fc_linpred
            self._fc_point = fc_point
        else:
            return fc_point

    def forecast(self, quantiles=[0.75, 0.9, 0.95, 0.99], method=None, attach_to_self=True):
        """
        Generate forecasts.
        
        Generates point and closed form distribution forecasts by cell, age, period and cohort
        as well as for the total forecast array. Currently supports forecasting for future 
        periods in response only models without parameter extrapolation.
                
        Parameters
        ----------
        
        quantiles : iterable of floats in (0, 1), optional
                    The quantiles for which the distribution forecast should be computed. 
                    (Default is [0.75, 0.9, 0.95, 0.99].)
        
        method : {'n_gauss', 'n_poisson', 't_odp', 't_gln'}, optional
                 Determines the forecasting method. 'n_gauss' uses standard Gaussian theory 
                 and is appropriate for 'gaussian_response'. 'n_poisson' is appropriate for 
                 'poisson_response' models and uses the theory in Martinez Miranda (2015).
                 't_odp' is appropriate for 'od_poisson_response' and uses the theory from
                 Harnau and Nielsen (2017). 't_gln' is appropriate for 'log_normal_response' 
                 and 'gen_log_normal_response' and uses the theory from Kuang and Nielsen 
                 (2018). (Default uses best choice for the model.)
        
        attach_to_self : bool, optional
                         If this is True the results are attached to self.forecasts. If False
                         the results are returned. (Default is True.)
        
        
        Returns
        -------
        
        dictionary of pandas.DataFrame's with keys 'Cell', 'Age', 'Period', 'Cohort', 'Total'
        and 'method'. Attached to self.forecasts if 'attach_to_self' is True and returned 
        otherwise. The DataFrames contain point forecasts, standard errors broken down into 
        process and estimation error, and quantiles of the forecast distribution.
        
        
        See also
        --------
        
        Vignettes in apc/vignettes/vignette_mesothelioma.ipynb and
        apc/vignettes/vignette_over_dispersed_apc.ipynb.
        
        
        Raises
        ------
        
        ValueError: matrices are not aligned 
            Raised if forecast would require parameter extrapolation.
        
        
        References
        ----------
                
        Harnau, J. and Nielsen, B. (2017) Asymptotic theory for over-dispersed 
        age-period-cohort and extended chain ladder models. To appear in Journal
        of the American Statistical Association.
        
        Kuang, D., & Nielsen, B. (2018). Generalized Log-Normal Chain-Ladder.
        ArXiv E-Prints, 1806.05939. Available from http://arxiv.org/abs/1806.05939
        
        Martinez Miranda, M.D., Nielsen, B. and Nielsen, J.P. (2015) Inference 
        and forecasting in the age-period-cohort model with unknown exposure 
        with an application to mesothelioma mortality. Journal of the Royal 
        Statistical Society A 178, 29-55.     
        
        Examples
        --------
        
        >>> model = apc.Model()
        >>> model.data_from_df(apc.loss_TA(), data_format='CL')
        >>> model.fit('od_poisson_response', 'AC')
        >>> model.forecast()
        
        >>> model = apc.Model()
        >>> model.data_from_df(apc.asbestos(), data_format='PA')
        >>> model.fit('poisson_response', 'AC')
        >>> model.forecast()
        
        """
        if method is None:
            family = self.family
            if family == 'gaussian_response':
                method = 'n_gauss'
            elif family == 'poisson_response':
                method = 'n_poisson'
            elif family == 'od_poisson_response':
                method = 't_odp'
            elif family in ('log_normal_response', 'gen_log_normal_response'):
                method = 't_gln'
            else:
                raise ValueError('Forecasting not supported for this family.')
        
        self._get_point_fc()
        
        def _agg(df, lvl):
            if lvl == 'Cell':
                return df
            elif lvl in ('Age', 'Period', 'Cohort'):
                return df.sum(level=lvl).sort_index()
            else:
                try:
                    return pd.DataFrame(df.sum(), columns=['Total']).T
                except ValueError: # scalar case
                    return pd.Series(df.sum(), index=['Total'], name=df.name)

        def _get_process_error(method, lvl):
            fc_point = self._fc_point
            fc_linpred = self._fc_linpred
            if method == 'n_gauss':
                pe_sq = pd.Series(self.s2, fc_point.index).rename('se_process')
            elif method == 'n_poisson':
                pe_sq = _agg(fc_point, lvl).rename('se_process')
            elif method == 't_odp':
                pe_sq = _agg(fc_point, lvl).rename('se_process') * self.s2
            elif method == 't_gln':
                pe_sq = _agg(np.exp(fc_linpred)**2, lvl).rename('se_process') * self.s2
            return np.sqrt(pe_sq)

        def _get_estimation_error(method, lvl):
            cov = self.cov_canonical
            if method == 'n_gauss':
                fc_X_A = _agg(self._fc_design, lvl)
                se_sq = pd.Series(
                    np.diag(fc_X_A.dot(cov).dot(fc_X_A.T)), 
                    fc_X_A.index, name='se_estimation_xi')
            elif method in ('n_poisson', 't_odp'):
                # estimation error for xi2
                X2, fc_X2 = self.design.iloc[:,1:], self._fc_design.iloc[:,1:]
                tau = self.fitted_values.sum()
                i_xi2_inv = cov.iloc[1:,1:] * tau
                pi = self.fitted_values/tau
                fc_H = fc_X2 - pi.dot(X2)
                fc_point = self._fc_point
                fc_pi = fc_point/tau
                fc_pi_H = (fc_pi * fc_H.T).T
                fc_pi_H_A = _agg(fc_pi_H, lvl)
                s_A_2 = np.einsum('ip,ip->i', fc_pi_H_A.dot(i_xi2_inv*tau), fc_pi_H_A)
                se_sq = pd.Series(s_A_2, fc_pi_H_A.index, name='se_estimation_xi2')
            elif method == 't_gln':
                fc_linpred = self._fc_linpred
                fc_X = self._fc_design
                fctr_A = _agg((np.exp(fc_linpred) * fc_X.T).T, lvl)
                se_sq = pd.Series(
                    np.diag(fctr_A.dot(cov).dot(fctr_A.T)), 
                    fctr_A.index, name='se_estimation_xi')
            if method == 't_odp':
                pi_A = _agg(fc_pi, lvl)
                se_sq = pd.concat([
                    se_sq, 
                    pd.Series(pi_A**2 * tau * self.s2, 
                              pi_A.index, name='se_estimation_tau')
                ], axis=1)
            return np.sqrt(se_sq)

        def _get_total_error(se_process, se_estimation):
            st_sq = pd.concat([
                se_process**2, se_estimation**2
            ], axis=1).sum(axis=1).rename('se_total')
            return np.sqrt(st_sq)

        def _get_quantiles(qs, method, lvl, se_total):
            if qs is None:
                qs = []
            if method in ('n_gauss', 't_odp', 't_gln'):
                cvs = stats.t.ppf(qs, self.df_resid)
            elif method == 'n_poisson':
                cvs = stats.norm.ppf(qs)
            fc_point_A = _agg(self._fc_point, lvl)
            return pd.DataFrame(
                (fc_point_A.values + np.outer(se_total, cvs).T).T, 
                fc_point_A.index, ['q_' + str(q) for q in np.asarray(qs)])

        def _get_fc_table(qs, method, lvl):
            point_fc = _agg(self._fc_point, lvl)
            se_proc = _get_process_error(method, lvl)
            se_est = _get_estimation_error(method, lvl)
            se_total = _get_total_error(se_proc, se_est)
            quants = _get_quantiles(qs, method, lvl, se_total)
            table = pd.concat([
                point_fc, se_total, se_proc, se_est, quants
            ], axis=1)
            return table
        
        fc_results = {}
        for lvl in ('Cell', 'Age', 'Period', 'Cohort', 'Total'):
            fc_results[lvl] = _get_fc_table(quantiles, method, lvl)
        fc_results['method'] = method
        
        if attach_to_self:
            self.forecasts = fc_results
        else:
            return fc_results
    
    def clone(self):
        """
        Clone model with attached data but without fitting.
        
        Clones the model with attached data so for example other model families can be fitted
        without overwriting the previous results. Spares the hassle of going through the
        data attachment process again.
        
        
        Returns
        -------
        
        Model Class after Model().data_from_df() was called.
        
        Examples
        --------
        
        >>> model = apc.Model()
        >>> model.data_from_df(**apc.Belgian_lung_cancer())
        >>> cloned_model = model.clone()
        
        """
        clone = Model()
        for attribute in ('data_format', 'data_vector', 
                          'I', 'J', 'K', 'L', 'n', 'time_adjust'):
            setattr(clone, attribute, getattr(self, attribute))

        return clone
    
    def plot_residuals(self, kind=None, transform=None, simplify_ranges='mean',
                       space=None, figsize=None, **kwargs):
        """
        
        Heatmap plot of residuals.
        
        Produces heatmaps of the residuals. The user can choose what space to plot
        the heatmaps in, e.g. 'AC' got age-cohort space.
        
        Parameters
        ----------
        
        simplify_ranges : {'start', 'mean', 'end', False}, optional
                          Default is 'mean'. If the time indices are ranges, such as
                          1955-1959, this determines if and how those should be
                          transformed. Allows for prettier axis labels.
        
        kind : {'anscombe', 'deviance', 'pearson', 'response'}, optional
               Determines what residuals are plotted. (Default is 'deviance'.)
        
        transform : function, optional
                    A transformation to be applied to the residuals.
        
        space : {'AC', 'AP', 'PA', 'PC', 'CA', 'CP'}, optional
                Specifies what goes on the axes (A = Age, P = period, C = cohort).
                By default this is set to 'self.data_format'.
        
        figsize : float tuple or list, optional
                  Specifies the figure size. If left empty matplotlib determines this
                  internally.
        
        **kwargs : any kwargs that seaborn.heatmap can handle, optional
                   The kwargs are fed through to seaborn.heatmap. Note that these are
                   applied to all heatmap plots symmetrically.
                   
        Returns
        -------
        
        Matplotlib figure attached to self.plotted_residuals
        
        Examples
        --------
        
        >>> import apc
        >>> model = apc.Model()
        >>> model.data_from_df(apc.loss_TA(), data_format='CL')
        >>> model.fit('od_poisson_response', 'AC')
        >>> model.plot_residuals(kind='pearson')
                
        """
        
        if kind is None:
            kind = 'deviance'
        
        try:
            residuals = self.residuals[kind]
        except AttributeError:
            raise AttributeError('Could not find "residuals", fit a model first.')
        
        if transform is not None:
            residuals = transform(residuals)
        
        self.plotted_residuals = self._plot_heatmaps(pd.DataFrame(residuals), simplify_ranges,
                                                     space, figsize, **kwargs)
    
    def plot_forecast(self, by='Period', ic=False, from_to=(None, None),
                      aggregate=False, figsize=None):
        """
        Plot forecast over a specified time dimension.

        Generates a plot of response and point forecasts with one and two standard
        error bands over a specified time dimension. Allows for intercept correction.

        Parameters
        ----------

        by : {'Age', 'Period', 'Cohort'}, optional
             Level to aggregate by. (Default is 'Period'.)

        ic : bool, optional
             Whether intercept correction should be applied. If True, multiply point
             forecasts by the ratio of the last realization to the last fitted value.
             (Default is False.)

        from_to : Tuple, optional
                  Specifies the plotted range. (Default is (None, None), plotting everything)

        aggregate : bool, optional
                    Whether response and point forecast should be aggregated. Mostly relevant
                    if 'by' is not equal to 'Period'. (Default is False.)

        Returns
        -------

        Matplotlib figure attached to self.plotted_forecast.

        Examples
        --------

        Plot forecast of mesothelioma mortality by period with intercept correction.

        >>> import apc
        >>> model = apc.Model()
        >>> model.data_from_df(apc.asbestos())
        >>> model.fit('poisson_response', 'AC')
        >>> model.forecast()
        >>> model.plot_forecast(ic=True)

        Plot forecast of reserve by cohort / accident year, aggregated.

        >>> import apc
        >>> model = apc.Model()
        >>> model.data_from_df(apc.loss_TA())
        >>> model.fit('od_poisson_response', 'AC')
        >>> model.forecast()
        >>> model.plot_forecast(by='Cohort', aggregate=True)

        """

        flt = pd.IndexSlice[from_to[0]:from_to[1]]

        response = self.data_vector['response'].sum(level=by).loc[flt]
        fitted = self.fitted_values.sum(level=by).rename('fitted').loc[flt]
        point_fc = self.forecasts[by]['point_forecast'].copy().loc[flt]
        if ic:
            ic_factor = response.sort_index().iloc[-1]/fitted.sort_index().iloc[-1]
            point_fc *= ic_factor
        if aggregate:
            try:
                point_fc += response[point_fc.index]
            except KeyError:
                pass
        se_total = self.forecasts[by]['se_total'].loc[flt]
        ci1 = (point_fc - se_total, point_fc + se_total)
        ci2 = (point_fc - 2*se_total, point_fc + 2*se_total)

        fig, ax = plt.subplots(figsize=figsize)
        response.plot(ax=ax, style='o')
        fitted.plot(ax=ax, style='k-')
        point_fc.plot(ax=ax, style='k--')

        ax.fill_between(ci1[0].index, ci1[0], ci1[1],alpha=0.2, color='blue',
                        label='1 std error')
        ax.fill_between(ci1[0].index, ci1[0], ci2[0], alpha=0.1, color='green',
                        label='2 std error')
        ax.fill_between(ci1[0].index, ci1[1], ci2[1],alpha=0.1, color='green')

        title = 'Forecasts by {}'.format(by)
        if ic:
            title += ', Intercept Corrected'
        if aggregate:
            title += ', Aggregated'

        ax.set_title(title)
        plt.legend()
        fig.tight_layout()

        self.plotted_forecast = fig