import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
import collections
import matplotlib.pyplot as plt
import seaborn as sns

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
        ...                      sheetname = ['response', 'rates'], index_col = 0)
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
            dd_age.loc[slope_age == slope_age[0] + i, i:anchor_index] = (
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
        
    def fit(self, family, predictor, design_components=None, attach_to_self=True):
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
    
        family : {"binomial_dose_response", 
                  "poisson_response", "od_poisson_response", 
                  "gaussian_rates", "gaussian_response", 
                  "log_normal_rates", "log_normal_response"}
                  
                  "poisson_response"
                      Poisson family with log link. Only responses are 
                      used. Inference is done in a multinomial model, 
                      conditioning on the overall level as documented in
                      Martinez Miranda, Nielsen and Nielsen (2015).
                  "od_poisson_response"
                      Poisson family with log link. Only responses are 
                      used. Inference is done in an over-dispersed Poisson
                      model as documented in Harnau and Nielsen (2017). 
                      Note that limit distributions are t and F, not 
                      normal and chi2.
                  "binomial_dose_response"
                      Binomial family with logit link. Gives a logistic
                      regression.
                  "gaussian_rates"
                      Gaussian family with identity link. The dependent
                      variable are rates.
                  "gaussian_response"
                      Gaussian family with identity link. Gives a 
                      regression on the responses.
                  "log_normal_response"
                      Gaussian family with identity link. Dependent 
                      variable are log responses.
                  "log_normal_rates"
                      Gaussian family with identity link. Dependent 
                      variable are log rates.              
        
        predictor : {'APC', 
                     'AP', 'AC', 'PC', 
                     'Ad', 'Pd', 'Cd', 
                     'A', 'P', 'C',
                     't', 'tA', 'tP', 'tC', 
                     '1'}
                           
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
        
        
                         
        Returns
        -------
        
        The following variables attached to self.
        
        deviance : float
                   Corresponds to the deviance of 'fit.deviance', except for Gaussian
                   and log-normal models where it is - 2 * log-likelihood, rather 
                   than RSS.
        
        RSS : float (only for Gaussian and log-normal models)
              Sum of squared residuals, on the log-scale for log-nromal models.
        
        s2 : float (only for Gaussian and log-normal models)
             Normal variance estimator 'RSS / df_resid'.
        
        sigma2 : float (only for Gaussian and log-normal models)
                 Maximum likelihood normal variance estimator 'RSS / n'.
        
        para_table : pandas.DataFrame
                     Dataframe with four columns: coefficients, standard errors,
                     z-stats/t-stats (ratio of coefficients to standard errors) and 
                     p-values. 
        
        cov_canonical : pandas.DataFrame
                        Normalized covariance matrix. For Poisson and over-dispersed
                        Poisson models this is the 
        
        Notes
        -----
        
        'cov_canonical' generally equals 'fit.normalized_cov_params', except for 
        over-dispersed Poisson models when it is adjusted to a multinomial covariance;
        see Harnau and Nielsen (2017).
        
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
        
        Kuang, D., Nielsen, B. and Nielsen, J.P. (2008a) Identification of the 
        age-period-cohort model and the extended chain ladder model. Biometrika 
        95, 979-986. 
    
        Martinez Miranda, M.D., Nielsen, B. and Nielsen, J.P. (2015) Inference 
        and forecasting in the age-period-cohort model with unknown exposure 
        with an application to mesothelioma mortality. Journal of the Royal 
        Statistical Society A 178, 29-55.
        
        Nielsen, B. (2014) Deviance analysis of age-period-cohort models. 
        Nuffield Discussion Paper 2014-W03
        
        Nielsen, B. (2015) apc: An R package for age-period-cohort analysis. 
        R Journal 7, 52-64.
        
        """
        
        ## Model family
        supported_families = ("binomial_dose_response", 
                              "poisson_response", "od_poisson_response",
                              "gaussian_rates", "gaussian_response", 
                              "log_normal_rates", "log_normal_response")
        if family not in supported_families:
            raise ValueError("\'family\' not understood. Check the help.")
        
        ## Model predictor
        supported_predictors = ("APC", "AP", "AC", "PC", "Ad", "Pd", "Cd", 
                                "A", "P", "C", "t", "tA", "tP", "tC", "1")
        if predictor not in supported_predictors:
            raise ValueError("\'predictor\' not understood. Check the help.")
            
        # Get the design matrix
        design = self._get_design(predictor, design_components)
        
        # Get the data
        response = self.data_vector['response']        
        if family is 'binomial_dose_response':
            dose = self.data_vector['dose']
        elif family in ('gaussian_rates', 'log_normal_rates'):
            rate = self.data_vector['rate']
        
        # Create the glm object
        if family is 'binomial_dose_response':
            glm = sm.GLM(pd.concat((response, dose - response), axis = 1), design, 
                         family=sm.families.Binomial(sm.families.links.logit))    
        elif family in ('poisson_response', 'od_poisson_response'):
            glm = sm.GLM(response, design, 
                         family=sm.families.Poisson(sm.families.links.log))        
        elif family is 'gaussian_response':
            glm = sm.GLM(response, design, 
                         family = sm.families.Gaussian(sm.families.links.identity))
        elif family is 'gaussian_rates':
            glm = sm.GLM(rate, design, 
                         family = sm.families.Gaussian(sm.families.links.identity))
        elif family is 'log_normal_response':
            glm = sm.GLM(np.log(response), design, 
                         family = sm.families.Gaussian(sm.families.links.identity))
        elif family is 'log_normal_rates':
            glm = sm.GLM(np.log(rate), design, 
                         family = sm.families.Gaussian(sm.families.links.identity))
        
        # Fit the model
        fit = glm.fit()
        
        # Gather results (Note: the 'fit' object does NOT necessarily provide
        # correct (asymptotic) results! For Poisson and over-dispersed Poisson 
        # the errors are based on large n asymptotics which are invalid here)     
        xi_dim = design.shape[1]
        
        coefs_canonical = fit.params
        coefs_canonical.rename('coef', inplace = True)
        cov_canonical = fit.normalized_cov_params
        
        if family not in ('poisson_response', 'od_poisson_response'):            
            std_errs = fit.bse
            std_errs.rename('std err', inplace = True)
            t_stat = fit.tvalues
            t_stat.rename('z', inplace = True)
            p_values = fit.pvalues
            p_values.rename('P>|z|', inplace = True)            
        else:         
            # Adjust covariance matrix
            c22 = cov_canonical.iloc[1:xi_dim,1:xi_dim]
            c21 = cov_canonical.iloc[1:xi_dim,0]
            c11 = cov_canonical.iloc[0,0]
            cov_canonical.iloc[1:xi_dim,1:xi_dim] = c22 - np.outer(c21,c21)/c11
            cov_canonical.iloc[0,:] = 0
            cov_canonical.iloc[:,0] = 0
            
            if family is 'od_poisson_response':
                cov_canonical = cov_canonical * (fit.deviance / fit.df_resid)
                    
            std_errs = pd.Series(np.sqrt(np.diag(cov_canonical)),
                                 index = cov_canonical.index)
            std_errs.rename('std err', inplace=True)
            
            t_stat = coefs_canonical.divide(std_errs)
            t_stat.rename('t stat', inplace=True)
            
            if family is 'poisson_response':
                t_stat.rename('z', inplace = True)
                p_values = 2 * pd.Series(1 - stats.norm.cdf(abs(t_stat)), 
                                     index = coefs_canonical.index)
                p_values.rename('P>|z|', inplace=True)
            else:
                t_stat.rename('t', inplace = True)
                p_values = 2 * pd.Series(1 - stats.t.cdf(abs(t_stat), fit.df_resid), 
                                     index = coefs_canonical.index)
                p_values.rename('P>|t|', inplace=True)
            
            # In mixed parametrization cannot make inference about level.
            t_stat[0] = std_errs[0] = p_values[0] = np.nan
            
        para_table = pd.concat((coefs_canonical, std_errs, t_stat, p_values), 
                                    axis = 1)
        df_resid = fit.df_resid
        cov_canonical = cov_canonical
        fitted_values = fit.fittedvalues
        
        if family in ("gaussian_rates", "gaussian_response", 
                      "log_normal_rates", "log_normal_response"):
            RSS = fit.deviance
            sigma2 = RSS / fit.nobs
            s2 = RSS / fit.df_resid
            deviance = fit.nobs * (1 + np.log(2 * np.pi) + np.log(sigma2))
            aic = fit.aic
        else:
            deviance = fit.deviance        
        
        if family in ("log_normal_rates", "log_normal_response"):
            fitted_values = np.exp(fitted_values)
        
        output = {'para_table': para_table,
                  'df_resid': df_resid,
                  'predictor': predictor,
                  'family': family,
                  'design': design,
                  'deviance': deviance, 
                  'fitted_values': fitted_values}
        try:
            output['RSS'] = RSS
            output['sigma2'] = sigma2
            output['s2'] = s2
            output['aic'] = aic
        except NameError:
            pass
        
        if attach_to_self:
            for key, value in output.items():
                setattr(self, key, value)
        else:
            return output

        
    def fit_table(self, family=None, reference_predictor=None, 
                  design_components=None, attach_to_self=True):
        
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
        
        family : see help for Model().fit() for a description (optional)
                 If not specified attempts to read from self.
        
        model_design : see help for Model().fit() for a description (optional)
                       If not specified attempts to read from self.
        
        design_components : pandas.DataFrame (optional)
                            Output of Model()._get_design_components(). Can 
                            speed up computations if 'fit_table' is called 
                            repeatedly. If not provided this is computed 
                            internally.
                            
        attach_to_self : bool (optional)
                         Default True. If this is True the deviance_table is
                         attached to self.deviance_table. If False the table
                         is returned.
        
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
            p-value of the deviance, -2logL, compared to a chi-square. Left out
            in Gaussian case which has no saturated model.
            
        "LR_vs_{ref}"
            The likelihood ratio statistic against the reference model.
            
         "df_vs_{ref}"
            Degrees of freedom against the reference model.
        "P>chi_sq"
            p-value of log likelihood ratio statistic compared to a chi-square.
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
        >>> model.data_from_df(loss_TA(), time_adjust=1)
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
                raise AttributeError("Could not infer 'reference_predictor'. Either " + 
                                     "specify as input or fit a model first")
        
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
            
            family = ref_fit['family']
            reference_predictor = ref_fit['predictor']
            
            ref_deviance = ref_fit['deviance']
            sub_deviance = sub_fit['deviance']
            ref_df = ref_fit['df_resid']
            sub_df = sub_fit['df_resid']
            try:
                sub_aic = sub_fit['aic']
            except KeyError:
                pass
            
            if ref_fit['predictor'] == sub_fit['predictor']:
                LR = np.nan
                df = np.nan
                p_LR = np.nan
            else:
                LR = sub_deviance - ref_deviance
                df = sub_df - ref_df
                p_LR = 1 - stats.chi2.cdf(LR, df)
            
            if family in ("gaussian_rates", "gaussian_response", 
                          "log_normal_rates", "log_normal_response"):
                keys = ('-2logL', 'df_resid', 
                        'LR_vs_{}'.format(reference_predictor), 
                        'df_vs_{}'.format(reference_predictor), 
                        'P>chi_sq', 'aic')
                values = (sub_deviance, sub_df, LR, df, p_LR, sub_aic)
            elif family is 'poisson_response':
                p_deviance = 1 - stats.chi2.cdf(sub_deviance, sub_df)
                keys = ('deviance', 'df_resid', 'P>chi_sq', 
                        'LR_vs_{}'.format(reference_predictor), 
                        'df_vs_{}'.format(reference_predictor))
                values = (sub_deviance, sub_df, p_deviance, LR, df, p_LR)                
            elif family is 'od_poisson_response':
                if ref_fit['predictor'] == sub_fit['predictor']:
                    F = np.nan
                    p_F = np.nan
                    p_deviance = 1 - stats.chi2.cdf(sub_deviance, sub_df)
                else:                    
                    F = (LR/df) / (ref_deviance/ref_df)
                    p_F = 1 - stats.f.cdf(F, df, ref_df)
                    p_deviance = np.nan
                keys = ('deviance', 'df_resid', 'P>chi_sq', 
                        'LR_vs_{}'.format(reference_predictor), 
                        'df_vs_{}'.format(reference_predictor), 
                        'F_vs_{}'.format(reference_predictor),
                        'P>F')
                values = (sub_deviance, sub_df, p_deviance, LR, df, F, p_F)
                
            return collections.OrderedDict(zip(keys, values))
        
        ref_fit = self.fit(family, reference_predictor, design_components, 
                               attach_to_self=False)
        
        deviance_table = pd.DataFrame([
            _fill_row(ref_fit, self.fit(
                family, sub_pred, design_components, attach_to_self=False)
                     ) for sub_pred in [reference_predictor] + sub_predictors],
            index = [reference_predictor] + sub_predictors)
        
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
        
        simplify_ranges : {'start', 'mean', 'end', False}
        
        """
        try:
            col_from_to = col_range.str.split('-', expand=True).astype(int)
        except AttributeError: #not a column range            
            return col_range
        if simplify_ranges == 'start':
            col_simple = col_from_to.iloc[:,0].astype(int)
        elif simplify_ranges == 'end':
            col_simple = col_from_to.iloc[:,1].astype(int)
        elif simplify_ranges == 'mean':
            col_simple = col_from_to.mean(axis=1).round().astype(int)
        else:
            raise ValueError("'simplify_range' must be one of 'start', " +
                             "'mean', 'end' or False")
        
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
        ...                      sheetname = ['response', 'rates'], index_col = 0)
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
        Takes a 'data_vector' with one column and maps it into a two-dimensional
        array in 'space', e.g. 'CP' for cohort-period.
        """
        row_idx, col_idx = space[0], space[1]
        space_dict = {'A': 'Age', 'P': 'Period', 'C': 'Cohort'}
        
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
        
        figsize : float tuple or list, optional
                  Specifies the figure size. If left empty matplotlib determines this
                  internally.
        
        space : {'AC', 'AP', 'PA', 'PC', 'CA', 'CP'}, optional
                Specifies what goes on the axes (A = Age, P = period, C = cohort). 
                By default this is set to 'self.data_format'.
            
        **kwargs : any kwargs that seaborn.heatmap can handle, optional
                   The kwargs are fed through to seaborn.heatmap. Note that these are
                   applied to all heatmap plots symmetrically.
                   
        Returns
        -------
        
        Matplotlib figure attached to self.plotted_data_heatmaps
        
        Examples
        --------

        >>> import pandas as pd
        >>> data = pd.read_excel('./data/Belgian_lung_cancer.xlsx', 
        ...                      sheetname = ['response', 'rates'], index_col = 0)
        >>> import apc
        >>> model = apc.Model()
        >>> model.data_from_df(data['response'], rate=data['rates'], 
        ...                    data_format='AP')
        >>> model.plot_data_heatmaps()
        
        """    
        
        try:
            data_vector = self.data_vector
        except AttributeError:
            raise AttributeError("Could not find 'data_vector', run " + 
                                 "Model().data_from_df() first.")
        
        if space is None:
            space = self.data_format
        
        idx_names = data_vector.index.names
        
        if simplify_ranges:
            _simplify_range = self._simplify_range
            data_vector = data_vector.reset_index()
            data_vector[idx_names] = data_vector[idx_names].apply(
                lambda col: _simplify_range(col, simplify_ranges))
            data_vector.set_index(idx_names, inplace=True)
        
        fig, ax = plt.subplots(nrows=1, ncols=data_vector.shape[1], sharey=True,
                               figsize=figsize)

        for i, col in enumerate(data_vector.columns):
            try:
                active_ax = ax[i]
            except TypeError:
                active_ax = ax
            _vector_to_array = self._vector_to_array
            col_vector = data_vector[col]
            col_array = _vector_to_array(col_vector, space)
            sns.heatmap(ax=active_ax, data=col_array, **kwargs)   
            active_ax.set_title(col_vector.name)
            if i > 0:
                active_ax.set_ylabel('')
            
        fig.tight_layout()
        
        self.plotted_data_heatmaps = fig


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
        
        aggregate : {'mean', 'sum'}, optional
                    Determines whether aggregation to reduce the number of groups is done
                    by summings or averaging. Default is 'mean'.        
        
        simplify_ranges : {'start', 'mean', 'end', False}, optional
                          Default is 'mean'. If the time indices are ranges, such as 
                          1955-1959, this determines if and how those should be 
                          transformed. Allows for prettier axis labels. Default is 'False'.
        
        logy : bool, optional
               Specifies whether the y-axis uses a log-scale. Default is 'False'.
                    
        figsize : float tuple or list, optional
                  Specifies the figure size. If left empty matplotlib determines this
                  internally.
        
        Notes
        -----
        
        Parts of the description are taken from the R package apc.
        
        
        Returns
        -------
        
        Matplotlib figure(s) attached to self.plotted_data_within. If dose/rate is available
        this is a dictionary with separate figures for response, dose, and rate as values.
        
        Examples
        --------
        
        >>> import pandas as pd
        >>> data = pd.read_excel('./data/Belgian_lung_cancer.xlsx', 
        ...                      sheetname = ['response', 'rates'], index_col = 0)
        >>> import apc
        >>> model = apc.Model()
        >>> model.data_from_df(data['response'], rate=data['rates'], 
        ...                    data_format='AP')
        >>> model.plot_data_within(figsize=(10,6))
                
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
            