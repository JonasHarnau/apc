import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from .get_design import get_design
from ._ReturnValue import _ReturnValue

def fit_model(formatted_data, model_family, model_design, design_components = None):
    
    """
        
    Fits an age-period-cohort model.
    
    The model is parametrised in terms of the canonical parameter introduced by
    Kuang, Nielsen and Nielsen (2008), see also the implementation in Martinez
    Miranda, Nielsen and Nielsen (2015), and Nielsen (2014, 2015). This
    parametrisation has a number of advantages: it is freely varying, it is the
    canonical parameter of a regular exponential family, and it is invariant to
    extentions of the data matrix.

    'fit_model' can be be used for all three age period cohort factors, or for
    submodels with fewer of these factors. It can be used in a pure response 
    setting or in a dose-response setting. Can handle binomial, Gaussian, log-
    normal, over-dispersed Poisson and Poisson models.

    
    Parameters
    ----------
    
    formatted_data : output of 'apc.format_data'
    
    model_family : {"binomial_dose_response", "poisson_response", 
                    "od_poisson_response", "poisson_dose_response",
                    "gaussian_rates", "gaussian_response", "log_normal_rates",
                    "log_normal_response"}
                    Specifies the family used when calling
                    'statsmodels.api.glm'.
                    "poisson_response"
                        Poisson family with log link. Only responses are used.
                        Inference is done in a multinomial model, conditioning
                        on the overall level as documented in Martinez Miranda,
                        Nielsen and Nielsen (2015).
                    "od_poisson_response"
                        Poisson family with log link. Only responses are used.
                        Inference is done in an over-dispersed Poisson model as
                        documented in Harnau and Nielsen (2016). Note that limit
                        distributions are t and F not normal and chi2.
                    "poisson_dose_response"
                        Poisson family with log link and doses as offset.
                    "binomial_dose_response"
                        Binomial family with logit link. Gives a logistic
                        regression.
                    "gaussian_rates"
                        Gaussian family with identity link. The dependent
                        variable are rates.
                    "gaussian_response"
                        Gaussian family with identity link. Gives a regression
                        on the responses.
                    "log_normal_response"
                        Gaussian family with identity link. Dependent variable
                        are log responses.
                    "log_normal_rates"
                        Gaussian family with identity link. Dependent variable
                        are log rates.
                        
                        
    
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
    
    Class
    Apart from the inputs, the class contains the following.
    
    deviance : float
               Corresponds to the deviance of 'fit.deviance', except for Gaussian and
               log-normal models where it is - 2 * log-likelihood, rather than RSS.
    
    fit :  output from statsmodels.api.GLM().fit()
    
    RSS : float or 'None'
          For Gaussian and log-normal models, this is the sum of squared residuals, else
          'None'
    
    s2 : float or 'None'
         For Gaussian and log-normal models, this is the unbiased normal variance
         estimator 'RSS / fit.df_resid', else 'None'.
    
    sigma2 : float or 'None'
             For Gaussian and log-normal models, this is the maximum likelihood normal
             variance estimator 'RSS / fit.nobs', else 'None'.
    
    coefs_canonical_table : pandas.DataFrame
                            Dataframe with four columns: coefficients, standard errors,
                            t-values (ratio of coefficients to standard errors) and p-
                            values. 
    
    cov_canonical : pandas.DataFrame
                    Normalized covariance matrix. 
    
    
    Notes
    -----
    
    'cov_canonical' generally equals 'fit.normalized_cov_params', except for over-
    dispersed Poisson models when it is adjusted to a multinomial covariance; see Harnau
    and Nielsen (2016)
    
    'deviance' for Gaussian and log-normal models equals - 2 * log-likelihood, not RSS.
        
    p-values for 'coefs_canonical_table' are generally computed from a normal
    distribution. The exception is an over-dispersed Poisson model for which these come
    from a t distribution; see Harnau and Nielsen (2016).
    
    The description is largely taken from the R package apc.
    
    
    Examples
    --------
    
    >>> import apc
    >>> data = apc.data_Italian_bladder_cancer()
    >>> model = apc.fit_model(data, 'poisson_dose_response', 'APC)
    >>> model.coefs_canonical_table
    
    >>> import apc
    >>> data = apc.data_loss_TA()
    >>> model = apc.fit_model(data, 'od_poisson_response', 'APC)
    >>> model.coefs_canonical_table
    
    
    References
    ----------
    
    Harnau, J. and Nielsen, B. (2016) Asymptotic theory for over-dispersed age-
    period-cohort and extended chain ladder models. Mimeo.
    
    Kuang, D., Nielsen, B. and Nielsen, J.P. (2008a) Identification of the age-
    period-cohort model and the extended chain ladder model. Biometrika 95, 979-
    986. 
    
    Martinez Miranda, M.D., Nielsen, B. and Nielsen, J.P. (2015) Inference and
    forecasting in the age-period-cohort model with unknown exposure with an
    application to mesothelioma mortality. Journal of the Royal Statistical
    Society A 178, 29-55.
    
    Nielsen, B. (2014) Deviance analysis of age-period-cohort models.
    
    Nielsen, B. (2015) apc: An R package for age-period-cohort analysis. R
    Journal 7, 52-64.
    
    """
    
    model_design_list = ["APC", "AP", "AC", "PC", "Ad", "Pd", 
        "Cd", "A", "P", "C", "t", "tA", "tP", "tC", "1"]
    
    model_family_list = ["binomial_dose_response", "poisson_response", 
        "od_poisson_response", "poisson_dose_response", "gaussian_rates", 
        "gaussian_response", "log_normal_rates", "log_normal_response"]
    
    model_family_gaussian = ["gaussian_rates", "gaussian_response", 
        "log_normal_rates", "log_normal_response"]
    
    model_family_mixed = ["poisson_response", "od_poisson_response"]
    
    if model_design not in model_design_list:
        raise ValueError("\'model_design\' has wrong argument")
    
    if model_family not in model_family_list:
        raise ValueError('\'model_family\' has wrong argument')
 
    mixed_par = model_family in model_family_mixed and model_design is not '1'
    mixed_par_1 = model_design is '1' and model_family in model_family_mixed
    
    response = formatted_data.data_as_vector['Response']
    
    if formatted_data.dose is not None:
        dose = formatted_data.data_as_vector['Dose']
        rate = formatted_data.data_as_vector['Rate']
    
    design = get_design(formatted_data, model_design, design_components)
    
    if model_family is 'binomial_dose_response':
        glm = sm.GLM(pd.concat((response, dose - response), axis = 1), 
                     design, family=sm.families.Binomial(sm.families.links.logit))
    
    if model_family in model_family_mixed:
        glm = sm.GLM(response, design, family=sm.families.Poisson(sm.families.links.log))
    
    if model_family is 'poisson_dose_response':
        glm = sm.GLM(response, design, family = sm.families.Poisson(sm.families.links.log), offset = np.log(dose))
    
    if model_family is 'gaussian_response':
        glm = sm.GLM(response, design, family = sm.families.Gaussian(sm.families.links.identity))
    
    if model_family is 'gaussian_rates':
        glm = sm.GLM(rate, design, family = sm.families.Gaussian(sm.families.links.identity))
    
    if model_family is 'log_normal_response':
        glm = sm.GLM(np.log(response), design, family = sm.families.Gaussian(sm.families.links.identity))
    
    if model_family is 'log_normal_rates':
        glm = sm.GLM(np.log(rate), design, family = sm.families.Gaussian(sm.families.links.identity))
    
    fit = glm.fit()
                     
    xi_dim = design.shape[1]
        
    coefs_canonical = fit.params
    coefs_canonical.rename('coef', inplace = True)
    cov_canonical = fit.normalized_cov_params
    
    if not mixed_par and not mixed_par_1:
        
        std_errs = fit.bse
        std_errs.rename('std err', inplace = True)
        t_values = fit.tvalues
        t_values.rename('z', inplace = True)
        p_values = fit.pvalues
        p_values.rename('P>|z|', inplace = True)
        
    else:
        
        if mixed_par:
            c22 = cov_canonical.iloc[1:xi_dim,1:xi_dim]
            c21 = cov_canonical.iloc[1:xi_dim,0]
            c11 = cov_canonical.iloc[0,0]
            cov_canonical.iloc[1:xi_dim,1:xi_dim] = c22 - np.outer(c21,c21)/c11
        
        cov_canonical.iloc[0,:] = 0
        cov_canonical.iloc[:,0] = 0
        
        std_errs = pd.Series(np.sqrt(np.diag(cov_canonical)),
                               index = cov_canonical.index)
        std_errs.rename('std err', inplace = True)
        
        if model_family is 'od_poisson_response':
            std_errs = std_errs * np.sqrt(fit.deviance / fit.df_resid)
    
        t_values = coefs_canonical.divide(std_errs)
        t_values.rename('t', inplace = True)
        
        if model_family is not 'od_poisson_response':
            p_values = 2 * pd.Series(1 - stats.norm.cdf(abs(t_values)), 
                                 index = coefs_canonical.index)
        else:
            p_values = 2 * pd.Series(1 - stats.t.cdf(abs(t_values), fit.df_resid), 
                                 index = coefs_canonical.index)
        
        p_values.rename('P>|t|', inplace = True)
        
        t_values[0] = std_errs[0] = p_values[0] = np.nan
        
    coefs_canonical_table = pd.concat((coefs_canonical, std_errs,
                                      t_values, p_values), axis = 1)
    
    if model_family in model_family_gaussian:
        RSS = fit.deviance
        sigma2 = RSS / fit.nobs
        s2 = RSS / fit.df_resid
        deviance = fit.nobs * (1 + np.log(2 * np.pi) + np.log(sigma2))
    else:
        RSS = None
        sigma2 = None
        s2 = None
        deviance = fit.deviance
    
    return _ReturnValue(fit = fit, deviance = deviance, model_family = model_family, 
                        RSS = RSS, model_design = model_design, s2 = s2, sigma2= sigma2,
                        coefs_canonical_table = coefs_canonical_table,
                        cov_canonical = cov_canonical,
                        formatted_data = formatted_data)
