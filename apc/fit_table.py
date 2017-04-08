import numpy as np
import pandas as pd
from scipy import stats
from .fit_model import fit_model
from .get_design_components import get_design_components

def fit_table(formatted_data, model_family, model_design_reference = 'APC'):

    """
        
    Produces an age-period-cohort deviance table.
    
    'fit_table' produces a deviance table for 15 combinations of the three factors and
    linear trends: "APC", "AP", "AC", "PC", "Ad", "Pd", "Cd", "A", "P", "C", "t", "tA",
    "tP", "tC", "1"; see Nielsen (2014) for a discussion of these. See also Nielsen
    (2015) who uses the equivalent function of the R package apc.
    
    
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
                 
    Returns
    -------
    
    pandas.DataFrame
    The dataframe has the following columns.
    
    "-2logL"
        -2 log Likelihood up to some constant. If the model family is Poisson or
        binomial (logistic) this is the same as the glm deviance: That is the difference
        in -2 log likelihood value between estimated model and the saturated model. If
        the model family is Gaussian it is different from the traditional glm deviance.
        Here the -2 log likelihood value is measured in a model with unknown variance,
        which is the standard in regression analysis, whereas in the glm package the
        deviance is the residual sum of squares, which can be interpreted as the -2 log
        likelihood value in a model with variance set to one.
        
    "df_resid"
        Degrees of freedom of residual: n_obs - len(parameter). 
        
    "P>chi_sq"
        p-value of the deviance, -2logL, compared to a chi-square. Left out in Gaussian
        case which has no saturated model.
        
    "LR_vs_{ref}"
        The likelihood ratio statistic against the reference model.
        
     "df_vs_{ref}"
        Degrees of freedom against the reference model.

    "P>chi_sq"
        p-value of log likelihood ratio statistic compared to a chi-square.

    "aic"
        Akaike's "An Information Criterion", minus twice the maximized log-likelihood
        plus twice the number of parameters upto a constant. It is take directly from
        the glm function. 
        
    "F"
        Only included for "od_poisson_response". F statistic: Ratio of 'LR_vs_{ref} /
        df_vs_{ref}' to '-2logL / df_resid'
        
    "P>F"
       Only included for "od_poisson_response". p-value of "F" statistic compared to F-
       distribution. 
    
    
    Examples
    --------
    
    >>> import apc
    >>> data = apc.data_Italian_bladder_cancer()
    >>> apc.fit_table(data, 'poisson_dose_response', 'APC)
    
    >>> import apc
    >>> data = apc.data_Belgian_lung_cancer()
    >>> apc.fit_table(data, 'poisson_dose_response', 'AC')
    
    
    References
    ----------
    
    Harnau, J. and Nielsen, B. (2016) Asymptotic theory for over-dispersed age-
    period-cohort and extended chain ladder models. Mimeo.
    
    Nielsen, B. (2015) apc: An R package for age-period-cohort analysis. R
    Journal 7, 52-64.
    
    """
    
    model_family_list = ["binomial_dose_response", "poisson_response", 
        "od_poisson_response", "poisson_dose_response", "gaussian_rates", 
        "gaussian_response", "log_normal_rates", "log_normal_response"]
    
    model_family_gaussian = ["gaussian_rates", "gaussian_response", 
        "log_normal_rates", "log_normal_response"]
    
    model_family_od = ["od_poisson_response"]
    
    if model_family not in model_family_list:
        raise ValueError('\'model_family\' has wrong argument')
        
    design_components = get_design_components(formatted_data)
    
    if model_design_reference is "APC": 
        submodel_design_list = ["AP", "AC", "PC", "Ad", "Pd", "Cd", "A", 
                                "P", "C", "t", "tA", "tP", "tC", "1"]
    if model_design_reference is "AP":
        submodel_design_list = ["Ad", "Pd", "A", "P", "t", "tA", "tP", "1"]
    if model_design_reference is "AC": 
        submodel_design_list = ["Ad", "Cd", "A", "C", "t", "tA", "tC", "1"]
    if model_design_reference is "PC": 
        submodel_design_list = ["Pd", "Cd", "P", "C", "t", "tP", "tC", "1"]
    if model_design_reference is "Ad": 
        submodel_design_list = ["A", "t", "tA", "1"]
    if model_design_reference is "Pd": 
        submodel_design_list = ["P", "t", "tP", "1"]
    if model_design_reference is "Cd": 
        submodel_design_list = ["C", "t", "tC", "1"]
    if model_design_reference is "A":
        submodel_design_list = ["tA", "1"]
    if model_design_reference is "P": 
        submodel_design_list = ["tP", "1"]
    if model_design_reference is "C": 
        submodel_design_list = ["tC", "1"]
    if model_design_reference is "t": 
        submodel_design_list = ["tA", "tP", "tC", "1"]
    
    def _fit_tab_line_glm(fit_U, fit_R, model_family = model_family):
        
        dev_U = fit_U.deviance
        dev_R = fit_R.deviance
        df_U = fit_U.fit.df_resid
        df_R = fit_R.fit.df_resid
        LR = dev_R - dev_U
        df = df_R - df_U
        aic = fit_R.fit.aic
        if model_family in model_family_gaussian: 
            return np.round((dev_R, df_R, LR, df, 1 - stats.chi2.cdf(LR, df), 
                             aic), decimals=3)
        else:
            return np.round((dev_R, df_R, 1 - stats.chi2.cdf(dev_R, df_R),
                             LR, df, 1 - stats.chi2.cdf(LR, df), aic), 
                            decimals = 3)
    
    fit_reference = fit_model(formatted_data, model_family, model_design_reference,
                              design_components)
    
    fit_tab = pd.DataFrame((_fit_tab_line_glm(fit_reference, 
                                              fit_model(formatted_data, 
                                                        model_family, 
                                                        sub_design, 
                                                        design_components))
                           for sub_design in [model_design_reference] + 
                            submodel_design_list),
                      index = [model_design_reference] + submodel_design_list)
    
    if model_family in model_family_gaussian:
        fit_tab.iloc[0, range(2,5)] = np.nan
    else:
        fit_tab.iloc[0, range(3,6)] = np.nan
                            
    if model_family in model_family_od:
        f_stats = (fit_tab.iloc[:,3]/fit_tab.iloc[:,4]) / (fit_tab.iloc[0,0]/fit_tab.iloc[0,1])
        p_values = pd.Series(np.insert(1 - stats.f.cdf(
            f_stats[1:], fit_tab.iloc[1:,4], fit_tab.iloc[0,1]), 0, np.nan), 
                             index = [model_design_reference] + submodel_design_list)
        fit_tab = pd.concat((fit_tab, f_stats, np.round(p_values, decimals = 3)), axis = 1)
     
    if model_family in model_family_gaussian:
        fit_tab.columns = ['-2logL', 'df_resid', 'LR_vs_{}'.format(model_design_reference), 
                           'df_vs_{}'.format(model_design_reference), 'P>chi_sq', 'aic']
        
    elif model_family in model_family_od:
        fit_tab.columns = ['-2logL', 'df_resid', 'P>chi_sq', 
                     'LR_vs_{}'.format(model_design_reference), 'df_vs_{}'.format(model_design_reference),
                     'P>chi_sq', 'aic', 'F', 'P>F']
    else:
        fit_tab.columns = ['-2logL', 'df_resid', 'P>chi_sq', 
                     'LR_vs_{}'.format(model_design_reference), 'df_vs_{}'.format(model_design_reference),
                     'P>chi_sq', 'aic']
        
    return fit_tab