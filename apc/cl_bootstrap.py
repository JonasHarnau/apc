import numpy as np
import pandas as pd
import apc

def _chain_ladder(df, target_idx):
    """
    Generates chain-ladder point forecasts for arrays.

    The function allows for vectorized chain-ladder estimation for more then one
    sample. Still, it is not quite tuned for top efficiency. However, it can deal
    with negative values as sometimes encountered in bootstrap draws.
    
    
    Parameters
    ----------
    
    df : pandas.DataFrame with age-period-cohort MultiIndex
         A dataframe with an index such as the data_vector output by 
         Model().data_from_df() or the draws output by Model().simulate().

    target_idx : pandas.MultiIndex of the target array
                 Index corresponding to the cells of the target array for
                 forecasting.
                 
    
    Returns
    -------
    
    pandas.DataFrame with index corresponding to target_idx and as many columns
    as df has. Contains the chain-ladder point forecasts.

    """
    df = df.reset_index().set_index(['Age', 'Cohort', 'Period']).sort_index()
    age_min, age_max = df.index.levels[0].min(), df.index.levels[0].max()
    coh_min, coh_max = df.index.levels[1].min(), df.index.levels[1].max()
    per_max = df.index.levels[2].max()
    time_adjust = coh_max - per_max + 1
    
    # row-sums in a run-off triangle
    coh_sums = df.sum(level='Cohort')

    # compute development factors
    dev_fctrs = []
    for i in np.arange(age_min, age_max):
        dev_fctrs.append((df.loc[pd.IndexSlice[:age_max+1-i, :coh_min+i-1], :].sum()
                         /df.loc[pd.IndexSlice[:age_max-i, :coh_min+i-1], :].sum())
                        )
    dev_fctrs = pd.DataFrame(dev_fctrs, index=np.arange(age_max,age_min,-1))

    fc_point = pd.DataFrame(None, index=target_idx, columns=df.columns)
    fc_point = fc_point.reset_index().set_index(
        ['Age', 'Cohort', 'Period']).sort_index()

    # produce forecasts
    for k in np.arange(coh_min+1,coh_max+1):
        for i in np.arange(age_max,age_max-(k-coh_min), -1):
            fc_point.loc[pd.IndexSlice[i, k, :], :] = (
                coh_sums.loc[k, :] 
                * dev_fctrs.loc[np.arange(per_max-k+time_adjust+1,i), :].prod() 
                * (dev_fctrs.loc[i, :]-1)
            ).values

    return fc_point.astype(float)

def bootstrap_forecast(data, quantiles=[0.75, 0.9, 0.95, 0.99], B=999, 
                       adj_residuals=True, seed=None):
    """
    Generate bootstrap forecasts.
    
    Generates bootstrap forecasts for chain-ladder run-off triangles based on 
    the bootstrap technique in England and Verrall (1999) and England (2002).
    
    
    Parameters
    ----------
    
    data : pandas.DataFrame
           Data for a run-off triangle that passes Model.data_from_df().
    
    quantiles : iterable of floats in (0, 1), optional
                The quantiles for which the distribution forecast should be computed. 
                (Default is [0.75, 0.9, 0.95, 0.99].)
                
    B : int, optional
        The number of bootstrap draws. (Default is 999.)

    adj_residuals : bool, optional
                    Determines whether residuals are adjusted by n/df_resid. (Default 
                    is True.)
    
    seed : int, optional
       The random seed used to generate the draws.
    
    
    Returns
    -------
    
    dictionary of pandas.DataFrame's with keys 'Cell', 'Age', 'Period', 'Cohort', 'Total'.
    DataFrames contain descriptive statistics over bootstrap draws, including mean, 
    standard deviation, median etc.
    
    
    See also
    --------
    
    Vignettes in apc/vignettes/vignette/over_dispersed_apc.ipynb and 
    apc/vignettes/vignette/generazlied_log_normal.ipynb.
    
    
    Notes
    -----
    
    The function uses the chain-ladder technique for computation to both speed
    things up and to allow for negative values which are excluded by the standard
    Poisson fit procedure with statsmodels. The chain-ladder technique corresponds 
    to an age-cohort Poisson (and over-dispersed Poisson) model in run-off triangles.
    As such, the function cannot handle any other predictors or data shapes.
    
    
    References
    ----------
    
    England, P., & Verrall, R. (1999). Analytic and bootstrap estimates of 
    prediction errors in claims reserving. Insurance: Mathematics and Economics, 
    25(3), 281–293. 
    
    England, P. D. (2002). Addendum to “Analytic and bootstrap estimates of
    prediction errors in claims reserving.” Insurance: Mathematics and Economics,
    31(3), 461–466.
    
    Examples
    --------
    
    >>> apc.bootstrap_forecast(apc.loss_TA())
    
    """
    
    model = apc.Model()
    model.data_from_df(data, data_format='CL')
    model.fit('od_poisson_response', 'AC')
    
    np.random.seed(seed)
    rsdls = model.residuals['pearson'].copy()
    if adj_residuals:
        rsdls *= np.sqrt(model.n/model.df_resid)

    rsdl_draws = pd.DataFrame(
        np.random.choice(rsdls, size=[model.n, B]), index=rsdls.index)
    rlzd_insmpl = rsdl_draws.multiply(
        np.sqrt(model.fitted_values), axis=0).add(model.fitted_values, axis=0)

    # chain ladder estimation to deal with negatives
    target_index = model._get_fc_design('AC').index
    bs_fc_point = _chain_ladder(rlzd_insmpl, target_index)

    # now for the process error
    scale = (rsdls**2).sum()/model.df_resid
    shape = (bs_fc_point.abs()/scale)
    bs_oosmpl = np.sign(bs_fc_point) * np.random.gamma(shape, scale)
    
    fc_bootstrap = {'Cell': bs_oosmpl.T.describe(quantiles).T,
                    'Age': bs_oosmpl.sum(level='Age').T.describe(quantiles).T.sort_index(),
                    'Cohort': bs_oosmpl.sum(level='Cohort').T.describe(quantiles).T.sort_index(),
                    'Period': bs_oosmpl.sum(level='Period').T.describe(quantiles).T.sort_index(),
                    'Total': pd.DataFrame(bs_oosmpl.sum().describe(quantiles).rename('Total')).T}

    return fc_bootstrap