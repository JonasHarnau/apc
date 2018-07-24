import numpy as np
import pandas as pd

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

def bootstrap_forecast(model, qs=[0.75, 0.9, 0.95, 0.99], B=999, adj_residuals=True, seed=None):
    """
    Generate bootstrap forecasts
    """
    np.random.seed(seed)
    #generate Pearson residuals
    rsdls = model.residuals['pearson']
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
    
    fc_bootstrap = {'Cell': bs_oosmpl.T.describe(qs).T,
                    'Age': bs_oosmpl.sum(level='Age').T.describe(qs).T,
                    'Cohort': bs_oosmpl.sum(level='Cohort').T.describe(qs).T,
                    'Period': bs_oosmpl.sum(level='Period').T.describe(qs).T,
                    'Total': pd.DataFrame(bs_oosmpl.sum().describe(qs).rename('Total')).T}

    return fc_bootstrap