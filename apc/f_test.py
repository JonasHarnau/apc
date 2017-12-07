import numpy as np
from scipy import stats
import pandas as pd

def f_test(model_full, sub_models):
    """
    F-test for common linear predictors.
    """
    sub_families = [sub_model.family for sub_model in sub_models]
    # Check if all models have the same family
    if sub_families.count(model_full.family) != len(sub_families):
        raise ValueError('Model families do not match ' +
                         'across models.')
    # Check if sub-samples combine to full sample
    samples_combined = pd.concat(
        [sub_model.data_vector for sub_model in sub_models]
    )
    def _sort_and_order_index(df):
        return df.reorder_levels(
            ['Age', 'Cohort', 'Period']
        ).sort_index(level='Age')
    if not _sort_and_order_index(model_full.data_vector).equals(
        _sort_and_order_index(samples_combined)):
        raise ValueError('Sub samples do not combined to ' +
                         'full sample.' )
    # Check if all models have the same predcitor
    # This is just a warning since it is not necessary strictly 
    # required by the theory.
    sub_preds = [sub_model.predictor 
                      for sub_model in sub_models]
    if sub_preds.count(model_full.predictor) != len(sub_preds):
        print('Model predictors do not match across models.')
    
    s2_sub = np.array([sub_model.s2 for sub_model in sub_models])
    s2_full = model_full.s2
    df_sub = np.array([sub_model.df_resid for sub_model in sub_models])
    df_full = model_full.df_resid
    n_restrictions = df_full - np.sum(df_sub)
    f_numerator = (df_full * s2_full - s2_sub.dot(df_sub))/n_restrictions
    f_denominator = s2_sub.dot(df_sub)/df_sub.sum()
    F = f_numerator/f_denominator
    p_value = stats.distributions.f.sf(F, dfn=n_restrictions,
                                       dfd=df_sub.sum())
    output = {'F_stat': F, 'p_value': p_value}
    return output