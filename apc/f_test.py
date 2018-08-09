"""F-test for common linear predictors."""

import numpy as np
from scipy import stats
import pandas as pd


def f_test(model_full, sub_models):
    """
    F-test for common linear predictors.

    Performs an F-test for common linear predictors between model_full
    and the combined sub_models. The idea is described in Harnau (2018).
    The test checks whether we can reject that the linear predictors on
    estimated on the sub_models can be estimated jointly on model_full.

    Parameters
    ----------
    model_full : apc.Model
        The full model with prior call to `Model.fit()``.
    sub_models : list
        List of `apc.Model`s with prior calls to `Model.fit()`. Should be
        sub-models of `model_full`. The `Model.data_vector`s have to
        combine to the `Model.data_vector` of `full_model`.

    Returns
    -------
    test_results : dict
        Dictionary with keys `F_stat` and `p_value`.

    See Also
    --------
    Vignette in apc/vignettes/vignette_misspecification.ipynb.

    Notes
    -----
    For interpretation, small p-values speaks against the null hypothesis
    of common linear predictors.

    Tests are valid for gaussian models, log-normal and over-dispersed
    Poisson models (Harnau 2018) and generalized log-normal models (Kuang
    and Nielsen 2018).

    References
    ----------
    - Harnau, J. (2018). Misspecification Tests for Log-Normal and
    Over-Dispersed Poisson Chain-Ladder Models. Risks, 6(2), 25. Open Access:
    https://doi.org/10.3390/RISKS6020025
    - Kuang, D., & Nielsen, B. (2018). Generalized Log-Normal Chain-Ladder.
    ArXiv E-Prints, 1806.05939. Download from http://arxiv.org/abs/1806.05939

    Examples
    --------
    >>> model = apc.Model()
    >>> model.data_from_df(apc.loss_VNJ())
    >>> model.fit('log_normal_response', 'AC')
    >>> sub_models = [model.sub_model(coh_from_to=(1,5)),
    ...               model.sub_model(coh_from_to=(6,10))]
    >>> apc.f_test(model, sub_models)

    """
    sub_families = [sub_model.family for sub_model in sub_models]
    # Check if all models have the same family
    if sub_families.count(model_full.family) != len(sub_families):
        raise ValueError('Model families do not match ' +
                         'across models.')
    # Check if sub-samples combine to full sample
    samples_combined = pd.concat(
        [sub_model.data_vector for sub_model in sub_models]
    ).rename(columns={'response': 'response_sub'})
    merged_dfs = pd.merge(model_full.data_vector.reset_index(),
                          samples_combined.reset_index(),
                          on=['Age', 'Period', 'Cohort'])
    if (merged_dfs['response'] != merged_dfs['response_sub']).any():
        raise ValueError('Sub samples do not combined to ' +
                         'full sample.')
    # Check if all models have the same predictor. This is just a warning
    # since it is not strictly required by the theory.
    sub_preds = [sub_model.predictor for sub_model in sub_models]
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
    test_results = {'F_stat': F, 'p_value': p_value}
    return test_results
