"""Bartlett test for common dispersion."""

import numpy as np
from scipy import stats


def bartlett_test(models):
    """
    Bartlett test for common dispersion.

    Performs an Bartlett test  for common dispersion between the models. The
    idea for age-period-cohort models is described in Harnau (2018). The test
    checks whether we can reject that the dispersion is common across models.

    Parameters
    ----------
    models : list
        List of fitted apc.Models.

    Returns
    -------
    test_results : dict
        Dictionary with keys `B`, `LR`, `m` and `p_value`.

    See Also
    --------
    Vignette in apc/vignettes/vignette_misspecification.ipynb.

    Notes
    -----
    For interpretation, a small p-value speaks against the hypothesis that the
    dispersion is equal across models.

    Tests are valid for gaussian models (Bartlett 1937), log-normal and
    over-dispersed Poisson (Harnau 2018) and generalized log-normal models
    (Kuang and Nielsen 2018).

    References
    ----------
    - Bartlett, M. S. (1937). Properties of Sufficiency and Statistical Tests.
    Proceedings of the Royal Society A: Mathematical, Physical and Engineering
    Sciences, 160(901), 268–282.
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
    >>> apc.bartlett_test(sub_models)

    """
    # Check if all models have the same family
    families = [model.family for model in models]
    if families.count(families[0]) != len(families):
        raise ValueError('Model families must match ' +
                         'across models.')
    s2 = np.array([model.s2 for model in models])
    df = np.array([model.df_resid for model in models])
    df_sum = np.sum(df)
    s_bar = s2.dot(df)/df_sum
    m = len(models)
    LR = df_sum * np.log(s_bar) - df.dot(np.log(s2))
    C = 1 + 1/(3*(m-1)) * (np.sum(1/df) - 1/df_sum)
    p_value = stats.distributions.chi2.sf(LR/C, df=m-1)
    test_results = {'B': LR/C, 'LR': LR, 'C': C, 'm': m, 'p_value': p_value}
    return test_results
