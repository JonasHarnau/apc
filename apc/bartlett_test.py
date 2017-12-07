import numpy as np
from scipy import stats

def bartlett_test(models):
    """
    Performs a Bartlett test.
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
    output = {'B': LR/C, 'LR': LR, 'C': C, 'm': m, 'p_value': p_value}
    return output