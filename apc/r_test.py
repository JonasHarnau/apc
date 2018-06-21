import apc
import numpy as np
import pandas as pd
from quad_form_ratio import saddlepoint_cdf_R

def r_test(response, family_null, predictor = 'APC',
           R_stat = 'wls_ls', R_dist = 'wls_ls',  **kwargs):
    """
    R-test for (generalized) log-normal vs over-dispersed Poisson.
    """
    model = apc.Model()
    model.data_from_df(response, **kwargs)
    Z = np.log(model.data_vector)
    
    model.fit('log_normal_response', predictor)
    RSS = model.RSS
    tau_ls = model.fitted_values.sum()
    sqrt_Pi_ls = np.diag(np.sqrt(model.fitted_values/tau_ls))
    X = model.design

    if ('ql' in R_stat) or ('ls' in R_stat) or ('ql' in R_dist):
        model.fit('od_poisson_response', predictor)
        D = model.deviance
        tau_ql = model.fitted_values.sum()
        sqrt_Pi_ql = np.diag(np.sqrt(model.fitted_values/tau_ql)) 
    if ('wls_ls' in R_stat) or ('wls_ls' in R_dist):
        X_star_ls = sqrt_Pi_ls.dot(X)
        Z_star_ls = sqrt_Pi_ls.dot(Z) 
        wls_ls_fit = np.linalg.lstsq(X_star_ls, Z_star_ls, rcond=0.) # X_star_ls is never singular
        xi_star_ls, RSS_star_ls = wls_ls_fit[0][:, 0], wls_ls_fit[1][0]
        fitted_star_ls = np.exp(X.dot(xi_star_ls))
        sqrt_Pi_star_ls = np.diag(np.sqrt(fitted_star_ls/fitted_star_ls.sum()))
    if ('wls_ql' in R_stat) or ('wls_ql' in R_dist):
        X_star_ql = sqrt_Pi_ql.dot(X)
        Z_star_ql = sqrt_Pi_ql.dot(Z)
        wls_ql_fit = np.linalg.lstsq(X_star_ql, Z_star_ql, rcond=0.)
        xi_star_ql, RSS_star_ql = wls_ql_fit[0][:, 0], wls_ql_fit[1][0]
        fitted_star_ql = np.exp(X.dot(xi_star_ql))
        sqrt_Pi_star_ql = np.diag(np.sqrt(fitted_star_ql/fitted_star_ql.sum()))

    if R_stat == 'ls': R = tau_ls * RSS / D
    elif R_stat == 'ql': R = tau_ql * RSS / D
    elif R_stat == 'wls_ls': R = RSS / RSS_star_ls
    elif R_stat == 'wls_ql': R = RSS / RSS_star_ql
    else: raise ValueError('"R_stat" must be one of "ls", "ql", "wls_ls" or "wls_ql"')
        
    if R_dist == 'ls': sqrt_Pi = sqrt_Pi_ls
    elif R_dist == 'ql': sqrt_Pi = sqrt_Pi_ql
    elif R_dist == 'wls_ls': sqrt_Pi = sqrt_Pi_star_ls
    elif R_dist == 'wls_ql': sqrt_Pi = sqrt_Pi_star_ql
    else: raise ValueError('"R_dist" must be one of "ls", "ql", "wls_ls" or "wls_ql"')

    Q, _ = np.linalg.qr(X)
    M = np.identity(model.n) - Q.dot(Q.T)
    
    X_star = sqrt_Pi.dot(X)        
    Q_star, _ = np.linalg.qr(X_star)
    M_star = np.identity(model.n) - Q_star.dot(Q_star.T)
    
    A_ln, B_ln = M, sqrt_Pi.dot(M_star).dot(sqrt_Pi)
    A_odp, B_odp = np.linalg.inv(sqrt_Pi).dot(M).dot(np.linalg.inv(sqrt_Pi)), M_star

    if family_null == 'gen_log_normal_response':
        p_value = 1 - saddlepoint_cdf_R(A_ln, B_ln, [R]).iloc[0]
        power_at_R = 1 - saddlepoint_cdf_R(A_odp, B_odp, [R]).iloc[0]
    elif family_null == 'od_poisson_response':
        p_value = saddlepoint_cdf_R(A_odp, B_odp, [R]).iloc[0]
        power_at_R = saddlepoint_cdf_R(A_ln, B_ln, [R]).iloc[0]
    else: raise ValueError('"family_null" must be "gen_log_normal_response" ' + 
                           'or "od_poisson_response')

    return {'R_stat' : R, 'p_value' : p_value, 'power_at_R': power_at_R}