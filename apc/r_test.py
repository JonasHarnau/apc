import apc
import numpy as np
import pandas as pd
from quad_form_ratio import saddlepoint_cdf_R

def r_test(response, family_null, predictor = 'APC',
              R_est = 'wls_ls', Pi_est = 'wls_ls',  **kwargs):
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

    if ('ql' in R_est) or ('ls' in R_est) or ('ql' in Pi_est):
        model.fit('od_poisson_response', predictor)
        D = model.deviance
        tau_ql = model.fitted_values.sum()
        sqrt_Pi_ql = np.diag(np.sqrt(model.fitted_values/tau_ql)) 
    if ('wls_ls' in R_est) or ('wls_ls' in Pi_est):
        X_star_ls = sqrt_Pi_ls.dot(X)
        Z_star_ls = sqrt_Pi_ls.dot(Z) 
        wls_ls_fit = np.linalg.lstsq(X_star_ls, Z_star_ls, rcond=0.) # X_star_ls is never singular
        xi_star_ls, RSS_star_ls = wls_ls_fit[0][:, 0], wls_ls_fit[1][0]
        fitted_star_ls = np.exp(X.dot(xi_star_ls))
        sqrt_Pi_star_ls = np.diag(np.sqrt(fitted_star_ls/fitted_star_ls.sum()))
    if ('wls_ql' in R_est) or ('wls_ql' in Pi_est):
        X_star_ql = sqrt_Pi_ql.dot(X)
        Z_star_ql = sqrt_Pi_ql.dot(Z)
        wls_ql_fit = np.linalg.lstsq(X_star_ql, Z_star_ql, rcond=0.)
        xi_star_ql, RSS_star_ql = wls_ql_fit[0][:, 0], wls_ql_fit[1][0]
        fitted_star_ql = np.exp(X.dot(xi_star_ql))
        sqrt_Pi_star_ql = np.diag(np.sqrt(fitted_star_ql/fitted_star_ql.sum()))

    if R_est == 'ls': R = tau_ls * RSS / D
    elif R_est == 'ql': R = tau_ql * RSS / D
    elif R_est == 'wls_ls': R = RSS / RSS_star_ls
    elif R_est == 'wls_ql': R = RSS / RSS_star_ql
    else: raise ValueError('"R_est" must be one of "ls", "ql", "wls_ls" or "wls_ql"')
        
    if Pi_est == 'ls': sqrt_Pi = sqrt_Pi_ls
    elif Pi_est == 'ql': sqrt_Pi = sqrt_Pi_ql
    elif Pi_est == 'wls_ls': sqrt_Pi = sqrt_Pi_star_ls
    elif Pi_est == 'wls_ql': sqrt_Pi = sqrt_Pi_star_ql
    else: raise ValueError('"Pi_est" must be one of "ls", "ql", "wls_ls" or "wls_ql"')

    Q, _ = np.linalg.qr(X)
    M = np.identity(model.n) - Q.dot(Q.T)
    
    X_star = sqrt_Pi.dot(X)        
    Q_star, _ = np.linalg.qr(X_star)
    M_star = np.identity(model.n) - Q_star.dot(Q_star.T)
    
    A_ln, B_ln = M, sqrt_Pi.dot(M_star).dot(sqrt_Pi)
    A_odp, B_odp = np.linalg.inv(sqrt_Pi).dot(M).dot(np.linalg.inv(sqrt_Pi)), M_star

    if family_null == 'log_normal_response':
        p_value = 1 - saddlepoint_cdf_R(A_ln, B_ln, [R]).iloc[0]
        power_at_R = 1 - saddlepoint_cdf_R(A_odp, B_odp, [R]).iloc[0]
    elif family_null == 'od_poisson_response':
        p_value = saddlepoint_cdf_R(A_odp, B_odp, [R]).iloc[0]
        power_at_R = saddlepoint_cdf_R(A_ln, B_ln, [R]).iloc[0]
    else: raise ValueError('"family_null" must be "log_normal_response" ' + 
                           'or "od_poisson_response')

    return {'R_stat' : R, 'p_value' : p_value, 'power_at_R': power_at_R}