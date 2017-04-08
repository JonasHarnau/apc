import numpy as np
import pandas as pd
from .format_data import format_data

def data_loss_TA():
    
    """
    
    Organises loss data into 'format_data' output. 
    
    The data set is taken from Table 1 of Verrall (1991), who attributes the 
    data to Taylor and Ashe (1983). It includes a run-off triangle: 'response'
    is paid amounts (units not reported). These data are also analysed in
    various papers, e.g. England and Verrall (1999).

    The 'data_format' is 'CL'.
    
    
    Notes
    -----
    
    The data description is taken from the R package apc.
       
       
    References
    ----------
    
    England, P., Verrall, R.J. (1999) Analytic and bootstrap estimates of
    prediction errors in claims reserving Insurance: Mathematics and Economics
    25, 281-293

    Taylor, G.C., Ashe, F.R. (1983) Second moments of estimates of outstanding
    claims Journal of Econometrics 23, 37-61

    Verrall, R.J. (1991) On the estimation of reserves from loglinear models
    
    """
    
    TA_response = pd.DataFrame(np.array([[  357848.,   766940.,   610542.,   482940.,   527326.,   574398.,
          146342.,   139950.,   227229.,    67948.],
       [  352118.,   884021.,   933894.,  1183289.,   445745.,   320996.,
          527804.,   266172.,   425046.,       np.nan],
       [  290507.,  1001799.,   926219.,  1016654.,   750816.,   146923.,
          495992.,   280405.,       np.nan,       np.nan],
       [  310608.,  1108250.,   776189.,  1562400.,   272482.,   352053.,
          206286.,       np.nan,       np.nan,       np.nan],
       [  443160.,   693190.,   991983.,   769488.,   504851.,   470639.,
              np.nan,       np.nan,       np.nan,       np.nan],
       [  396132.,   937085.,   847498.,   805037.,   705960.,       np.nan,
              np.nan,       np.nan,       np.nan,       np.nan],
       [  440832.,   847631.,  1131398.,  1063269.,       np.nan,       np.nan,
              np.nan,       np.nan,       np.nan,       np.nan],
       [  359480.,  1061648.,  1443370.,       np.nan,       np.nan,       np.nan,
              np.nan,       np.nan,       np.nan,       np.nan],
       [  376686.,   986608.,       np.nan,       np.nan,       np.nan,       np.nan,
              np.nan,       np.nan,       np.nan,       np.nan],
       [  344014.,       np.nan,       np.nan,       np.nan,       np.nan,       np.nan,
              np.nan,       np.nan,       np.nan,       np.nan]]),
                              index = range(1,11), columns = range(1,11))
    
    return format_data(response = TA_response, data_format = 'CL', label = "loss TA")