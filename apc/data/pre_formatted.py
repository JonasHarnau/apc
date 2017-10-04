import numpy as np
import pandas as pd

def loss_TA():
    """
    Loss data as pandas.DataFrame for use with Model().data_from_df(). 
    
    The data set is taken from Taylor and Ashe (1983). Data are a 
    run-off triangle containing paid amounts (units not reported). 
    These data are also analysed in various papers, e.g. England and
    Verrall (1999).


    Notes
    -----
    
    The data description is largely taken from the R package apc.
       
       
    References
    ----------
    
    England, P., Verrall, R.J. (1999) Analytic and bootstrap estimates of
    prediction errors in claims reserving Insurance: Mathematics and Economics
    25, 281-293

    Taylor, G.C., Ashe, F.R. (1983) Second moments of estimates of outstanding
    claims Journal of Econometrics 23, 37-61
    
    """
    
    TA = pd.DataFrame(np.array([
        [  357848., 766940., 610542., 482940., 527326., 574398., 146342., 
         139950., 227229.,  67948.],
        [  352118., 884021., 933894.,  1183289., 445745., 320996., 527804., 
         266172., 425046.,   np.nan],
        [  290507.,  1001799., 926219.,  1016654., 750816., 146923., 495992., 
         280405.,   np.nan,   np.nan],
        [  310608.,  1108250., 776189.,  1562400., 272482., 352053., 206286., 
         np.nan,   np.nan,   np.nan],
        [  443160., 693190., 991983., 769488., 504851., 470639., np.nan, np.nan, 
         np.nan, np.nan],
        [  396132., 937085., 847498., 805037., 705960., np.nan, np.nan, np.nan, 
         np.nan, np.nan],
        [  440832., 847631.,  1131398.,  1063269., np.nan, np.nan, np.nan, np.nan,
         np.nan, np.nan],
        [  359480.,  1061648.,  1443370., np.nan, np.nan, np.nan, np.nan, np.nan, 
         np.nan, np.nan],
        [  376686., 986608., np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
         np.nan],
        [  344014., np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
         np.nan]
    ]),
                      index = range(1,11), columns = range(1,11))
    
    TA.index.name = 'Accident Year'
    TA.columns.name = 'Development Year'
    
    return TA


def loss_VNJ(include_counts=False):
    
    """
    
    Organises motor data into pandas.DataFrame. 
    
    The data description is taken from the R package apc.
    
    The data set is taken from tables 1,2 of Verrall, Nielsen and Jessen (2010).
    Data from Codan, Danish subsiduary of Royal & Sun Alliance. It is a
    portfolio of third party liability from motor policies. The time units are
    in years. There are two run-off triangles: "response" is paid amounts
    (units not reported) "counts" is number of reported claims.

    These data are also analysed in e.g. Martinez Miranda, Nielsen, Nielsen and
    Verrall (2011) and Kuang, Nielsen, Nielsen (2015).

    The 'data_format' is 'CL'.
    
    Parameters
    ----------
    
    include_counts : bool, optional
                     Whether the count data should be included in the output. 
                     Default is False
                     
    Returns
    -------
    
    pandas.DataFrame with response if 'include_counts' is false. Otherwise
    dictionary of data frames, one for response and one for counts.
    
    
    References
    ----------
    
    Verrall R, Nielsen JP, Jessen AH (2010) Prediction of RBNS and IBNR claims
    using claim amounts and claim counts ASTIN Bulletin 40, 871-887

    Martinez Miranda, M.D., Nielsen, B., Nielsen, J.P. and Verrall, R. (2011)
    Cash flow simulation for a model of outstanding liabilities based on claim
    amounts and claim numbers. ASTIN Bulletin 41, 107-129

    Kuang D, Nielsen B, Nielsen JP (2015) The geometric chain-ladder
    Scandinavian Acturial Journal 2015, 278-300.
    
    """
    
    VNJ_response = pd.DataFrame(np.array([
        [ 451288.,  339519.,  333371.,  144988.,   93243.,   45511., 25217.,   20406.,   31482.,    1729.],
        [ 448627.,  512882.,  168467.,  130674.,   56044.,   33397., 56071.,   26522.,   14346.,      np.nan],
        [ 693574.,  497737.,  202272.,  120753.,  125046.,   37154., 27608.,   17864.,      np.nan,      np.nan],
        [ 652043.,  546406.,  244474.,  200896.,  106802.,  106753., 63688.,      np.nan,      np.nan,      np.nan],
        [ 566082.,  503970.,  217838.,  145181.,  165519.,   91313., np.nan,      np.nan,      np.nan,      np.nan],
        [ 606606.,  562543.,  227374.,  153551.,  132743.,      np.nan, np.nan,      np.nan,      np.nan,      np.nan],
        [ 536976.,  472525.,  154205.,  150564.,      np.nan,      np.nan, np.nan,      np.nan,      np.nan,      np.nan],
        [ 554833.,  590880.,  300964.,      np.nan,      np.nan,      np.nan, np.nan,      np.nan,      np.nan,      np.nan],
        [ 537238.,  701111.,      np.nan,      np.nan,      np.nan,      np.nan, np.nan,      np.nan,      np.nan,      np.nan],
        [ 684944.,      np.nan,      np.nan,      np.nan,      np.nan,      np.nan, np.nan,      np.nan,      np.nan,      np.nan]
    ]),
                               index = range(1,11), columns = range(1,11))
    
    VNJ_response.index.name = 'Accident Year'
    VNJ_response.columns.name = 'Development Year'
    
    
    VNJ_counts = pd.DataFrame(np.array([
        [ 6238,  831,   49,    7,    1,    1,    2,    1,    2,     3],
        [ 7773, 1381,   23,    4,    1,    3,    1,    1,    3,    np.nan],
        [10306, 1093,   17,    5,    2,    0,    2,    2,   np.nan,    np.nan],
        [ 9639,  995,   17,    6,    1,    5,    4,   np.nan,   np.nan,    np.nan],
        [ 9511, 1386,   39,    4,    6,    5,   np.nan,   np.nan,   np.nan,    np.nan],
        [10023, 1342,   31,   16,    9,   np.nan,   np.nan,   np.nan,   np.nan,    np.nan],
        [ 9834, 1424,   59,   24,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,    np.nan],
        [10899, 1503,   84,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,    np.nan],
        [11954, 1704,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,    np.nan],
        [10989,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,    np.nan]
    ]),
                             index = range(1,11), columns = range(1,11))
    
    VNJ_counts.index.name = 'Accident Year'
    VNJ_counts.columns.name = 'Development Year'
    
    
    if include_counts:
        return {'response': VNJ_response, 'counts': VNJ_counts}
    else:
        return VNJ_response

