import numpy as np
import pandas as pd
from pkg_resources import resource_filename

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

def Belgian_lung_cancer():
    """
    Organises Belgian lung cancer data into pandas.DataFrame. 
    
    The data set is taken from table VIII of Clayton and Schifflers (1987),
    which contains age-specific incidence rates (per 100,000 person-years
    observation) of lung cancer in Belgian females during the period 1955-1978.
    Numerators are also available. The original source was the WHO mortality
    database.
 
    The 'data_format' is "AP". The original data set is unbalanced since the
    first four period groups cover 5 years, while the last covers 4 years. The
    unbalanced period group is not included in this data set.
        
       
    Notes
    -----
    
    The data description is largely taken from the R package apc.
 
 
    References
    ----------
    
    Clayton, D. and Schifflers, E. (1987) Models for temperoral variation in
    cancer rates. I: age-period and age-cohort models. Statistics in Medicine 6,
    449-467.
    
    """
    
    index = ['25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', 
             '60-64', '65-69', '70-74', '75-79']
    
    columns = ['1955-1959', '1960-1964', '1965-1969', '1970-1974']
    
    
    lung_cases = pd.DataFrame(np.array(([  3,   2,   7,   3],
                                        [ 11,  16,  11,  10],
                                        [ 11,  22,  24,  25],
                                        [ 36,  44,  42,  53],
                                        [ 77,  74,  68,  99],
                                        [106, 131,  99, 142],
                                        [157, 184, 189, 180],
                                        [193, 232, 262, 249],
                                        [219, 267, 323, 325],
                                        [223, 250, 308, 412],
                                        [198, 214, 253, 338])).reshape((11, 4)),
                             index = index, columns = columns)
    
    lung_rates = pd.DataFrame(np.array(([  0.19,   0.13,   0.5 ,   0.19],
                                        [  0.66,   0.98,   0.72,   0.71],
                                        [  0.78,   1.32,   1.47,   1.64],
                                        [  2.67,   3.16,   2.53,   3.38],
                                        [  4.84,   5.6 ,   4.93,   6.05],
                                        [  6.6 ,   8.5 ,   7.65,  10.59],
                                        [ 10.36,  12.  ,  12.68,  14.34],
                                        [ 14.76,  16.37,  18.  ,  17.6 ],
                                        [ 20.53,  22.6 ,  24.9 ,  24.33],
                                        [ 26.24,  27.7 ,  30.47,  36.94],
                                        [ 33.47,  33.61,  36.77,  43.69])).reshape((11, 4)),
                             index = index, columns = columns)
    
    return {'response': lung_cases, 'rate': lung_rates, 'data_format': 'AP'}

def loss_BZ():
    """
    Loss data as pandas.DataFrame for use with Model().data_from_df(). 
    
    The data set is taken from Table 3.5 of Barnett & Zehnwirth (2000).
    The data are also analysed in e.g. Kuang, Nielsen, Nielsen (2011).



    Notes
    -----
    
    The data description is largely taken from the R package apc.
       
       
    References
    ----------
    
    Barnett G, Zehnwirth B (2000) Best estimates for reserves. 
    Proc. Casualty Actuar. Soc. 87, 245–321.

    Kuang D, Nielsen B, Nielsen JP (2011) Forecasting in an extended
    chain-ladder-type model Journal of Risk and Insurance 78, 345-359
    
    """
    
    BZ = pd.DataFrame(np.array([
        [153638, 188412, 134534,  87456,  60348, 42404, 31238, 21252, 16622, 14440, 12200],
        [178536, 226412, 158894, 104686,  71448, 47990, 35576, 24818, 22662, 18000, np.nan],
        [210172, 259168, 188388, 123074,  83380, 56086, 38496, 33768, 27400, np.nan, np.nan],
        [211448, 253482, 183370, 131040,  78994, 60232, 45568, 38000, np.nan, np.nan, np.nan],
        [219810, 266304, 194650, 120098,  87582, 62750, 51000, np.nan, np.nan, np.nan, np.nan],
        [205654, 252746, 177506, 129522,  96786, 82400, np.nan, np.nan, np.nan, np.nan, np.nan],
        [197716, 255408, 194648, 142328, 105600, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        [239784, 329242, 264802, 190400,  np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        [326304, 471744, 375400, np.nan,  np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        [420778, 590400, np.nan, np.nan,  np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        [496200,  np.nan, np.nan, np.nan,  np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    ]),
                      index = range(1977,1988), columns = range(1,12))
    
    BZ.index.name = 'Accident Year'
    BZ.columns.name = 'Development Year'
    
    return BZ

def asbestos(sample='2007', balanced=True):
    """
    
    Asbestos mortality data formatted as pandas.DataFrame.
    
    Counts of mesothelioma deaths in the UK by age and period. Mesothelioma is 
    most often caused by exposure to asbestos. The data are organized in a 
    period-age table.
    
    Parameters
    ----------
    
    sample : {'2007', 'men_2013', 'women_2013'}, optional
             Determines what sample is to be used. Three samples are available.
             '2007' contains data from 1967-2007. 'men_2013' contains data for
             men from 1967-2013 and 'women_2013' contains data for the same
             period for women. (Default is '2007'.)
    
    balanced : bool, optional
               The data originally contains some unbalanced age groups both at
               the younger and older end. If True, these are droppe, otherwise
               they are included. Note that the package currently cannot handle
               unbalanced groups for modeling. (Default is True.)

    Notes
    -----
    
    The data description is largely taken from the R package apc.
    Martinez Miranda et al. (2015) use the '2007' sample. In their analysis they
    use the age-groups from 25-89 (note that this corresponds to dropping more
    then just the unbalanced age-groups). Nielsen (2015) considers the same 
    sample. Martinez-Miranda et al. (2016) use the update sample 'men_2013'. 
       
       
    References
    ----------
    
    Martínez Miranda, M. D., Nielsen, B., & Nielsen, J. P. (2015). Inference and
    forecasting in the age-period-cohort model with unknown exposure with an 
    application to mesothelioma mortality. Journal of the Royal Statistical 
    Society: Series A (Statistics in Society), 178(1), 29–55.

    Martínez-Miranda, M. D., Nielsen, B., & Nielsen, J. P. (2016). Simple 
    benchmark for mesothelioma projection for Great Britain. Occupational and 
    Environmental Medicine, 73(8), 561–563. 
    
    Nielsen, B. (2015). apc: An R Package for Age-Period-Cohort Analysis. The R
    Journal, 7(2), 52–64. 
    Open Access: https://journal.r-project.org/archive/2015-2/nielsen.pdf
    
    """
    asbestos = pd.read_excel(resource_filename('apc', 'data/asbestos_mortality.xlsx'), 
                             sample, index_col='Period')
    asbestos.columns.name = 'Age'
    
    if balanced:
        if sample == '2007':
            asbestos = asbestos.iloc[:, 4:-1]
        else: 
            asbestos = asbestos.iloc[:, 2:-1]
        asbestos.columns = asbestos.columns.astype(np.int64)
        
    return asbestos

def loss_KN():
    """
    Loss data as pandas.DataFrame for use with Model().data_from_df(). 
    
    The data set is taken from Kuang and Nielsen (2018). The data are from the insurer
    XL Group. Data are US casualty data organized in a run-off triangle containing
    gross paid and reported loss and allocated loss adjustment expense in 1000 USD. 

       
    References
    ----------
    
    Kuang, D., & Nielsen, B. (2018). Generalized Log-Normal Chain-Ladder. 
    ArXiv E-Prints, 1806.05939. Available from http://arxiv.org/abs/1806.05939
    
    """
    data = pd.read_csv(resource_filename('apc', 'data/xl_insurance.csv'), index_col=0)
    data.index.name = 'Cohort'
    data.columns.name = 'Age'
    
    return data      