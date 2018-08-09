"""Pre-formatted data for age-period-cohort analysis."""

import numpy as np
import pandas as pd
from pkg_resources import resource_filename


def loss_TA():
    """
    Loss data as pandas.DataFrame for use with Model().data_from_df().

    The data set is taken from Taylor and Ashe (1983). Data are a
    run-off triangle containing paid amounts (units not reported).
    These data are also analyzed in various papers, e.g. England and
    Verrall (1999).

    Parameters
    ----------
    None

    Returns
    -------
    data_ta : pandas.DataFrame

    Notes
    -----
    The data description is largely taken from the R package apc.

    References
    ----------
    - England, P., & Verrall, R. (1999). Analytic and bootstrap estimates of
    prediction errors in claims reserving. Insurance: Mathematics and
    Economics, 25(3), 281–293.
    - Taylor, G. C., & Ashe, F. R. (1983). Second moments of estimates of
    outstanding claims. Journal of Econometrics, 23(1), 37–61.

    """
    data_ta = pd.read_csv(resource_filename('apc', 'data/loss_TA.csv'),
                          index_col=0)
    data_ta.index.name = 'Accident Year'
    data_ta.columns.name = 'Development Year'

    return data_ta


def loss_VNJ(include_counts=False):
    """
    Organize motor data into pandas.DataFrame.

    The data set is taken from tables 1,2 of Verrall et al. (2010).
    Data from Codan, Danish subsidiary of Royal & Sun Alliance. It is a
    portfolio of third party liability from motor policies. The time units are
    in years. There are two run-off triangles: "response" is paid amounts
    (units not reported) "counts" is number of reported claims.

    These data are also analyzed in e.g. Martinez Miranda et al. (2011) and
    Kuang et al. (2015).

    The `data_format` is 'CL'.

    Parameters
    ----------
    include_counts : bool, optional
        Whether the count data should be included in the output. (Default
        False.)

    Returns
    -------
    If `include_counts` is False
        VNJ_response : pandas.DataFrame
    If `include_counts` is True
        VNJ_data : dictionary
            Dictionary of pandas.DataFrames with keys 'response' and 'counts'.

    Notes
    -----
    The data description is largely taken from the R package apc.

    References
    ----------
    - Verrall, R., Nielsen, J. P., & Jessen, A. H. (2010). Prediction of RBNS
    and IBNR claims using claim amounts and claim counts. ASTIN Bulletin,
    40(2), 871–887.
    - Martínez-Miranda, M. D., Nielsen, B., Nielsen, J. P., & Verrall, R.
    (2011). Cash flow simulation for a model of outstanding liabilities based
    on claim amounts and claim numbers. ASTIN Bulletin, 41(1), 107–129.
    - Kuang, D., Nielsen, B., & Nielsen, J. P. (2015). The geometric
    chain-ladder. Scandinavian Actuarial Journal, 2015(3), 278–300.

    """
    response_vnj = pd.read_excel(
        resource_filename('apc', 'data/loss_VNJ.xlsx'), 'response', index_col=0
        )
    response_vnj.index.name = 'Accident Year'
    response_vnj.columns.name = 'Development Year'

    counts_vnj = pd.read_excel(
        resource_filename('apc', 'data/loss_VNJ.xlsx'), 'counts', index_col=0
        )
    counts_vnj.index.name = 'Accident Year'
    counts_vnj.columns.name = 'Development Year'

    if include_counts:
        data_vnj = {'response': response_vnj, 'counts': data_vnj_counts}
    else:
        data_vnj = response_vnj

    return data_vnj


def Belgian_lung_cancer():
    """
    Organize Belgian lung cancer data into pandas.DataFrame.

    The data set is taken from table VIII of Clayton and Schifflers (1987),
    which contains age-specific incidence rates (per 100,000 person-years
    observation) of lung cancer in Belgian females during the period 1955-1978.
    Numerators are also available. The original source was the WHO mortality
    database.

    The 'data_format' is "AP". The original data set is unbalanced since the
    first four period groups cover 5 years, while the last covers 4 years. The
    unbalanced period group is not included in this data set.

    Parameters
    ----------
    None

    Returns
    -------
    lung_cases : dict
        Dictionary with keys 'response', 'rate' and 'data_format'.

    Notes
    -----
    The data description is largely taken from the R package apc.

    References
    ----------
    - Clayton, D. and Schifflers, E. (1987). Models for temporal variation in
    cancer rates. I: age-period and age-cohort models. Statistics in Medicine
    6, 449-467.

    """
    lung_cases = pd.read_excel(
        resource_filename('apc', 'data/Belgian_lung_cancer.xlsx'), 'response',
        index_col=0
        )
    lung_rates = pd.read_excel(
        resource_filename('apc', 'data/Belgian_lung_cancer.xlsx'), 'rates',
        index_col=0
        )
    lung_data = {'response': lung_cases, 'rate': lung_rates,
                 'data_format': 'AP'}
    return lung_data


def loss_BZ():
    """
    Loss data as pandas.DataFrame for use with Model().data_from_df().

    The data set is taken from Table 3.5 of Barnett & Zehnwirth (2000).
    The data are also analyzed in e.g. Kuang, Nielsen, Nielsen (2011).

    Parameters
    ----------
    None

    Returns
    -------
    data_bz : pandas.DataFrame

    Notes
    -----
    The data description is largely taken from the R package apc.

    References
    ----------
    Barnett G, Zehnwirth B (2000) Best estimates for reserves.
    Proc. Casualty Actuary. Soc. 87, 245–321.

    Kuang D, Nielsen B, Nielsen JP (2011) Forecasting in an extended
    chain-ladder-type model Journal of Risk and Insurance 78, 345-359.

    """
    data_bz = pd.read_csv(
        resource_filename('apc', 'data/loss_BZ.csv'), index_col=0
        )
    data_bz.index.name = 'Accident Year'
    data_bz.columns.name = 'Development Year'

    return data_bz


def asbestos(sample='2007', balanced=True):
    """
    Organize asbestos mortality data as pandas.DataFrame.

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
               the younger and older end. If True, these are dropped, otherwise
               they are included. Note that the package currently cannot handle
               unbalanced groups for modeling. (Default is True.)

    Returns
    -------
    asbestos : pandas.DataFrame

    Notes
    -----
    The data description is largely taken from the R package apc.
    Martinez Miranda et al. (2015) use the '2007' sample. In the analysis, they
    use the age-groups from 25-89 (note that this corresponds to dropping more
    then just the unbalanced age-groups). Nielsen (2015) considers the same
    sample. Martinez-Miranda et al. (2016) use the update sample 'men_2013'.

    References
    ----------
    - Martínez Miranda, M. D., Nielsen, B., & Nielsen, J. P. (2015). Inference
    and forecasting in the age-period-cohort model with unknown exposure with
    an application to mesothelioma mortality. Journal of the Royal Statistical
    Society: Series A (Statistics in Society), 178(1), 29–55.
    - Martínez-Miranda, M. D., Nielsen, B., & Nielsen, J. P. (2016). Simple
    benchmark for mesothelioma projection for Great Britain. Occupational and
    Environmental Medicine, 73(8), 561–563.
    - Nielsen, B. (2015). apc: An R Package for Age-Period-Cohort Analysis.
    The R Journal, 7(2), 52–64.
    Open Access: https://journal.r-project.org/archive/2015-2/nielsen.pdf

    """
    asbestos = pd.read_excel(
        resource_filename('apc', 'data/asbestos_mortality.xlsx'), sample,
        index_col='Period'
        )
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
    Organize loss data as pandas.DataFrame.

    The data set is taken from Kuang and Nielsen (2018). The data are from the
    insurer XL Group. Data are US casualty data organized in a run-off triangle
    containing gross paid and reported loss and allocated loss adjustment
    expense in 1000 USD.

    Parameters
    ----------
    None

    Returns
    -------
    data_kn : pandas.DataFrame

    References
    ----------
    - Kuang, D., & Nielsen, B. (2018). Generalized Log-Normal Chain-Ladder.
    ArXiv E-Prints, 1806.05939. Download: http://arxiv.org/abs/1806.05939

    """
    data_kn = pd.read_csv(
        resource_filename('apc', 'data/loss_KN.csv'), index_col=0
        )
    data_kn.index.name = 'Cohort'
    data_kn.columns.name = 'Age'

    return data_kn
