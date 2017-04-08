===
apc
===

This package is for age-period-cohort analysis. It is very much inspired by the R package of the same name. The package covers binomial, log-normal, normal, over-dispersed Poisson and Poisson models. The common factor is a linear age-period-cohort predictor. The package uses the identification method by Kuang et al. (2008) implemented as described by Nielsen (2015) who also discusses the use of the homonymous R package.

Usage
=====

1. The data needs to be organized in a format the remaining functions can handle. That is the purpose of the function ``format_data``. Several data sets organised in the right format come with the package. Those are listed below.

2. Once the output of ``format_data`` is obtained, we can plot the data. There are several options available. For an overview, ``plot_data_all`` generates the following four plots that may also be called individually.

    a) ``plot_data_heatmap`` yields a heatmap of the data. 
    b) ``plot_data_sparsity`` generates a sparsity plot. This can also be generated directly from ``plot_data_heatmap`` by changing an argument but is implemented for convenience.
    c) ``plot_data_sums`` show how data sums evolve over the individual time scales, that is over age, period and cohort. The numeric sums can be obtained through the function ``get_data_sums``.
    d) ``plot_data_within`` generates is the disaggregated version of ``plot_data_sums``: it plots how data within one time scale evolves over another time scale. For example, how  data accumulates for each period separately over age.

3. After plotting, we can estimate a model. The main input is, again, the outpout of ``format_data``. Further, we have to choose a ``model_family``. Then, we have two options.
    
    a) To get an overview, we can use the function ``fit_table``. This generates a deviance table that allows us to see what valid model reductions could look like, for instance if an age-period-cohort model may be reduced to an age-cohort model. The reduction considered are those discussed by Nielsen (2014). 
    b) Either after using ``fit_table``, or when we know what ``model_design`` to use (e.g. *APC* or *AC*), we can estimate a model with ``fit_model``. 

Plotting functionality for the ``fit_model`` results and forecasting are not yet implemented in the package. This is in the works.

Example Code
------------

As an example, following the scheme described above we can do the following. We will use the Belgian lung cancer data from Clayton and Schifflers (1987). These are included in the package in ``apc/data/data_Belgian_lung_cancer.xlsx`` and were downloaded from `here <http://users.ox.ac.uk/~nuff0078/apc/>`_.  Note that the data is also pre-formatted in the package. This can be called as::

    apc.data_Belgian_lung_cancer()

1. We read in the data using the package *pandas*. This outputs a dictionary cotaining "response" and "rates". We then use ``format_data``. Since the imported dataframes do not have names for the headers we have to let the function know the ``data_format``.::

    import pandas as pd
    import apc
    
    data = pd.read_excel('data_Belgian_lung_cancer.xlsx', sheetname = ['response', 'rates'], index_col = 0)
    
    formatted_data = apc.format_data(response = data['response'], rate = data['rates'], data_format = 'AP')
    
2. This is all we need. Now we can plot the data.::

    heatmap, sparsity, sums, r_within, d_within, m_within = apc.plot_data_all(formatted_data)
    
    # We can show the plots by calling one by one
    heatmap
    sparsity
    sums
    r_within
    d_within
    m_within
    
3. Let's use a Poisson model with exposure. 
    
    a) We can use ``fit_table`` to investigate if this is appropriate and to check for valid reductions from the "APC" model. This shows that we cannot reject the "APC" model against a saturated model (p-value 0.32) and also tells us that "AP", "AC" and "Ad" model are valid reductions. We can see if the "Ad" model, which is nested in both "AP" and "AC", can still not be rejected if we start from the "AP" model. This is confirmed (p-value 0.60). Finally, we can produce a table with the "Ad" model as reference and see if further reductions are appropriate. This is rejected.::
    
        apc.fit_table(formatted_data, 'poisson_dose_response')
        apc.fit_table(formatted_data, 'poisson_dose_response', 'AP')
        apc.fit_table(formatted_data, 'poisson_dose_response', 'Ad')

    b) It remains to estimate the model of choice; we choose the "Ad" model. We can then for instance show a coefficient table.::

        model = apc.fit_model(formatted_data, 'poisson_dose_response', 'Ad')
        model.coefs_canonical_table


Included Data
=============

The following data examples are included in the package at this time. The data are a subset of those available in the R package ``apc``. The description is taken from the R package.

* ``data_aids`` includes counts for AIDS cases. The data are the number of cases by the date of diagnosis and length of reporting delay, measured by quarter. A measure for exposure is not available. The data may be modeled by an over-dispersed Poisson model with "APC" design. *Source:* De Angelis and Gilks (1994). Also analysed by Davison and Hinkley (1998, Example 7.4).

* ``data_asbestos`` includes counts of deaths from mesothelioma in the UK. This dataset has no measure for exposure. It can be analysed using a Poisson model with an "APC" or an "AC" design. *Source:* Martinez Miranda et al. (2015). Also used in Nielsen (2015).

* ``data_Belgian_lung_cancer`` includes counts of deaths from lung cancer in the Belgium. This dataset includes a measure for exposure. It can be analysed using a Poisson model with an "APC", "AC", "AP" or "Ad" design. *Source:* Clayton and Schifflers (1987).

* ``data_Italian_bladder_cancer`` includes counts of deaths from bladder cancer in the Italy. This dataset includes a measure for exposure. It can be analysed using a Poisson model with an "APC" or an "AC" design. *Source:* Clayton and Schifflers (1987).

* ``data_loss_TA`` includes an insurance run-off triangle of paid amounts (units not reported). May be modeled with an over-dispersed Poisson model, for instance with *AC* design. *Source:* Verrall (1991) who attributes the data to Taylor and Ashe (1983). Data also analysed in various papers, e.g. England and Verrall (1999).

* ``data_loss_VNJ`` includes an insurance run-off triangle of paid amounts (units not reported). Data from Codan, Danish subsiduary of Royal & Sun Alliance. It is a portfolio of third party liability from motor policies. The time units are in years. Apart from the paid amounts, counts for the number of reported claims are available. *Source:* Verrall et al. (2010). Data also analysed in e.g. Martinez Miranda et al. (2011) and Kuang et al. (2015).


References
==========

* Clayton, D. and Schifflers, E. (1987) Models for temperoral variation in cancer rates. I: age-period and age-cohort models. Statistics in Medicine 6, 449-467.

* Davison, A.C. and Hinkley, D.V. (1998) Bootstrap methods and their application. Cambridge: Cambridge University Press.

* De Angelis, D. and Gilks, W.R. (1994) Estimating acquired immune deficiency syndrome incidence accounting for reporting delay. Journal of the Royal Statistical Sociey A 157, 31-40.

* England, P., Verrall, R.J. (1999) Analytic and bootstrap estimates of prediction errors in claims reserving Insurance: Mathematics and Economics 25, 281-293

* Kuang, D., Nielsen, B. and Nielsen, J.P. (2008) Identification of the age-period-cohort model and the extended chain ladder model. Biometrika 95, 979-986.

* Kuang D, Nielsen B, Nielsen JP (2015) The geometric chain-ladder Scandinavian Acturial Journal 2015, 278-300.

* Martinez Miranda, M.D., Nielsen, B., Nielsen, J.P. and Verrall, R. (2011) Cash flow simulation for a model of outstanding liabilities based on claim amounts and claim numbers. ASTIN Bulletin 41, 107-129

* Martinez Miranda, M.D., Nielsen, B. and Nielsen, J.P. (2015) Inference and forecasting in the age-period-cohort model with unknown exposure with an application to mesothelioma mortality. Journal of the Royal Statistical Society A 178, 29-55. 

* Nielsen, B. (2014) Deviance analysis of age-period-cohort models. *Download:* `Nuffield Discussion Paper <http://www.nuffield.ox.ac.uk/economics/papers/2014/apc_deviance.pdf>`_.

* Nielsen, B. (2015) apc: An R package for age-period-cohort analysis. R Journal 7, 52-64. *Download:* `Open Access <https://journal.r-project.org/archive/2015-2/nielsen.pdf>`_.

* Taylor, G.C., Ashe, F.R. (1983) Second moments of estimates of outstanding claims Journal of Econometrics 23, 37-61

* Verrall, R.J. (1991) On the estimation of reserves from loglinear models Insurance: Mathematics and Economics 10, 75-80

* Verrall R, Nielsen JP, Jessen AH (2010) Prediction of RBNS and IBNR claims using claim amounts and claim counts ASTIN Bulletin 40, 871-887