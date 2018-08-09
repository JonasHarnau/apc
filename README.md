# apc

This package is for age-period-cohort and extended chain-ladder analysis. It allows for model estimation and inference, visualization, misspecification testing, distribution forecasting and simulation. The package covers binomial, (generalized) log-normal, normal, over-dispersed Poisson and Poisson models. The common factor is a linear age-period-cohort predictor. The package uses the identification method by Kuang et al. (2008) implemented as described by Nielsen (2015) who also discusses the use of the R package ``apc`` which inspired this package.

## Latest changes

Version 1.0.1 fixes some typos and refactors production code.

Version 1.0.0 adds a number of new features. Among them are

* Vignettes to replicate papers (start with those!)
* Distribution forecasting
* Misspecification tests
* More plotting
* Simulating from an estimated model
* Sub-sampling

## Usage

1. import package: ``import apc``
2. Set up a model: ``model = apc.Model()``
3. Attach and format the data: ``model.data_from_df(pandas.DataFrame)``
4. Plot data
   * Plot data sums: ``model.plot_data_sums()``
   * Plot data heatmaps: ``model.plot_data_heatmaps()``
   * Plot data groups of one time-scale across another: ``model.plot_data_within()``
5. Fit and evaluate the model
   * Fit a model: ``model.fit(family, predictor)``
   * Plot residuals: ``model.plot_residuals()``
   * Generate ad-hoc identified parameterizations: ``model.identify()``
   * Plot parameter estimates: ``model.plot_parameters()``
   * Fit a deviance table to check for valid reductions: ``model.fit_table()``
6. Test model for misspecification
   * R test (generalized) log-normal against over-dispersed Poisson: ``apc.r_test(pandas.DataFrame, family_null, predictor)``
   * Split into sub-models: ``model.sub_model(age_from_to, per_from_to, coh_from_to)``
   * Bartlett test: ``apc.bartlett_test(sub_models)``
   * F test: ``apc.f_test(model, sub_models)``
7. Form distribution forecasts: ``model.forecast()``
8. Plot distribution forecasts: ``model.plot_forecast()``
9. Simulate from the model: ``model.simulate(repetitions)``

## Vignettes

The package includes vignettes that replicate the empirical applications of a number of papers.

* [Replicate Harnau and Nielsen (2017)](https://github.com/JonasHarnau/apc/blob/master/apc/vignettes/vignette_over_dispersed_apc.ipynb)
  * Non-Life Insurance Claim Reserving
  * Over-dispersed Poisson deviance analysis, parameter uncertainty, and distribution forecasting
* [Replicate Harnau (2018a)](https://github.com/JonasHarnau/apc/blob/master/apc/vignettes/vignette_misspecification.ipynb)
  * Non-Life Insurance Claim Reserving
  * Testing specification of log-normal or over-dispersed Poisson models with Bartlett and F tests
* [Replicate Harnau (2018b)](https://github.com/JonasHarnau/apc/blob/master/apc/vignettes/vignette_ln_vs_odp.ipynb)
  * Non-Life Insurance Claim Reserving
  * Direct testing between over-dispersed Poisson and (generalized) log-normal models
* [(Loosely) Replicate Martinez Miranda et al. (2015)](https://github.com/JonasHarnau/apc/blob/master/apc/vignettes/vignette_mesothelioma.ipynb)
  * Mesothelioma Mortality Forecasting
  * Data plotting, Poisson deviance analysis, parameter uncertainty, residual plots, and distribution forecasting including plots
* [Replicate Kuang and Nielsen (2018)](https://github.com/JonasHarnau/apc/blob/master/apc/vignettes/vignette_generalized_log_normal.ipynb)
  * Non-Life Insurance Claim Reserving
  * Estimating, testing and forecasting in generalized log-normal models. Comparison to over-dispersed Poisson modeling.
* [Replicate Nielsen (2014)](https://github.com/JonasHarnau/apc/blob/master/apc/vignettes/vignette_deviance_analysis.ipynb)
  * Analysis of Belgian lung cancer data
  * Estimating, testing and plotting in Poisson dose-response models. Testing for non-standard restrictions. Brief discussion of identification.

## Included Data

The following data are included in the package.

### Asbestos

These data are for counts of mesothelioma deaths in the UK in age-period space. They may be modeled with a Poisson model with "APC" or "AC" predictor. The data can be loaded by calling ``apc.asbestos()``.

*Source: Martinez Miranda et al. (2015).*

### Belgian Lung Cancer

These data includes counts of deaths from lung cancer in Belgium in age-period space. This dataset includes a measure for exposure. It can be analyzed using a Poisson model with an “APC”, “AC”, “AP” or “Ad” predictor. The data can be loaded by calling ``apc.Belgian_lung_cancer()``.

*Source: Clayton and Schifflers (1987).*

### Run-off triangle by Barnett and Zehnwirth (2000)

Data for an insurance run-off triangle in cohort-age (accident-development year) space. This data is pre-formatted. These data are well known to require a period/calendar effect for modeling. They may be modeled with an over-dispersed Poisson "APC" predictor. The data can be loaded by calling ``apc.loss_BZ()``.

*Source: Barnett and Zehnwirth (2000).*

### Run-off triangle by Taylor and Ashe (1983)

Data for an insurance run-off triangle in cohort-age (accident-development year) space. This data is pre-formatted.
May be modeled with an over-dispersed Poisson model, for instance with "AC" predictor. The data can be loaded by calling ``apc.loss_TA()``.

*Source: Taylor and Ashe (1983).*

### Run-off triangle by Verrall et al. (2010)

Data for insurance run-off triangle of paid amounts (units not reported) in cohort-age (accident-development year) space.
Data from Codan, Danish subsidiary of Royal & Sun Alliance.
It is a portfolio of third party liability from motor policies. The time units are in years.
Apart from the paid amounts, counts for the number of reported claims are available. The paid amounts may be modeled with an over-dispersed Poisson model with "APC" predictor. The data can be loaded by calling ``apc.loss_VNJ()``.

*Source: Verrall et al. (2010).*

### Run-off triangle by Kuang and Nielsen (2018)

These US casualty data are from the insurer XL Group. Entries are gross paid and reported loss and allocated loss adjustment expense in 1000 USD. Kuang and Nielsen (2018) consider a generalized log-normal model with "AC" predictor for these data. The data can be loaded by calling ``apc.loss_KN()``.

## Known Issues

* Index-ranges such as _1955-1959_ don't work with forecasting if the initial ``data_format`` was not CA or AC. The problem is that the forecasting design is generated by first casting the data into an AC array from which the future period index is generated.
* Index-ranges, such as _1955-1959_ in ``data_vector`` as output by ``Model().data_as_df()`` are strings. Thus, sorting may yield unintuitive results for breaks in length of the range components. For example, sorting 1-3, 4-9, 10-11 yields the ordering 1-3, 10-11, 4-9. This results in mislabeling of the coefficient names later on since those are taken from sorted indices. A possible, if ugly, fix could be to pad the ranges with zeros as needed.

## References

* Barnett, G., & Zehnwirth, B. (2000). Best estimates for reserves. *Proceedings of the Casualty Actuarial Society*, 87(167), 245–321.
* Clayton, D. and Schifflers, E. (1987). Models for temporal variation in cancer rates. I: age-period and age-cohort models. *Statistics in Medicine* 6, 449-467.
* Harnau, J., & Nielsen, B. (2017). Over-dispersed age-period-cohort models. *Journal of the American Statistical Association*. [Available online](https://doi.org/10.1080/01621459.2017.1366908)
* Harnau, J. (2018a). Misspecification Tests for Log-Normal and Over-Dispersed Poisson Chain-Ladder Models. *Risks*, 6(2), 25. [Open Access](https://doi.org/10.3390/RISKS6020025)
* Harnau, J. (2018b). Log-Normal or Over-Dispersed Poisson? *Risks*, 6(3), 70. [Open Access](https://doi.org/10.3390/RISKS6030070)
* Kuang, D., Nielsen, B., & Nielsen, J. P. (2008). Identification of the age-period-cohort model and the extended chain-ladder model. *Biometrika*, 95(4), 979–986. [Open Access](https://doi.org/10.1093/biomet/asn026)
* Kuang, D., & Nielsen, B. (2018). Generalized Log-Normal Chain-Ladder. *ArXiv E-Prints*, 1806.05939. [Download](http://arxiv.org/abs/1806.05939)
* Nielsen, B. (2014). Deviance analysis of age-period-cohort models. *Nuffield Discussion Paper*, (W03). [Download](http://www.nuffield.ox.ac.uk/economics/papers/2014/apc_deviance.pdf)
* Nielsen, B. (2015). apc: An R package for age-period-cohort analysis. *R Journal* 7, 52-64. [Open Access](https://journal.r-project.org/archive/2015-2/nielsen.pdf).
* Martinez Miranda, M. D., Nielsen, B., & Nielsen, J. P. (2015). Inference and forecasting in the age-period-cohort model with unknown exposure with an application to mesothelioma mortality. *Journal of the Royal Statistical Society: Series A (Statistics in Society)*, 178(1), 29–55.
* Taylor, G.C., Ashe, F.R. (1983). Second moments of estimates of outstanding claims. *Journal of Econometrics* 23, 37-61
* Verrall R., Nielsen J.P., Jessen A.H. (2010). Prediction of RBNS and IBNR claims using claim amounts and claim counts. *ASTIN Bulletin* 40, 871-887