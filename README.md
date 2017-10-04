# apc

This package is for age-period-cohort analysis. 
It is very much inspired by the R package of the same name. 
The package covers binomial, log-normal, normal, over-dispersed Poisson and Poisson models. 
The common factor is a linear age-period-cohort predictor. 
The package uses the identification method by Kuang et al. (2008) implemented as described
by Nielsen (2015) who also discusses the use of the R package of the same name.

## Latest changes

Version 0.2.0 is a complete rething of the package. It introduces the ``Model`` Class as
the primary object of interest. Currently, this comes with limited functionality which will
be expanded quickly to the functionality of the last version 0.1.0.  

## Usage

1. Specify a model: ``model = Model()``
2. Attach the data: ``model.data_from_df(pandas.DataFrame)``
3. Plot data sunms: ``model.plot_data_sums()``
4. Fit the model: ``model.fit(family, predictor)``
5. Fit a deviance table to check for valid reductions: ``model.fit_table()``

## Example

As an example, following the scheme described above we can do the following. 

First we load some data and set-up the Model object.
```
import pandas as pd
data = pd.read_excel('./apc/data/Belgian_lung_cancer.xlsx', 
                     sheetname = ['response', 'rates'], index_col = 0)
from apc.Model import Model
model = Model()
```

We have a look at the data. This is in age-period space.
```
print(data['response'])

       1955-1959  1960-1964  1965-1969  1970-1974
                                                 
25-29          3          2          7          3
30-34         11         16         11         10
35-39         11         22         24         25
40-44         36         44         42         53
45-49         77         74         68         99
50-54        106        131         99        142
55-59        157        184        189        180
60-64        193        232        262        249
65-69        219        267        323        325
70-74        223        250        308        412
75-79        198        214        253        338
```


We can now attach the data to the model. 
```
model.data_from_df(data['response'], rate=data['rates'], data_format='AP')
```
This computes some checks and reformats it into a long format with a multi-index
that can more easily be used later on. The missing third index, cohort in this case,
is computed automatically. We can have a look at the output.
```
print(model.data_vector.head())

                           response       dose  rate
Period    Age   Cohort                              
1955-1959 25-29 1926-1934         3  15.789474  0.19
          30-34 1921-1929        11  16.666667  0.66
          35-39 1916-1924        11  14.102564  0.78
          40-44 1911-1919        36  13.483146  2.67
          45-49 1906-1914        77  15.909091  4.84
```
Note that the model is capable of generating cohort labels that reflect the correct range.
For instance, an individual who is 25-29 in 1955-1959 was born between 1926 and 1934.

We can move on to plot the data sums
```
model.plot_data_sums()

model.plotted_data_sums
```
![data_sum_plot](https://user-images.githubusercontent.com/25103918/31182586-458c3aa6-a8f2-11e7-8953-8b8f036a99a3.png)
This function includes functionality to transform index ranges into integer indices to 
allow for prettier axis labels. We can choose to transform to start, mean, or end of the 
range, or to keep the range labels.

Next, we can fit a model to the data. For example, a log-normal model for the rates
with an age-period-cohort predictor is fit like this

```
model.fit('log_normal_rates', 'APC')
```

We can then, for example, look at the estimated coefficients.
```
print(model.para_table)

                      coef   std err          z         P>|z|
level             1.930087  0.171428  11.258910  2.093311e-29
slope_age         0.536182  0.195753   2.739070  6.161319e-03
slope_coh         0.187146  0.195788   0.955857  3.391443e-01
dd_age_35-39     -0.574042  0.291498  -1.969284  4.892050e-02
dd_age_40-44      0.202367  0.286888   0.705389  4.805682e-01
dd_age_45-49     -0.171580  0.285231  -0.601548  5.474753e-01
dd_age_50-54     -0.200673  0.284958  -0.704219  4.812966e-01
dd_age_55-59     -0.059358  0.284947  -0.208314  8.349839e-01
dd_age_60-64     -0.076937  0.284958  -0.269994  7.871649e-01
dd_age_65-69      0.027525  0.285231   0.096499  9.231242e-01
dd_age_70-74     -0.067239  0.286888  -0.234375  8.146940e-01
dd_age_75-79     -0.081487  0.291498  -0.279546  7.798256e-01
dd_per_1965-1969 -0.088476  0.171182  -0.516855  6.052575e-01
dd_per_1970-1974 -0.014666  0.171182  -0.085677  9.317229e-01
dd_coh_1886-1894  0.101196  0.430449   0.235094  8.141360e-01
dd_coh_1891-1899  0.044626  0.335289   0.133098  8.941163e-01
dd_coh_1896-1904 -0.030893  0.291071  -0.106137  9.154736e-01
dd_coh_1901-1909 -0.064800  0.285233  -0.227184  8.202810e-01
dd_coh_1906-1914  0.087591  0.285009   0.307328  7.585934e-01
dd_coh_1911-1919 -0.020926  0.284956  -0.073438  9.414579e-01
dd_coh_1916-1924 -0.078598  0.284956  -0.275826  7.826816e-01
dd_coh_1921-1929  0.126880  0.285009   0.445181  6.561890e-01
dd_coh_1926-1934  0.085676  0.285233   0.300371  7.638939e-01
dd_coh_1931-1939 -0.400922  0.291071  -1.377401  1.683882e-01
dd_coh_1936-1944  0.732671  0.335289   2.185191  2.887488e-02
dd_coh_1941-1949 -1.053534  0.430449  -2.447520  1.438430e-02
```

We can check if there are valid reductions from the APC predictor by evaluating 
a deviance table.
```
model.fit_table()
```
We inspect the results 
```
print(model.deviance_table)

         -2logL  df_resid   LR_vs_APC  df_vs_APC      P>chi_sq         aic
APC  -44.854193        18         NaN        NaN           NaN    7.145807
AP   -28.997804        30   15.856389       12.0  1.979026e-01   -0.997804
AC   -43.453528        20    1.400664        2.0  4.964203e-01    4.546472
PC    12.631392        27   57.485584        9.0  4.079167e-09   46.631392
Ad   -28.042198        32   16.811994       14.0  2.663353e-01   -4.042198
Pd    47.605631        39   92.459824       21.0  6.054524e-11   57.605631
Cd    12.710530        29   57.564723       11.0  2.618204e-08   42.710530
A    -14.916745        33   29.937447       15.0  1.214906e-02    7.083255
P    165.610725        40  210.464918       22.0  0.000000e+00  173.610725
C     88.261366        30  133.115559       12.0  0.000000e+00  116.261366
t     47.774702        41   92.628894       23.0  2.562911e-10   53.774702
tA    50.423353        42   95.277546       24.0  1.896069e-10   54.423353
tP   165.622315        42  210.476508       24.0  0.000000e+00  169.622315
tC    98.099334        42  142.953526       24.0  0.000000e+00  102.099334
1    165.809402        43  210.663595       25.0  0.000000e+00  167.809402
```
We can see that for example an age-cohort model (``AC``) seems to be a valid reduction.
A likelihood ratio test yields a p-value of close to 0.5. 
We can check for further reductions from the age-cohort model.
```
dev_table_AC = model.fit_table(reference_predictor='AC', attach_to_self=False)

print(dev_table_AC)

        -2logL  df_resid    LR_vs_AC  df_vs_AC      P>chi_sq         aic
AC  -43.453528        20         NaN       NaN           NaN    4.546472
Ad  -28.042198        32   15.411330      12.0  2.197089e-01   -4.042198
Cd   12.710530        29   56.164058       9.0  7.302826e-09   42.710530
A   -14.916745        33   28.536783      13.0  7.610743e-03    7.083255
C    88.261366        30  131.714894      10.0  0.000000e+00  116.261366
t    47.774702        41   91.228230      21.0  9.899548e-11   53.774702
tA   50.423353        42   93.876881      22.0  7.438616e-11   54.423353
tC   98.099334        42  141.552862      22.0  0.000000e+00  102.099334
1   165.809402        43  209.262930      23.0  0.000000e+00  167.809402
```
We could now consider a further reduction to an age-drift model (``Ad``) with a p-value of 0.22.

    
## Included Data

The following data examples are included in the package at this time. 

### Belgian Lung Cancer 

This data-set is currently provided in an excel spreadsheet.
It includes counts of deaths from lung cancer in the Belgium. 
This dataset includes a measure for exposure. It can be analysed using a Poisson model 
with an “APC”, “AC”, “AP” or “Ad” predictor. 

_Source: Clayton and Schifflers (1987)_.

### Loss TA 

Data for an insurance run-off triangle. This data is pre-formatted.
May be modeled with an over-dispersed Poisson model,
for instance with ``AC`` predictor. 

_Source: Taylor and Ashe (1983)_

## Known Issues

* Index-ranges, such as _1955-1959_ in ``data_vector`` as output by ``Model().data_as_df()``
are strings. Thus, sorting may yield unintuitive results for breaks in length of the range
components. For example, sorting 1-3, 4-9, 10-11 yields the ordering 1-3, 10-11, 4-9. 
This results in mis-labeling of the coefficient names later on since those are taken from
sorted indices. A possible, if ugly, fix could be to pad the ranges with zeros as needed. 

## References

*   Clayton, D. and Schifflers, E. (1987) Models for temperoral variation in cancer rates. I: age-period and age-cohort models. Statistics in Medicine 6, 449-467.
*   Taylor, G.C., Ashe, F.R. (1983) Second moments of estimates of outstanding claims Journal of Econometrics 23, 37-61

