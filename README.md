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
3. Fit the model: ``model.fit(family, predictor)``

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

## References

-   Clayton, D. and Schifflers, E. (1987) Models for temperoral variation in cancer rates. I: age-period and age-cohort models. Statistics in Medicine 6, 449-467.
-   Taylor, G.C., Ashe, F.R. (1983) Second moments of estimates of outstanding claims Journal of Econometrics 23, 37-61

 of RBNS and IBNR claims using claim amounts and claim counts ASTIN Bulletin 40, 871-887

