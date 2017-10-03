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

### Example Code

As an example, following the scheme described above we can do the following. 

```
import pandas as pd
data = pd.read_excel('./apc/data/Belgian_lung_cancer.xlsx', 
                     sheetname = ['response', 'rates'], index_col = 0)

from Model import Model
model = Model()
model.data_from_df(data['response'], rate=data['rates'], data_format='AP')
print(model.data_vector.head())
                           response       dose  rate
Period    Age   Cohort                              
1955-1959 25-29 1926-1934         3  15.789474  0.19
          30-34 1921-1929        11  16.666667  0.66
          35-39 1916-1924        11  14.102564  0.78
          40-44 1911-1919        36  13.483146  2.67
          45-49 1906-1914        77  15.909091  4.84

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

