import unittest
import pandas as pd
import numpy as np
import apc

class TestFitTable(unittest.TestCase):

    def test_TA_odp(self):
        model = apc.Model()
        model.data_from_df(apc.loss_TA(), data_format='CL')
        model.fit(family='od_poisson_response', predictor='AC')
        model.fit_table()
        dev_table_Ad = model.fit_table(reference_predictor='Ad', 
                                        attach_to_self=False)
        
        self.assertTrue(
            np.allclose(
                model.deviance_table.sum().values, 
                np.array(
                    [6.14205640e+07, 4.26000000e+02, 0.00000000e+00, 4.42934379e+07,
                     1.02000000e+02, 6.42698087e+01, 8.77350747e-01]
                )
            )
        )
        self.assertTrue(
            np.allclose(
                dev_table_Ad.sum().values, 
                np.array(
                    [3.34371797e+07, 2.48000000e+02, 0.00000000e+00, 2.20883978e+07,
                     2.80000000e+01, 5.10659264e+01, 5.28199268e-02]
                )
            )
        )
        
    def test_Belgian_ln_rates(self):
        model = apc.Model()
        model.data_from_df(**apc.Belgian_lung_cancer())
        model.fit(family='log_normal_rates', predictor='APC')
        model.fit_table()
        
        self.assertTrue(
            np.allclose(
                model.deviance_table.sum().values,
                np.array(
                    [694.2842826, 508.000000 , 1367.09717145,  238.000000, 
                     349.62681465, 2.73393206, 998.2842826]
                )
            )
        )
    
    def test_Belgian_bin_dose_response(self):
        data = apc.Belgian_lung_cancer()
        dose = (data['response']/data['rate'] * 10**5).astype(int)
        model = apc.Model()
        model.data_from_df(data['response'], dose=dose, data_format='AP')
        model.fit_table('binomial_dose_response', 'APC')
        
        self.assertTrue(np.allclose(
            model.deviance_table.astype(float).values,
            np.array([
                [2.02272942e+01, 1.80000000e+01, 3.20169615e-01, np.nan, np.nan, np.nan],
                [2.55616207e+01, 3.00000000e+01, 6.97305582e-01, 5.33432652e+00,
                 1.20000000e+01, 9.45870225e-01],
                [2.14563493e+01, 2.00000000e+01, 3.70723402e-01, 1.22905512e+00,
                 2.00000000e+00, 5.40896377e-01],
                [9.91929917e+01, 2.70000000e+01, 3.49109630e-10, 7.89656975e+01,
                 9.00000000e+00, 2.59348099e-13],
                [2.65878986e+01, 3.20000000e+01, 7.37036572e-01, 6.36060439e+00,
                 1.40000000e+01, 9.56568004e-01],
                [2.53472759e+02, 3.90000000e+01, 0.00000000e+00, 2.33245465e+02,
                 2.10000000e+01, 0.00000000e+00],
                [1.00677524e+02, 2.90000000e+01, 7.61992691e-10, 8.04502302e+01,
                 1.10000000e+01, 1.20758958e-12],
                [8.55939082e+01, 3.30000000e+01, 1.48750103e-06, 6.53666140e+01,
                 1.50000000e+01, 2.94677404e-08],
                [6.39083556e+03, 4.00000000e+01, 0.00000000e+00, 6.37060827e+03,
                 2.20000000e+01, 0.00000000e+00],
                [1.21719783e+03, 3.00000000e+01, 0.00000000e+00, 1.19697053e+03,
                 1.20000000e+01, 0.00000000e+00],
                [2.54429395e+02, 4.10000000e+01, 0.00000000e+00, 2.34202101e+02,
                 2.30000000e+01, 0.00000000e+00],
                [3.08059993e+02, 4.20000000e+01, 0.00000000e+00, 2.87832698e+02,
                 2.40000000e+01, 0.00000000e+00],
                [6.39139748e+03, 4.20000000e+01, 0.00000000e+00, 6.37117019e+03,
                 2.40000000e+01, 0.00000000e+00],
                [1.61214822e+03, 4.20000000e+01, 0.00000000e+00, 1.59192092e+03,
                 2.40000000e+01, 0.00000000e+00],
                [6.50047766e+03, 4.30000000e+01, 0.00000000e+00, 6.48025037e+03,
                 2.50000000e+01, 0.00000000e+00]
            ]), 
            equal_nan=True)
                       )

    def test_Belgian_poisson_dose_response(self):
        model = apc.Model()
        model.data_from_df(**apc.Belgian_lung_cancer())
        model.fit_table('poisson_dose_response', 'APC')
        
        self.assertTrue(
            np.allclose(
                model.deviance_table.astype(float).sum().values,
                np.array(
                    [2.33052840e+04, 5.08000000e+02, 2.12588574e+00, 2.30019096e+04,
                     2.38000000e+02, 2.44351741e+00]
                )
            )
        )
        
if __name__ == '__main__':
    unittest.main()