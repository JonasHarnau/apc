import unittest
import numpy as np
import apc

class TestFit(unittest.TestCase):

    def test_TA_odp(self):
        model = apc.Model()
        model.data_from_df(apc.loss_TA(), data_format='CL')
        model.fit(family='od_poisson_response', predictor='AC')
        
        self.assertEqual(round(model.deviance,3), 1903014.004)
        self.assertTrue(
            np.allclose(
                model.parameters.sum().values,
                np.array([10.99021854,  6.56495772,  0.37934664,  8.3525477])
            )
        )
        self.assertEqual(round(model.fitted_values.sum(),3), 34358090.000)

    def test_BZ_gln(self):
        model = apc.Model()
        model.data_from_df(apc.loss_BZ(), data_format='CL')
        model.fit(family='gen_log_normal_response', predictor='APC')
        
        self.assertEqual(round(model.deviance,3), -287.459)
        self.assertTrue(
            np.allclose(
                model.parameters.sum().values,
                np.array([11.83624119,   1.26201587, 312.66337107,  12.1751463])
            )
        )
        self.assertEqual(round(model.fitted_values.sum(),3), 10214114.721)
                
    def test_asbestos_poisson(self):
        model = apc.Model()
        model.data_from_df(apc.asbestos(), data_format='PA')
        model.fit(family='poisson_response', predictor='AC')
        
        self.assertEqual(round(model.deviance,3), 2599.565)
        self.assertTrue(
            np.allclose(
                model.parameters.sum().values,
                np.array(
                    [-2.20066707e+01,  1.89725621e+06, -7.39025813e+00,  1.04940965e+02]
                )
            )
        )
        self.assertEqual(model.fitted_values.sum().astype(int), 
                         model.data_vector.sum()[0])
        
    def test_Belgian_poisson_dose_response(self):
        model = apc.Model()
        model.data_from_df(**apc.Belgian_lung_cancer())
        model.fit(family='poisson_dose_response', predictor='APC')
        
        self.assertEqual(round(model.deviance,3), 20.225)
        self.assertTrue(np.allclose(
            model.parameters['P>|z|'].values,
            np.array([
                4.98090007e-194, 2.00792411e-011, 7.54393422e-002, 2.44873634e-001,
                3.78642275e-001, 4.49590657e-001, 1.71890762e-001, 7.15051524e-001,
                3.40338551e-001, 7.77531241e-001, 5.43258659e-001, 3.10146699e-001,
                3.27420794e-001, 3.02443187e-001, 4.90579409e-001, 8.10857155e-001,
                8.99207007e-001, 2.56294262e-001, 4.15989929e-001, 9.55962845e-001,
                9.06764652e-001, 5.55322937e-001, 3.42813423e-001, 4.50710269e-001,
                7.13277215e-001, 4.54604834e-001
            ]), 
            equal_nan=True)
                       )
        self.assertEqual(round(model.fitted_values.sum(),3), 6092.0)
        
    def test_Belgian_ln_rates(self):
        model = apc.Model()
        model.data_from_df(**apc.Belgian_lung_cancer())
        model.fit(family='log_normal_rates', predictor='APC')
        
        self.assertEqual(round(model.deviance,3), -44.854)
        self.assertTrue(
            np.allclose(
                model.parameters.sum().values,
                np.array([ 1.07781409,  7.31144396,  9.98467135, 15.2844013 ])
            )
        )
        self.assertEqual(round(model.fitted_values.sum(),3), 552.365)
        
    def test_Belgian_bin_dose_response(self):
        data = apc.Belgian_lung_cancer()
        dose = (data['response']/data['rate'] * 10**5).astype(int)
        model = apc.Model()
        model.data_from_df(data['response'], dose=dose, data_format='AP')
        model.fit('binomial_dose_response', 'APC')
        
        self.assertEqual(round(model.deviance,3), 20.227)
        self.assertTrue(np.allclose(
            model.parameters['P>|z|'].values,
            np.array([
                0.00000000e+00, 2.00519896e-11, 7.54361666e-02, 2.44877022e-01,
                3.78623439e-01, 4.49629054e-01, 1.71917810e-01, 7.15130617e-01,
                3.40413299e-01, 7.77302417e-01, 5.43443319e-01, 3.10193656e-01,
                3.27424287e-01, 3.02341523e-01, 4.90543748e-01, 8.10861452e-01,
                8.99103558e-01, 2.56217693e-01, 4.15956680e-01, 9.55997597e-01,
                9.06788804e-01, 5.55305952e-01, 3.42810478e-01, 4.50703994e-01,
                7.13277567e-01, 4.54607683e-01
            ]), 
            equal_nan=True)
                       )
        self.assertEqual(round(model.fitted_values.sum(),10), 0.0055324403)
        
if __name__ == '__main__':
    unittest.main()