import unittest
import numpy as np
import apc

class TestSimulate(unittest.TestCase):

    def test_asbestos(self):
        model = apc.Model()
        model.data_from_df(apc.asbestos(), data_format='PA')
        model.fit('poisson_response', 'AC')
        model.simulate(repetitions=10)
        model.simulate(repetitions=10, poisson_dgp='multinomial')
        model.simulate(repetitions=10, seed=1)
        self.assertTrue(np.allclose(
            model.draws, 
            model.simulate(repetitions=10, seed=1, attach_to_self=False)
        )
                       )
        model.simulate(
            repetitions=10, fitted_values=model.fitted_values * 10)

    def test_TA_odp(self):
        model = apc.Model()
        model.data_from_df(apc.loss_TA(), data_format='CL')
        model.fit('od_poisson_response', 'AC')
        model.simulate(repetitions=10)
        model.simulate(repetitions=10, sigma2=10)
        model.simulate(repetitions=10, od_poisson_dgp='neg_binomial')

    def test_BZ_ln(self):
        model = apc.Model()
        model.data_from_df(apc.loss_BZ(), data_format='CL')
        model.fit('log_normal_response', 'AC')
        model.simulate(repetitions=10)
        model.simulate(
            repetitions=10, fitted_values=model.fitted_values * 10)
        model.simulate(repetitions=10, sigma2=10)

    def test_VNJ_gln(self):
        model = apc.Model()
        model.data_from_df(apc.loss_VNJ(), data_format='CL')
        model.fit('gen_log_normal_response', 'APC')
        model.simulate(repetitions=10)
        model.simulate(
            repetitions=10, fitted_values=model.fitted_values * 10)
        model.simulate(repetitions=10, sigma2=10)
        
    def test_Belgian_ln_rates(self):
        model = apc.Model()
        model.data_from_df(**apc.Belgian_lung_cancer())
        model.fit('log_normal_rates', 'AC')
        model.simulate(repetitions=10)
        model.simulate(
            repetitions=10, fitted_values=model.fitted_values * 10)
        model.simulate(repetitions=10, sigma2=10)
        
    def test_Belgian_gauss_rates(self):
        model = apc.Model()
        model.data_from_df(**apc.Belgian_lung_cancer())
        model.fit('gaussian_rates', 'AC')
        model.simulate(repetitions=10)
        model.simulate(
            repetitions=10, fitted_values=model.fitted_values * 10)
        model.simulate(repetitions=10, sigma2=10)
        
    def test_Belgian_bin_dose_response(self):
        data = apc.Belgian_lung_cancer()
        dose = ((data['response'] / data['rate']) * 10**5).astype(int)
        model = apc.Model()
        model.data_from_df(data['response'], dose=dose, data_format='AP')
        model.fit('binomial_dose_response', 'Ad')
        model.simulate(repetitions=10)
        model.simulate(repetitions=10, fitted_values=model.fitted_values * 10)

    def test_Belgian_pois_dose_response(self):
        model = apc.Model()
        model.data_from_df(**apc.Belgian_lung_cancer())
        model.fit('poisson_dose_response', 'APC')
        model.simulate(repetitions=10)
        model.simulate(repetitions=10, fitted_values=model.fitted_values * 10)
                
if __name__ == '__main__':
    unittest.main()