import unittest
import numpy as np
import apc

class TestIdentify(unittest.TestCase):

    def test_BZ_ln(self):
        model = apc.Model()
        model.data_from_df(apc.loss_BZ())
        model.fit('log_normal_response', 'APC')
        model.identify()

        self.assertTrue(
            np.allclose(
            model.parameters_adhoc.sum().values,
            np.array(
                [11.52208551,   1.92807476, 309.55066566,  16.5398655]
            )
            )
        )
        self.assertTrue(
            np.allclose(
            model.identify('sum_sum', attach_to_self=False).sum().values,
            np.array(
                [-17.03573805,   5.27028714,  18.73830563,  19.19957677]
            )
            )
        )
    
    def test_TA_odp(self):
        model = apc.Model()
        model.data_from_df(apc.loss_TA(), data_format='CL')
        model.fit('od_poisson_response', 'AC')
        model.identify()

        self.assertTrue(
            np.allclose(
            model.parameters_adhoc.sum().values,
            np.array(
                [19.93493555, 12.35339911, 23.27262888, 12.98041616]
            )
            )
        )
        self.assertTrue(
            np.allclose(
            model.identify('sum_sum', attach_to_self=False).sum().values,
            np.array(
                [-40.35513352,  19.04311385, -62.3882387 ,   8.99524907]
            )
            )
        )

    def test_Belgian_gauss_rates(self):
        model = apc.Model()
        model.data_from_df(**apc.Belgian_lung_cancer())
        predictors = ['APC', 'AP', 'AC', 'PC', 'Ad', 'Pd', 'Cd',
                      'A', 'P', 'C', 't', 'tA', 'tP', 'tC', '1']
        for p in predictors:
            model.fit('gaussian_rates', p)
            model.identify('sum_sum')
            model.identify('detrend')

    def test_Belgian_bin_dose_response(self):
        data = apc.Belgian_lung_cancer()
        dose = (data['response']/data['rate'] * 10**5).astype(int)
        model = apc.Model()
        model.data_from_df(data['response'], dose=dose, data_format='AP')
        model.fit('binomial_dose_response', 'A')
        model.identify()

        self.assertTrue(
            np.allclose(
            model.parameters_adhoc.sum().values,
            np.array(
                [-6.56083837,  3.05256213, 18.37617231,  3.71767601]
            )
            )
        )
        self.assertTrue(
            np.allclose(
            model.identify('sum_sum', attach_to_self=False).sum().values,
            np.array(
                [-14.19804504,    3.52539429, -220.13119605,    5.58065264]
            )
            )
        )
        
if __name__ == '__main__':
    unittest.main()