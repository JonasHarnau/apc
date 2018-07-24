import unittest
import numpy as np
import apc

class TestCLBootstrap(unittest.TestCase):

    def test_TA(self):
        model = apc.Model()
        model.data_from_df(apc.loss_TA(), data_format='CL')
        model.fit('od_poisson_response', 'AC')
        apc.bootstrap_forecast(model, B=10)
        apc.bootstrap_forecast(model, B=10, quantiles=[0.9])
        apc.bootstrap_forecast(model, B=10, adj_residuals=False)
        f1 = apc.bootstrap_forecast(model, B=10, seed=1)
        f2 = apc.bootstrap_forecast(model, B=10, seed=1)
        for key in f1.keys():
            self.assertTrue(np.allclose(f1[key].values, f2[key].values))

if __name__ == '__main__':
    unittest.main()