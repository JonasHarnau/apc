import unittest
import numpy as np
import apc

class TestCLBootstrap(unittest.TestCase):

    def test_TA(self):
        apc.bootstrap_forecast(apc.loss_TA(), B=10)
        apc.bootstrap_forecast(apc.loss_TA(), B=10, quantiles=[0.9])
        apc.bootstrap_forecast(apc.loss_TA(), B=10, adj_residuals=False)
        f1 = apc.bootstrap_forecast(apc.loss_TA(), B=10, seed=1)
        f2 = apc.bootstrap_forecast(apc.loss_TA(), B=10, seed=1)
        for key in f1.keys():
            self.assertTrue(np.allclose(f1[key].values, f2[key].values))

if __name__ == '__main__':
    unittest.main()