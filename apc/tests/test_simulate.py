import unittest
import pandas as pd
import numpy as np
from apc.Model import Model
from apc.data.pre_formatted import loss_TA

class TestFit(unittest.TestCase):

    def test_TA_simulate(self):
        model = Model()
        model.data_from_df(loss_TA(), time_adjust=1)
        # Poisson
        model.fit('poisson_response', 'APC')
        model.simulate(repetitions=100)
        model.simulate(repetitions=100, seed=1)
        self.assertTrue(np.allclose(
            model.draws, 
            model.simulate(
                repetitions=100, seed=1, attach_to_self=False
            )
        )
                       )

if __name__ == '__main__':
    unittest.main()
