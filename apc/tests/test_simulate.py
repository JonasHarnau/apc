import unittest
import numpy as np
import apc

class TestSimulate(unittest.TestCase):

    def test_TA(self):
        model = apc.Model()
        model.data_from_df(apc.loss_TA(), data_format='CL')
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
