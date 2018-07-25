import unittest
import apc
import numpy as np

class TestForecast(unittest.TestCase):

    def test_asbestos(self):
        model = apc.Model()
        model.data_from_df(apc.asbestos(), data_format='PA')
        model.fit('poisson_response', 'AC')
        model.forecast()

        self.assertTrue(np.allclose(
            model.forecasts['Period'].iloc[1:,:].sum().values,
            np.array([
                 89522.62828757,  19154.90737731,   2335.13483308,  18598.16171896,
                102442.41697952, 114070.62982483, 121029.64716106, 134083.60634223
            ]),
            rtol=1e-3
        )
                       )
        self.assertTrue(np.allclose(
            model.forecasts['Age'].iloc[1:,:].sum().values,
            np.array([
                 91459.55642795,  18533.1328676 ,   1953.54949513,  18419.97856853,
                103959.96458617, 115210.72186887, 121943.847244  , 134574.07067381
            ]),
            rtol=1e-3
        )
                       )
        self.assertTrue(np.allclose(
            model.forecasts['Cohort'].iloc[1:,:].sum().values,
            np.array([
                 91457.32811512,  31828.56234787,   2042.01815982,  31462.3595823 ,
                112925.36718224, 132247.27202107, 143810.65433367, 165501.63646686
            ]),
            rtol=1e-3
        )
                       )
        self.assertTrue(np.allclose(
            model.forecasts['Total'].sum().values,
            np.array([
                 91459.55642795,  17726.0160027 ,    302.42281069,  17723.43600354,
                103415.57253358, 114176.35998707, 120616.25814139, 132696.43607104
            ]),
            rtol=1e-3
        )
                       )
        
    def test_TA(self):
        model = apc.Model()
        model.data_from_df(apc.loss_TA(), data_format='CL')
        model.fit('od_poisson_response', 'AC')
        model.forecast()

        self.assertTrue(np.allclose(
            model.forecasts['Period'].iloc[1:,:].sum().values,
            np.array([
                13454319.7860026 ,  3264385.92189263,  2168550.32629664,
                 2371474.89252264,   527736.31618658, 15678560.7866679 ,
                17716020.93110489, 18965575.06511772, 21401447.92641674
            ])
        )
                       )
        self.assertTrue(np.allclose(
            model.forecasts['Age'].iloc[1:,:].sum().values,
            np.array([
                17824052.09094379,  5446971.56429764,  2703382.56455396,
                 4610723.51207401,   699136.02022292, 21535432.35438537,
                24935149.09611903, 27020161.71376697, 31084672.01592125
            ])
        )
                       )
        self.assertTrue(np.allclose(
            model.forecasts['Cohort'].iloc[1:,:].sum().values,
            np.array([
                18586221.79737546,  5480908.10291801,  2617925.92215297,
                 4585571.92139958,   729031.5957391 , 22320725.25838809,
                25741623.42490841, 27839626.40066406, 31929460.02408297
            ])
        )
                       )
        self.assertTrue(np.allclose(
            model.forecasts['Total'].sum().values,
            np.array([
                18680855.61192424,  2952921.04370912,   993729.36519777,
                 2682411.51615798,   732743.54115671, 20692875.08949177,
                22535935.034052  , 23666265.45020602, 25869724.35611854
            ])
        )
                       )

    def test_BZ(self):
        model = apc.Model()
        model.data_from_df(apc.loss_BZ(), data_format='CL')
        model.fit('od_poisson_response', 'AC')
        model.forecast(method='n_poisson')
        fc = model.forecast(attach_to_self=False)
        self.assertEqual(fc['method'], 't_odp')
        model.forecast([0.9])

    def test_BZ(self):
        model = apc.Model()
        model.data_from_df(apc.loss_BZ(), data_format='CL')
        model.fit('log_normal_response', 'AC')
        model.forecast()
        self.assertEqual(model.forecasts['method'], 't_gln')
        
if __name__ == '__main__':
    unittest.main()