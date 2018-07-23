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
            model.forecasts['Period'].sum().values,
            np.array([
                 91459.55642795,  19194.38303242,   2379.14537886,  18602.76915763,
                104405.97104465, 116058.14805281, 123031.50697592, 136112.36858894
            ]),
            rtol=1e-4
        )
                       )
        self.assertTrue(np.allclose(
            model.forecasts['Age'].sum().values,
            np.array([
                 91459.55642795,  18530.10224627,   1953.54949942,  18416.93017594,
                103957.92046314, 115206.83797136, 121938.8623155 , 134567.02039431
            ]),
            rtol=1e-4
        )
                       )
        self.assertTrue(np.allclose(
            model.forecasts['Cohort'].sum().values,
            np.array([
                91459.55642795,  31825.08791674,   2043.51091326,  31457.21623455,
                112925.25202688, 132245.04767125, 143807.16771585, 165495.78204423
            ]),
            rtol=1e-4
        )
                       )
        self.assertTrue(np.allclose(
            model.forecasts['Total'].sum().values,
            np.array([
                 91459.55642795,  17722.25572794,    302.42281069,  17719.67518128,
                103413.0362668 , 114171.54100107, 120610.07303981, 132687.68836385
            ]),
            rtol=1e-4
        )
                       )
        
    def test_TA(self):
        model = apc.Model()
        model.data_from_df(apc.loss_TA(), data_format='CL')
        model.fit('od_poisson_response', 'AC')
        model.forecast()
        self.assertTrue(np.allclose(
            model.forecasts['Period'].sum().values,
            np.array([
                18680855.61192424,  3853068.68918822,  2694176.16680765,
                2632065.07486532,   732743.54115671, 21306204.93088691,
                23711090.28324363, 25185982.65945481, 28061128.45326858
            ])
        )
                       )
        self.assertTrue(np.allclose(
            model.forecasts['Age'].sum().values,
            np.array([
                18680855.61192424,  5752510.40523726,  2916201.5473328,
                4831972.96156796,   732743.54115671, 22600419.59921904,
                26190837.81617259, 28392805.76046776, 32685308.03064466
            ])
        )
                       )
        self.assertTrue(np.allclose(
            model.forecasts['Cohort'].sum().values,
            np.array([
                18680855.61192424,  5471357.3881681,  2688654.172517,
                4532709.71500557,   732743.54115671, 22408851.54231987,
                25823788.64874514, 27918135.76536402, 32000842.68029259
            ])
        )
                       )
        self.assertTrue(np.allclose(
            model.forecasts['Total'].sum().values,
            np.array([
                18680855.61192424,  2734655.62105958,   993729.36519777,
                2440067.66656778,   732743.54115671, 20544156.49012863,
                22250986.49429648, 23297768.43679893, 25338358.48090614
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
        
if __name__ == '__main__':
    unittest.main()