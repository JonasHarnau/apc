import unittest
import apc
import matplotlib.pyplot as plt

class TestPlotForecast(unittest.TestCase):

    def test_asbestos(self):
        model = apc.Model()
        model.data_from_df(apc.asbestos(), data_format='PA')
        model.fit('poisson_response', 'AC')
        model.forecast()
        model.plot_forecast()
        model.plot_forecast(ic=True)
        model.plot_forecast('Age', aggregate=True)
        model.plot_forecast('Cohort')
        model.plot_forecast(from_to=(1986, 2032))
        model.plot_forecast(figsize=(8,6))
        plt.close('all')
        
    def test_TA(self):
        model = apc.Model()
        model.data_from_df(apc.loss_TA(), data_format='CL')
        model.fit('od_poisson_response', 'Ad')
        model.forecast()
        model.plot_forecast()
        model.plot_forecast(ic=True)
        model.plot_forecast('Age', aggregate=True)
        model.plot_forecast('Cohort')
        model.plot_forecast(from_to=(8, 13))
        plt.close('all')        

if __name__ == '__main__':
    unittest.main()