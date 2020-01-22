import unittest
import apc
import matplotlib.pyplot as plt

class TestPlotFit(unittest.TestCase):

    def test_TA(self):
        model = apc.Model()
        model.data_from_df(apc.loss_TA(), data_format='CL')
        model.fit('od_poisson_response', 'Ad')
        model.plot_parameters()
        for style in ('detrend', 'sum_sum'):
            model.plot_parameters(style)
        plt.close('all')
        for sr in ('start', 'mean', 'end', False):
            model.plot_parameters(simplify_ranges=sr)
        plt.close('all')
        model.plot_parameters(around_coef=False)
        
    def test_asbestos(self):
        model = apc.Model()
        model.data_from_df(apc.asbestos(), data_format='PA')
        model.fit('poisson_response', 'AC')
        model.plot_parameters()
        for style in ('detrend', 'sum_sum'):
            model.plot_parameters(style)
        plt.close('all')
        for sr in ('start', 'mean', 'end', False):
            model.plot_parameters(simplify_ranges=sr)
        plt.close('all')
        model.plot_parameters(around_coef=False)
        
    def test_Belgian(self):
        model = apc.Model()
        model.data_from_df(**apc.Belgian_lung_cancer())
        model.fit('log_normal_rates', 'APC')
        model.plot_parameters()
        for style in ('detrend', 'sum_sum'):
            model.plot_parameters(style)
        plt.close('all')
        for sr in ('start', 'mean', 'end', False):
            model.plot_parameters(simplify_ranges=sr)
        plt.close('all')
        model.plot_parameters(around_coef=False)
        
if __name__ == '__main__':
    unittest.main()