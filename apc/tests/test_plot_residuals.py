import unittest
import apc
import matplotlib.pyplot as plt

class TestPlotResiduals(unittest.TestCase):

    def test_TA(self):
        model = apc.Model()
        model.data_from_df(apc.loss_TA(), data_format='CL')
        model.fit('od_poisson_response', 'AC')
        for sr in ('start', 'mean', 'end', False):
            model.plot_residuals(simplify_ranges=sr)
        plt.close('all')
        for sp in ('AC', 'AP', 'PA', 'PC', 'CA', 'CP'):
            model.plot_data_heatmaps(space=sp)
        plt.close('all')
        for r in ('anscombe', 'deviance', 'pearson', 'response'):
            model.plot_residuals(kind=r)

    def test_Belgian(self):
        model = apc.Model()
        model.data_from_df(**apc.Belgian_lung_cancer())
        model.fit('log_normal_rates', 'APC')
        for sr in ('start', 'mean', 'end', False):
            model.plot_residuals(simplify_ranges=sr)
        plt.close('all')
        for sp in ('AC', 'AP', 'PA', 'PC', 'CA', 'CP'):
            model.plot_data_heatmaps(space=sp)
        plt.close('all')
        for r in ('anscombe', 'deviance', 'pearson', 'response'):
            model.plot_residuals(kind=r)

if __name__ == '__main__':
    unittest.main()