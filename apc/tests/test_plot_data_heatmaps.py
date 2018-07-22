import unittest
import apc
import matplotlib.pyplot as plt

class TestPlotDataHeatmaps(unittest.TestCase):

    def test_TA(self):
        model = apc.Model()
        model.data_from_df(apc.loss_TA(), data_format='CL')
        model.plot_data_heatmaps()
        for simplify_ranges in ('start', 'mean', 'end', False):
            model.plot_data_heatmaps(simplify_ranges)
        plt.close('all')
        for space in ('AC', 'AP', 'PA', 'PC', 'CA', 'CP'):
            model.plot_data_heatmaps(space=space)
        plt.close('all')
        model.plot_data_heatmaps(vmin=20)
                    
    def test_Belgian(self):
        model = apc.Model()
        model.data_from_df(**apc.Belgian_lung_cancer())
        model.plot_data_heatmaps()
        for simplify_ranges in ('start', 'mean', 'end', False):
            model.plot_data_heatmaps(simplify_ranges=simplify_ranges)
        plt.close('all')
        for space in ('AC', 'AP', 'PA', 'PC', 'CA', 'CP'):
            model.plot_data_heatmaps(space=space)
        plt.close('all')
        model.plot_data_heatmaps(vmax=50)

if __name__ == '__main__':
    unittest.main()