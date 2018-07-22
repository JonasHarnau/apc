import unittest
import apc
import matplotlib.pyplot as plt

class TestPlotDataSums(unittest.TestCase):

    def test_TA(self):
        model = apc.Model()
        model.data_from_df(apc.loss_TA(), data_format='CL')
        model.plot_data_sums()
        model.plot_data_sums(logy=True)
        for simplify_ranges in ('start', 'mean', 'end', False):
            model.plot_data_sums(simplify_ranges)
        plt.close('all')

    def test_Belgian(self):
        model = apc.Model()
        model.data_from_df(**apc.Belgian_lung_cancer())
        model.plot_data_sums()
        model.plot_data_sums(logy=True)
        for simplify_ranges in ('start', 'mean', 'end', False):
            model.plot_data_sums(simplify_ranges)
        plt.close('all')

if __name__ == '__main__':
    unittest.main()