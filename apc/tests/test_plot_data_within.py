import unittest
import apc
import matplotlib.pyplot as plt

class TestPlotDataWithin(unittest.TestCase):

    def test_TA(self):
        model = apc.Model()
        model.data_from_df(apc.loss_TA(), data_format='CL')
        model.plot_data_within()
        for simplify_ranges in ('start', 'mean', 'end', False):
            model.plot_data_within(simplify_ranges=simplify_ranges)
        plt.close('all')
        for aggregate in ('mean', 'sum'):
            model.plot_data_within(aggregate=aggregate)
        plt.close('all')
        model.plot_data_within(logy=True)
        model.plot_data_within(n_groups='all')
            
    def test_Belgian(self):
        model = apc.Model()
        model.data_from_df(**apc.Belgian_lung_cancer())
        model.plot_data_within()
        for simplify_ranges in ('start', 'mean', 'end', False):
            model.plot_data_within(simplify_ranges=simplify_ranges)
        plt.close('all')
        for aggregate in ('mean', 'sum'):
            model.plot_data_within(aggregate=aggregate)
        plt.close('all')
        model.plot_data_within(logy=True)
        model.plot_data_within(n_groups='all')

if __name__ == '__main__':
    unittest.main()