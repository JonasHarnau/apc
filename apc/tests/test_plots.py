import unittest
import pandas as pd
from apc.Model import Model
from apc.data.pre_formatted import loss_TA
import matplotlib.pyplot as plt

class TestFit(unittest.TestCase):

    def test_TA_plot_data_sums(self):
        model = Model()
        model.data_from_df(loss_TA(), time_adjust=1)
        model.plot_data_sums()
        model.plot_data_sums(logy=True)
        for simplify_ranges in ('start', 'mean', 'end', False):
            model.plot_data_sums(simplify_ranges)
        plt.close('all')
            
    def test_TA_plot_data_heatmaps(self):
        model = Model()
        model.data_from_df(loss_TA(), time_adjust=1)
        model.plot_data_heatmaps()
        for simplify_ranges in ('start', 'mean', 'end', False):
            model.plot_data_heatmaps(simplify_ranges)
        plt.close('all')
        for space in ('AC', 'AP', 'PA', 'PC', 'CA', 'CP'):
            model.plot_data_heatmaps(space=space)
        plt.close('all')
        model.plot_data_heatmaps(vmin=20)
        
    def test_TA_plot_data_within(self):
        model = Model()
        model.data_from_df(loss_TA(), time_adjust=1)
        model.plot_data_within()
        for simplify_ranges in ('start', 'mean', 'end', False):
            model.plot_data_within(simplify_ranges=simplify_ranges)
        plt.close('all')
        for aggregate in ('mean', 'sum'):
            model.plot_data_within(aggregate=aggregate)
        plt.close('all')
        model.plot_data_within(logy=True)
        model.plot_data_within(n_groups='all')
            
    def test_Belgian_plot_data_sums(self):
        data = pd.read_excel('./apc/data/Belgian_lung_cancer.xlsx', 
                             sheetname = ['response', 'rates'], 
                             index_col = 0)
        model = Model()
        model.data_from_df(data['response'], rate=data['rates'], 
                           data_format='AP')
        model.plot_data_sums()
        model.plot_data_sums(logy=True)
        for simplify_ranges in ('start', 'mean', 'end', False):
            model.plot_data_sums(simplify_ranges)
        plt.close('all')
            
    def test_Belgian_plot_data_heatmaps(self):
        data = pd.read_excel('./apc/data/Belgian_lung_cancer.xlsx', 
                             sheetname = ['response', 'rates'], 
                             index_col = 0)
        model = Model()
        model.data_from_df(data['response'], rate=data['rates'], 
                           data_format='AP')
        model.plot_data_heatmaps()
        for simplify_ranges in ('start', 'mean', 'end', False):
            model.plot_data_heatmaps(simplify_ranges=simplify_ranges)
        plt.close('all')
        for space in ('AC', 'AP', 'PA', 'PC', 'CA', 'CP'):
            model.plot_data_heatmaps(space=space)
        plt.close('all')
        model.plot_data_heatmaps(vmax=50)

    def test_Belgian_plot_data_within(self):
        data = pd.read_excel('./apc/data/Belgian_lung_cancer.xlsx', 
                             sheetname = ['response', 'rates'], 
                             index_col = 0)
        model = Model()
        model.data_from_df(data['response'], rate=data['rates'], 
                           data_format='AP')
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