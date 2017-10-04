import unittest
import pandas as pd
from apc.Model import Model
from apc.data.pre_formatted import loss_TA

class TestFit(unittest.TestCase):

    def test_TA_plot_data_sums(self):
        model = Model()
        model.data_from_df(loss_TA(), time_adjust=1)
        model.plot_data_sums()
        model.plot_data_sums(logy=True)
        for simplify_ranges in ('start', 'mean', 'end', False):
            model.plot_data_sums(simplify_ranges)
            
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
        

if __name__ == '__main__':
    unittest.main()
    
    
    
    
    