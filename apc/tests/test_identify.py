import unittest
import pandas as pd
from apc.Model import Model
from apc.data.pre_formatted import loss_TA
from apc.data.pre_formatted import Belgian_lung_cancer

class TestFit(unittest.TestCase):

    def test_TA_identify(self):
        model = Model()
        model.data_from_df(loss_TA())
        predictors = ['APC', 'AP', 'AC', 'PC', 'Ad', 'Pd', 'Cd',
                      'A', 'P', 'C', 't', 'tA', 'tP', 'tC', '1']
        for p in predictors:
            model.fit('log_normal_response', p)
            model.identify('sum_sum')
            model.identify('detrend')
            model.fit('od_poisson_response', p)
            model.identify('sum_sum')
            model.identify('detrend')
            
    def test_Belgian_Lung_Cancer_identify(self):
        model = Model()
        model.data_from_df(Belgian_lung_cancer()['response'],
                          rate=Belgian_lung_cancer()['rate'],
                          data_format='AP')
        predictors = ['APC', 'AP', 'AC', 'PC', 'Ad', 'Pd', 'Cd',
                      'A', 'P', 'C', 't', 'tA', 'tP', 'tC', '1']
        for p in predictors:
            model.fit('od_poisson_response', p)
            model.identify('sum_sum')
            model.identify('detrend')
            model.fit('log_normal_rates', p)
            model.identify('sum_sum')
            model.identify('detrend')
            model.fit('gaussian_rates', p)
            model.identify('sum_sum')
            model.identify('detrend')

if __name__ == '__main__':
    unittest.main()