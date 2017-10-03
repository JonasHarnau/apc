import unittest
import pandas as pd
from apc.Model import Model
from apc.data.pre_formatted import loss_TA


class TestDataFromDf(unittest.TestCase):

    def test_TA(self):
        model = Model()
        model.data_from_df(loss_TA(), time_adjust=1)
        
        self.assertEqual(model.data_format, 'CA')
        self.assertEqual(model.I, 10)
        self.assertEqual(model.J, 10)
        self.assertEqual(model.K, 10)
        self.assertEqual(model.L, 0)
        self.assertEqual(model.n, 55)
        self.assertEqual(round(model.data_vector.sum()['response'],3), 
                         34358090.0)
        
    def test_Belgian(self):
        data = pd.read_excel('./apc/data/Belgian_lung_cancer.xlsx', 
                             sheetname = ['response', 'rates'], 
                             index_col = 0)
        model = Model()
        model.data_from_df(data['response'], rate=data['rates'], 
                           data_format='AP')
        
        self.assertEqual(model.data_format, 'AP')
        self.assertEqual(model.I, 11)
        self.assertEqual(model.J, 4)
        self.assertEqual(model.K, 14)
        self.assertEqual(model.L, 10)
        self.assertEqual(model.n, 44)
        self.assertEqual(round(model.data_vector.sum()['response'],3), 
                         6092.0)
        self.assertEqual(round(model.data_vector.sum()['dose'],3), 
                         590.843)
        self.assertEqual(round(model.data_vector.sum()['rate'],3),
                         553.150)
    
if __name__ == '__main__':
    unittest.main()