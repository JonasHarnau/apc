import unittest
import pandas as pd
import apc
from importlib.resources import files

class TestDataFromDf(unittest.TestCase):

    def test_asbestos(self):
        model = apc.Model()
        model.data_from_df(apc.asbestos())
        
        self.assertEqual(model.data_format, 'PA')
        self.assertEqual(model.I, 70)
        self.assertEqual(model.J, 41)
        self.assertEqual(model.K, 110)
        self.assertEqual(model.L, 69)
        self.assertEqual(model.n, 2870)
        self.assertEqual(model.data_vector.sum()['response'], 32164)    
    
    def test_TA(self):
        model = apc.Model()
        model.data_from_df(apc.loss_TA())
        
        self.assertEqual(model.data_format, 'CA')
        self.assertEqual(model.I, 10)
        self.assertEqual(model.J, 10)
        self.assertEqual(model.K, 10)
        self.assertEqual(model.L, 0)
        self.assertEqual(model.n, 55)
        self.assertEqual(model.time_adjust, 1)
        self.assertAlmostEqual(
            model.data_vector.sum()['response'], 34358090.0, 3)

    def test_BZ(self):
        model = apc.Model()
        model.data_from_df(apc.loss_BZ(), data_format='CL')
        
        self.assertEqual(model.data_format, 'CL')
        self.assertEqual(model.I, 11)
        self.assertEqual(model.J, 11)
        self.assertEqual(model.K, 11)
        self.assertEqual(model.L, 0)
        self.assertEqual(model.n, 66)
        self.assertEqual(model.time_adjust, 0)
        self.assertAlmostEqual(
            model.data_vector.sum()['response'], 10221194.0, 3) 
        
    def test_Belgian(self):
        data = pd.read_excel(
            files('apc') / 'data/Belgian_lung_cancer.xlsx', 
            sheet_name = ['response', 'rates'], 
            index_col = 0)
        model = apc.Model()
        model.data_from_df(data['response'], rate=data['rates'], 
                           data_format='AP')
        
        self.assertEqual(model.data_format, 'AP')
        self.assertEqual(model.I, 11)
        self.assertEqual(model.J, 4)
        self.assertEqual(model.K, 14)
        self.assertEqual(model.L, 10)
        self.assertEqual(model.n, 44)
        self.assertAlmostEqual(model.data_vector.sum()['response'], 
                         6092.0, 3)
        self.assertAlmostEqual(model.data_vector.sum()['dose'], 
                         590.843, 3)
        self.assertAlmostEqual(model.data_vector.sum()['rate'],
                         553.150, 3)
    
if __name__ == '__main__':
    unittest.main()