import unittest
import pandas as pd
import numpy as np
from apc.Model import Model
from apc.data.pre_formatted import loss_TA

class TestFit(unittest.TestCase):

    def test_TA_odp(self):
        model = Model()
        model.data_from_df(loss_TA(), time_adjust=1)
        model.fit(family='od_poisson_response', predictor='AC')
        
        self.assertEqual(round(model.deviance,3), 1903014.004)
        self.assertTrue(np.allclose(
            model.para_table['P>|t|'].values,
            np.array([
                np.nan,   3.50394713e-09,   1.65458672e-02,
        1.35749045e-04,   9.20260458e-01,   8.11965018e-03,
        4.67631895e-01,   5.17889796e-01,   5.58375527e-01,
        2.35466770e-01,   1.08695074e-01,   1.67254401e-01,
        9.84463128e-01,   7.89979049e-01,   6.32105935e-01,
        8.62556947e-01,   7.99177515e-01,   3.19345261e-01,
        9.21765023e-01
            ]), 
            equal_nan=True)
                       )
        self.assertEqual(round(model.fitted_values.sum(),3), 34358090.000)
        
    def test_Belgian_poisson(self):
        data = pd.read_excel('./apc/data/Belgian_lung_cancer.xlsx', 
                             sheetname = ['response', 'rates'], 
                             index_col = 0)
        model = Model()
        model.data_from_df(data['response'], rate=data['rates'], 
                           data_format='AP')
        model.fit(family='poisson_response', predictor='APC')
        
        self.assertEqual(round(model.deviance,3), 20.375)
        self.assertTrue(np.allclose(
            model.para_table['P>|z|'].values,
            np.array([
            np.nan,  0.        ,  0.17280369,  0.23579819,  0.41136118,
        0.47691628,  0.13648925,  0.60152048,  0.2058326 ,  0.89385125,
        0.13918414,  0.01034656,  0.42659362,  0.29697439,  0.55719711,
        0.51180712,  0.69814128,  0.04134531,  0.85814056,  0.17150821,
        0.07831835,  0.78393006,  0.90303859,  0.43318881,  0.76576625,
        0.61892638
            ]), 
            equal_nan=True)
                       )
        self.assertEqual(round(model.fitted_values.sum(),3), 6092.0)
        
    def test_Belgian_ln_rates(self):
        data = pd.read_excel('./apc/data/Belgian_lung_cancer.xlsx', 
                             sheetname = ['response', 'rates'], 
                             index_col = 0)
        model = Model()
        model.data_from_df(data['response'], rate=data['rates'], 
                           data_format='AP')
        model.fit(family='log_normal_rates', predictor='APC')
        
        self.assertEqual(round(model.deviance,3), -44.854)
        self.assertTrue(np.allclose(
            model.para_table['P>|z|'].values,
            np.array([
         2.09331128e-29,   6.16131902e-03,   3.39144271e-01,
         4.89205018e-02,   4.80568185e-01,   5.47475298e-01,
         4.81296622e-01,   8.34983934e-01,   7.87164891e-01,
         9.23124206e-01,   8.14693970e-01,   7.79825603e-01,
         6.05257510e-01,   9.31722939e-01,   8.14136002e-01,
         8.94116268e-01,   9.15473641e-01,   8.20280965e-01,
         7.58593410e-01,   9.41457948e-01,   7.82681623e-01,
         6.56188983e-01,   7.63893913e-01,   1.68388206e-01,
         2.88748761e-02,   1.43842965e-02
            ]), 
            equal_nan=True)
                       )
        self.assertEqual(round(model.fitted_values.sum(),3), 552.365)
        

if __name__ == '__main__':
    unittest.main()
    
    
    
    
    