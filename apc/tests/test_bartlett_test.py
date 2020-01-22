import unittest
import apc

class TestBartlettTest(unittest.TestCase):

    def test_VNJ(self):
        model = apc.Model()
        model.data_from_df(apc.loss_VNJ(), data_format='CL')
        model.fit('log_normal_response', 'AC')
        sub_models = [model.sub_model(coh_from_to=(1,5)), 
                      model.sub_model(coh_from_to=(6,10))]
        bartlett = apc.bartlett_test(sub_models)
        self.assertAlmostEqual(bartlett['B'], 2.794, 3)
        self.assertAlmostEqual(bartlett['LR'], 2.956, 3)
        self.assertAlmostEqual(bartlett['C'], 1.058, 3)
        self.assertAlmostEqual(bartlett['m'], 2, 3)
        self.assertAlmostEqual(bartlett['p_value'], 0.095, 3)
        
    def test_BZ(self):
        model = apc.Model()
        model.data_from_df(apc.loss_BZ(), data_format='CL')
        model.fit('od_poisson_response', 'APC')
        sub_models = [model.sub_model(per_from_to=(1977,1981)),
                      model.sub_model(per_from_to=(1982,1984)),
                      model.sub_model(per_from_to=(1985,1987))]
        bartlett = apc.bartlett_test(sub_models)
        self.assertAlmostEqual(bartlett['B'], 1.835, 3)
        self.assertAlmostEqual(bartlett['LR'], 2.235, 3)
        self.assertAlmostEqual(bartlett['C'], 1.218, 3)
        self.assertAlmostEqual(bartlett['m'], 3, 3)
        self.assertAlmostEqual(bartlett['p_value'], 0.4, 3)

if __name__ == '__main__':
    unittest.main()