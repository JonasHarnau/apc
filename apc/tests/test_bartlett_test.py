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
        self.assertEqual(round(bartlett['B'], 3), 2.794)
        self.assertEqual(round(bartlett['LR'], 3), 2.956)
        self.assertEqual(round(bartlett['C'], 3), 1.058)
        self.assertEqual(round(bartlett['m'], 3), 2)
        self.assertEqual(round(bartlett['p_value'], 3), 0.095)
        
    def test_BZ(self):
        model = apc.Model()
        model.data_from_df(apc.loss_BZ(), data_format='CL')
        model.fit('od_poisson_response', 'APC')
        sub_models = [model.sub_model(per_from_to=(1977,1981)),
                      model.sub_model(per_from_to=(1982,1984)),
                      model.sub_model(per_from_to=(1985,1987))]
        bartlett = apc.bartlett_test(sub_models)
        self.assertEqual(round(bartlett['B'], 3), 1.835)
        self.assertEqual(round(bartlett['LR'], 3), 2.235)
        self.assertEqual(round(bartlett['C'], 3), 1.218)
        self.assertEqual(round(bartlett['m'], 3), 3)
        self.assertEqual(round(bartlett['p_value'], 3), 0.4)

if __name__ == '__main__':
    unittest.main()