import unittest
import apc

class TestFTest(unittest.TestCase):

    def test_VNJ(self):
        model = apc.Model()
        model.data_from_df(apc.loss_VNJ(), data_format='CL')
        model.fit('log_normal_response', 'AC')
        sub_models = [model.sub_model(coh_from_to=(1,5)), 
                      model.sub_model(coh_from_to=(6,10))]
        f = apc.f_test(model, sub_models)
        self.assertAlmostEqual(f['F_stat'], 0.242, 3)
        self.assertAlmostEqual(f['p_value'], 0.912, 3)

    def test_BZ(self):
        model = apc.Model()
        model.data_from_df(apc.loss_BZ(), data_format='CL')
        model.fit('od_poisson_response', 'APC')
        sub_models = [model.sub_model(per_from_to=(1977,1981)),
                      model.sub_model(per_from_to=(1982,1984)),
                      model.sub_model(per_from_to=(1985,1987))]
        f = apc.f_test(model, sub_models)
        self.assertAlmostEqual(f['F_stat'], 1.855, 3)
        self.assertAlmostEqual(f['p_value'], 0.133, 3)
        
if __name__ == '__main__':
    unittest.main()