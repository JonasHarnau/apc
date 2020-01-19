import unittest
import apc

class TestRTest(unittest.TestCase):

    def test_VNJ(self):
        r = apc.r_test(apc.loss_VNJ(), 
                       'gen_log_normal_response', 'AC')
        self.assertAlmostEqual(r['R_stat'], 113.185, 3)
        self.assertAlmostEqual(r['p_value'], 0.00114, 5)
        self.assertAlmostEqual(r['power_at_R'], 0.82657, 5)

    def test_BZ(self):
        r = apc.r_test(apc.loss_BZ(), 
                       'gen_log_normal_response', 'APC', 
                       R_stat='ls', R_dist='wls_ql',
                       data_format='CL')
        self.assertAlmostEqual(r['R_stat'], 113.92399, 5)
        self.assertAlmostEqual(r['p_value'], 0.01754, 5)
        self.assertAlmostEqual(r['power_at_R'], 0.86713, 5)
        
    def test_TA(self):
        r = apc.r_test(apc.loss_TA(),
                       'od_poisson_response', 'Ad', 
                       R_stat='ql', R_dist='ls',
                       data_format='CL')
        self.assertAlmostEqual(r['R_stat'], 73.16521, 5)
        self.assertAlmostEqual(r['p_value'], 0.78526, 5)
        self.assertAlmostEqual(r['power_at_R'], 0.99896, 5)
        
if __name__ == '__main__':
    unittest.main()