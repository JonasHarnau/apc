import unittest
import apc

class TestRTest(unittest.TestCase):

    def test_VNJ(self):
        r = apc.r_test(apc.loss_VNJ(), 
                       'gen_log_normal_response', 'AC')
        self.assertEqual(round(r['R_stat'],3), 113.185)
        self.assertEqual(round(r['p_value'],5), 0.00114)
        self.assertEqual(round(r['power_at_R'],5), 0.82657)

    def test_BZ(self):
        r = apc.r_test(apc.loss_BZ(), 
                       'gen_log_normal_response', 'APC', 
                       R_stat='ls', R_dist='wls_ql',
                       data_format='CL')
        self.assertEqual(round(r['R_stat'],5), 113.92399)
        self.assertEqual(round(r['p_value'],5), 0.01754)
        self.assertEqual(round(r['power_at_R'],5), 0.86713)
        
    def test_TA(self):
        r = apc.r_test(apc.loss_TA(),
                       'od_poisson_response', 'Ad', 
                       R_stat='ql', R_dist='ls',
                       data_format='CL')
        self.assertEqual(round(r['R_stat'],5), 73.16521)
        self.assertEqual(round(r['p_value'],5), 0.78526)
        self.assertEqual(round(r['power_at_R'],5), 0.99896)
        
if __name__ == '__main__':
    unittest.main()