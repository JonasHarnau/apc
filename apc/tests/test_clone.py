import unittest
import apc

class TestClone(unittest.TestCase):

    def test_asbestos(self):
        model = apc.Model()
        model.data_from_df(apc.asbestos(), data_format='PA')
        clone = model.clone()
        for att in ('data_format', 'I', 'J', 'K', 'L',
                    'n', 'time_adjust'):
            self.assertEqual(getattr(model, att), getattr(clone, att))
        model.K = None
        self.assertFalse(clone.K == model.K)

if __name__ == '__main__':
    unittest.main()