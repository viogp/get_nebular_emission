#python -m unittest tests/test_att.py 

import unittest
import numpy as np
from numpy.testing import assert_allclose

import gne.gne_const as c
import gne.gne_att as att

class TestPredict(unittest.TestCase):
    def test_get_f_saito20(self):
        expect = 1
        self.assertAlmostEqual(att.get_f_saito20(2.9),expect,2)
        for z in [1,2.8]:
            expect=0.44 + 0.2*z
            self.assertAlmostEqual(att.get_f_saito20(z),expect,2)

    def test_get_A_lambda(self):  
        tau = 1.0; costheta = 1.0; albedo = 0.0
        expected = -2.5 * np.log10((1.0 - np.exp(-1.0)) / 1.0)
        result = att.get_A_lambda(tau, costheta=costheta, albedo=albedo)       
        np.testing.assert_almost_equal(result, expected, decimal=10)

        tau = np.array([0.5, 1.0, 1.5, 2.0])
        costheta = 0.6
        albedo = 0.4        
        result = att.get_A_lambda(tau, costheta=costheta, albedo=albedo)
        # Verify output shape matches tau
        assert result.shape == tau.shape
        # Verify all results are finite
        assert np.all(np.isfinite(result))
        # Verify A_lambda increases with tau (more optical depth = more attenuation)
        assert np.all(np.diff(result) > 0)

        tau = np.array([0.5, 1.0, 1.5])
        costheta = np.array([0.5, 0.6, 0.7])
        albedo = np.array([0.2, 0.3, 0.4])
        result = att.get_A_lambda(tau, costheta=costheta, albedo=albedo)
        assert result.shape == tau.shape
        assert np.all(np.isfinite(result))

    def test_find_line_index(self):
        line_names = ['OII3727','Hbeta']
        ind = att.find_line_index('Hbeta',line_names)
        self.assertEqual(1, ind)
        ind = att.find_line_index('OII3726',line_names)
        self.assertEqual(0, ind)
        ind = att.find_line_index('OII3728',line_names)
        self.assertEqual(0, ind)
        ind = att.find_line_index('OII3730',line_names)
        self.assertEqual(None, ind)

        
if __name__ == '__main__':
    unittest.main(verbosity=2)
