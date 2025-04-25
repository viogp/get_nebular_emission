#python3 -m unittest tests/test_model_UnH.py 

from unittest import TestCase
import numpy as np

import src.gne_const as c
import src.gne_model_UnH as unh

class TestPredict(TestCase):
    def test_get_alphaB(self):
        decimals = 8
        self.assertAlmostEqual(unh.get_alphaB(5000.),4.54e-13,decimals)
        self.assertAlmostEqual(unh.get_alphaB(1000.),4.54e-13,decimals)
        val = (1.43e-13+2.59e-13)/2.
        self.assertAlmostEqual(unh.get_alphaB(15000.),val,decimals)
        self.assertAlmostEqual(unh.get_alphaB(30000.),1.43e-13,decimals)  

    #def test_get_Q_agn(self):
    #    atol = 1e-7
    #    Lagn = np.array([0.])
    #    expected = Lagn
    #    vals = unh.get_Q_agn(Lagn,-1.7)
    #    np.testing.assert_allclose(vals,expected,atol=atol)

    #def test_get_U_panuzzo(self):
    #    atol = 1e-7
    #    Q = np.array([1e50])
    #    expected = Lagn
    #    vals = unh.get_Q_agn(Lagn,-1.7)
    #    np.testing.assert_allclose(vals,expected,atol=atol)
        
if __name__ == '__main__':
    unittest.main()
