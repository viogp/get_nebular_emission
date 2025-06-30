#python3 -m unittest tests/test_model_UnH.py 

from unittest import TestCase
import numpy as np
import src.gne_const as c
import src.gne_model_UnH as unh

class TestPredict(TestCase):
    decimals = 8
    atol = 1e-7
    
    def test_get_alphaB(self):
        self.assertAlmostEqual(unh.get_alphaB(5000.),
                               4.54e-13,self.decimals)
        self.assertAlmostEqual(unh.get_alphaB(1000.),
                               4.54e-13,self.decimals)
        val = (1.43e-13+2.59e-13)/2.
        self.assertAlmostEqual(unh.get_alphaB(15000.),
                               val,self.decimals)
        self.assertAlmostEqual(unh.get_alphaB(30000.),
                               1.43e-13,self.decimals)  

    def test_get_Q_agn(self):
        Lagn = np.array([0.])
        alpha =c.alpha_NLR_feltre16
        expected = Lagn
        with self.assertRaises(SystemExit):
            unh.get_Q_agn(Lagn,alpha,model_spec="invalid",verbose=False)
        vals = unh.get_Q_agn(Lagn,alpha)
        np.testing.assert_allclose(vals,expected,atol=self.atol)

    #def test_get_U_panuzzo(self):
    #    Q = np.array([1e50])
    #    expected = Lagn
    #    vals = unh.get_Q_agn(Lagn,-1.7)
    #    np.testing.assert_allclose(vals,expected,atol=self.atol)
 
if __name__ == '__main__':
    unittest.main()
