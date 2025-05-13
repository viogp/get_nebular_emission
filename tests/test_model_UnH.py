#python3 -m unittest tests/test_model_UnH.py 

from unittest import TestCase

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
        

if __name__ == '__main__':
    unittest.main()
