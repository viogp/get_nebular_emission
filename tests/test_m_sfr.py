#python3 -m unittest tests/test_m_sfr.py 

import unittest
import numpy as np
from numpy.testing import assert_allclose

import src.gne_const as c
import src.gne_m_sfr as msf

class TestPredict(unittest.TestCase):
    def test_get_lm_tot(self):
        # Generate data
        ncomp = 2; lms = np.zeros((3,ncomp))
        lms[0,0]=10;  lms[1,0]=11.2; lms[2,0]=12
        lms[0,1]=10.5;lms[1,1]=11.8; lms[2,1]=c.notnum

        # Test
        expected = np.array([10.619,11.897,12.])
        vals = msf.get_lm_tot(lms)
        np.testing.assert_allclose(vals,expected, atol=0.001)
        
if __name__ == '__main__':
    unittest.main()
