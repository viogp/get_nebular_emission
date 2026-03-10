#python -m unittest tests/test_plots.py 

import unittest
import pytest
import numpy as np
from numpy.testing import assert_allclose
import os

import gne.gne_const as c
import gne.gne_plots as plt

ex_root = 'output/iz61/ivol'
ex_end = 'ex.hdf5'
required_file = ex_root+'0/'+ex_end

@pytest.mark.skipif(not os.path.exists(required_file), 
                   reason=f"Test data file not found: {required_file}")

class TestPredict(unittest.TestCase):
    def test_contour2Dsigma(self):
        levels,colors = plt.contour2Dsigma()
        assert_allclose(levels[1:],c.sigma_2Dprobs,rtol=0.01)
        self.assertEqual(colors[0][-1],1.0)

        nl=3
        levels,colors = plt.contour2Dsigma(n_levels=nl)
        assert_allclose(levels[1:],c.sigma_2Dprobs[0:nl],rtol=0.01)
        
    def test_get_obs_bpt(self):
        z=0.1
        valx,valy,obsdata=plt.get_obs_bpt(z,'NII')
        self.assertTrue(obsdata)
        self.assertAlmostEqual(valy[0],
                               np.log10(1.05924e-15/3.26841e-15),places=5)
        self.assertAlmostEqual(valx[0],
                               np.log10(3.95472e-15/1.00695e-14),places=5)
        valx,valy,obsdata=plt.get_obs_bpt(z,'SII')
        self.assertTrue(obsdata)
        self.assertAlmostEqual(valx[0],
                               np.log10(3.08815e-15/1.00695e-14),places=5)
        
        z=1.5
        valx,valy,obsdata=plt.get_obs_bpt(z,'NII')
        self.assertTrue(obsdata)
        self.assertAlmostEqual(valx[0],-0.5654081,places=5)
        self.assertAlmostEqual(valy[0],0.3271674,places=5)        
        valx,valy,obsdata=plt.get_obs_bpt(z,'SII')
        self.assertTrue(obsdata)
        self.assertAlmostEqual(valx[0],-0.4765437,places=5)
        self.assertAlmostEqual(valy[0],0.5962023,places=5)        

        z=2
        valx,valy,obsdata=plt.get_obs_bpt(z,'NII')
        self.assertFalse(obsdata)

                
if __name__ == '__main__':
    unittest.main()
