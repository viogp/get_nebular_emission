#python -m unittest tests/test_epsilon.py 

from unittest import TestCase
import numpy as np
import src.gne_const as c
import src.gne_epsilon as e

class TestPredict(TestCase):
    atol = 1e-7
    
    def test_surface_density_disc(self):
        R  = np.array([1/c.mega/c.parsec,1.]) 
        MD = np.array([1/c.Msun,0])
        hD = np.array([1/c.mega/c.parsec,1.])
        expected = np.array([np.exp(-1.)/(2*np.pi),0])
        vals = e.surface_density_disc(R,MD,hD)
        np.testing.assert_allclose(vals,expected,atol=self.atol)

        R  = np.array([1/c.mega/c.parsec]) 
        MD = np.array([0,1/c.Msun])
        hD = np.array([1.,1/c.mega/c.parsec])
        expected = np.array([0,np.exp(-1.)/(2*np.pi)])
        vals = e.surface_density_disc(R,MD,hD)
        np.testing.assert_allclose(vals,expected,atol=self.atol)


        
if __name__ == '__main__':
    unittest.main()
