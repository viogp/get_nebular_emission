#python -m unittest tests/test_flux.py 

from unittest import TestCase
import numpy as np
from numpy.testing import assert_allclose
from unittest.mock import patch

import src.gne_cosmology as cosmo
import src.gne_flux as fx

class TestPredict(TestCase):
    def test_L2flux(self):
        """Set up cosmology for tests."""
        cosmo.set_cosmology(omega0=0.3, omegab=0.045, lambda0=0.7, h0=0.7)
        expect = 1

        # Test single value
        luminosity = np.array([1e42])  # erg/s
        zz = 0.1
        flux = fx.L2flux(luminosity, zz)
        self.assertGreater(flux[0], 0)
        d_L = cosmo.luminosity_distance(zz, cm=True)
        expected_flux = luminosity[0]/(4*np.pi*d_L**2)
        assert_allclose(flux[0], expected_flux, rtol=1e-10)

        # Test zz=0
        flux = fx.L2flux(luminosity, 0.)
        self.assertGreater(flux[0], 0)
                
        # Test array
        luminosity = np.array([1e40, 1e42, 1e44, 1e43])
        zz = 0.5
        flux = fx.L2flux(luminosity, zz)        
        self.assertEqual(flux.shape, luminosity.shape)
        self.assertGreater(flux[0], 0)
        self.assertGreater(flux[1], 0)
        self.assertGreater(flux[2], 0)
        self.assertGreater(flux[3], 0)
        
        # Test zeros
        with patch('builtins.print') as mock_print:
            flux = fx.L2flux(np.array([0.,-1.]), zz)        
            self.assertEqual(flux[0], 0)
            self.assertEqual(flux[1], 0)
        mock_print.assert_called_once()
        self.assertIn('WARNING', mock_print.call_args[0][0])

if __name__ == '__main__':
    unittest.main()
