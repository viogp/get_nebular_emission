# python -m unittest tests/test_flux.py 

import unittest
import numpy as np
from numpy.testing import assert_allclose
from unittest.mock import patch

import gne.gne_cosmology as cosmo
import gne.gne_flux as fx

class TestPredict(unittest.TestCase):
    def test_L2flux(self):
        """Set up cosmology for tests."""
        cosmo.set_cosmology(omega0=0.3, omegab=0.045, lambda0=0.7, h0=0.7)
        expect = 1

        # Test single value
        luminosity = 1e42 # erg/s
        zz = 0.1
        flux = fx.L2flux(luminosity, zz)
        self.assertGreater(flux, 0)
        d_L = cosmo.luminosity_distance(zz, cm=True)
        expected_flux = luminosity/(4*np.pi*d_L**2)
        assert_allclose(flux, expected_flux, rtol=1e-10)

        # Test zz=0
        flux = fx.L2flux(luminosity, 0.)
        self.assertGreater(flux, 0)
                
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

        
    def test_flux2L(self):
        """Test flux to luminosity conversion."""
        cosmo.set_cosmology(omega0=0.3, omegab=0.045, lambda0=0.7, h0=0.7)

        # Test single value
        flux = 1e-14  # erg/s/cm^2
        zz = 0.1
        lum = fx.flux2L(flux, zz)
        self.assertGreater(lum, 0)
        d_L = cosmo.luminosity_distance(zz, cm=True)
        expected_lum = flux*(4*np.pi*d_L**2)
        assert_allclose(lum, expected_lum, rtol=1e-10)

        # Test zz=0
        lum = fx.flux2L(flux, 0.)
        self.assertGreater(lum, 0)

        # Test array
        flux = np.array([1e-16, 1e-14, 1e-12, 1e-13])
        zz = 0.5
        lum = fx.flux2L(flux, zz)
        self.assertEqual(lum.shape, flux.shape)
        self.assertGreater(lum[0], 0)
        self.assertGreater(lum[1], 0)
        self.assertGreater(lum[2], 0)
        self.assertGreater(lum[3], 0)

        # Test zeros and negative values
        with patch('builtins.print') as mock_print:
            lum = fx.flux2L(np.array([0., -1.]), zz)
            self.assertEqual(lum[0], 0)
            self.assertEqual(lum[1], 0)
        mock_print.assert_called_once()
        self.assertIn('WARNING', mock_print.call_args[0][0])

        # Test round-trip consistency: L -> flux -> L
        original_lum = np.array([1e40, 1e42, 1e44])
        zz = 0.3
        flux_intermediate = fx.L2flux(original_lum, zz)
        recovered_lum = fx.flux2L(flux_intermediate, zz)
        assert_allclose(recovered_lum, original_lum, rtol=1e-10)


if __name__ == '__main__':
    unittest.main()
