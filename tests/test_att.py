#python -m unittest tests/test_att.py 

from unittest import TestCase
import numpy as np
from numpy.testing import assert_allclose

import src.gne_const as c
import src.gne_att as att

class TestPredict(TestCase):
    def test_get_f_saito20(self):
        expect = 1
        self.assertAlmostEqual(att.get_f_saito20(2.9),expect,2)
        for z in [1,2.8]:
            expect=0.44 + 0.2*z
            self.assertAlmostEqual(att.get_f_saito20(z),expect,2)

if __name__ == '__main__':
    unittest.main()
