#python3 -m unittest tests/test_io.py 

import shutil
import unittest
from numpy.testing import assert_allclose
import numpy as np
import src.gne_io as io
import src.gne_const as c

txtfile = 'data/example_data/iz61/GP20_31p25kpc_z0_example_vol0.txt'
hf5file = 'data/example_data/iz61/GP20_31p25kpc_z0_example_vol0.hdf5'

class TestStringMethods(unittest.TestCase):
    
    def test_outroot(self):
        expath = 'output/a'
        self.assertEqual(io.get_outroot('a/example.txt',100),
                         'output/iz100/example')
        self.assertEqual(io.get_outroot('example.txt',32,outpath=expath),
                         expath+'/iz32/example')
        shutil.rmtree(expath)

    def test_plotpath(self):
        expath = 'plots/'
        self.assertEqual(io.get_plotpath('root'),expath)
        shutil.rmtree(expath)
        expath = 'output/a/plots/'
        self.assertEqual(io.get_plotpath('output/a/root'),expath)
        self.assertEqual(io.get_plotpath('output/a/'),expath)
        shutil.rmtree(expath)
        
    #def test_outnom(self):
        #self.assertEqual(io.get_outnom('a/example.txt',100),
        #                 'output/iz100/example.hdf5')
        #self.assertEqual(io.get_outnom('example.txt',39,ftype='plots'),
        #                 'output/iz39/plots/bpt_example.pdf')

        
        
    def test_read_mgas_hr(self):
        sel=[0,1]
        incols = [[6, 11]]
        expect_m = np.array([[6.69049152e+08,2.09834368e+08]])
        expect_r = np.array([[3.02573503e-03,0.0017807]])
        mm,rr = io.read_mgas_hr(txtfile,incols,sel,inputformat='txt')
        assert_allclose(mm,expect_m,rtol=0.01)  
        assert_allclose(rr,expect_r,rtol=0.01)  

        incols = [['data/mgas_disk','data/rhm_disk'],
                  ['data/mgas_bulge','data/rhm_bulge']]
        expect_m = np.array([[6.69049152e+08,2.09834368e+08],[0,0]])
        expect_r = np.array([[3.02573503e-03,0.0017807],[0,0]])
        mm,rr = io.read_mgas_hr(hf5file,incols,sel)
        assert_allclose(mm,expect_m,rtol=0.01)  
        assert_allclose(rr,expect_r,rtol=0.01)  

    def test_get_mgas_hr(self):
        sel=[0]
        incols = [['data/mgas_disk','data/rhm_disk']]
        expect_m = np.array([[6.69049152e+08]])
        expect_r = np.array([[3.02573503e-03]])
        mm,rr = io.get_mgas_hr(hf5file,incols,[5],sel)
        assert_allclose(mm,expect_m,rtol=0.01)
        assert_allclose(rr,expect_r,rtol=0.01)

        sel=[0,1,3]
        rtype=[0,0]
        incols = [['data/mgas_disk','data/rhm_disk'],
                  ['data/mgas_bulge','data/rhm_bulge']]
        expect_m = np.array([[6.69049152e+08,2.09834368e+08,3.23166387e+09],
                             [0,0,0]])
        expect_r = np.array([[3.02573503e-03,0.0017807,0.00372818],
                             [0,0,0.00125381]])
        
        mm,rr = io.get_mgas_hr(hf5file,incols,rtype,sel)
        assert_allclose(mm,expect_m,rtol=0.01)  
        assert_allclose(rr,expect_r,rtol=0.01)  

        mm,rr = io.get_mgas_hr(hf5file,incols,[1,2],sel)
        assert_allclose(mm,expect_m,rtol=0.01)
        exp = np.copy(expect_r)
        exp[0,:] = expect_r[0,:]/c.re2hr_exp
        exp[1,:] = expect_r[1,:]/2./c.re2hr_exp
        assert_allclose(rr,exp,rtol=0.01)  

        h=2.
        mm,rr = io.get_mgas_hr(hf5file,incols,rtype,sel,
                               h0=h,units_h0=True)
        assert_allclose(mm,expect_m/h,rtol=0.01)  
        assert_allclose(rr,expect_r/h,rtol=0.01)  


        
if __name__ == '__main__':
    unittest.main()
