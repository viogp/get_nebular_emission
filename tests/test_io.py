#python3 -m unittest tests/test_io.py 

import shutil
import unittest
import numpy as np
import h5py
from numpy.testing import assert_allclose

import src.gne_io as io
import src.gne_const as c

opath = 'data/example_data/iz61/'
root = 'GP20_31p25kpc_z0_example_vol0'
txtfile = opath+root+'.txt'
hf5file = opath+root+'.hdf5'

class TestStringMethods(unittest.TestCase):

    def test_outroot(self):
        expath = 'output/iz100/'
        self.assertEqual(io.get_outroot('test/example.txt',100),
                         expath+'example')
        shutil.rmtree(expath)
        
        expath = 'output/test'
        self.assertEqual(io.get_outroot('example.txt',32,outpath=expath),
                         expath+'/iz32/example')
        shutil.rmtree(expath)

    def test_plotpath(self):
        expath = 'plots/'
        self.assertEqual(io.get_plotpath('root'),expath)
        shutil.rmtree(expath)
        expath = 'output/test/'
        self.assertEqual(io.get_plotpath('output/test/root'),expath+'plots/')
        self.assertEqual(io.get_plotpath('output/test/'),expath+'plots/')
        shutil.rmtree(expath)

        
    def test_outnom(self):
        snap = 1
        expath = 'output/iz'+str(snap)+'/'
        self.assertEqual(io.get_outnom(hf5file,snap),expath+root+'.hdf5')
        ft = 'plots'; pt='test'
        self.assertEqual(io.get_outnom(hf5file,snap,ftype=ft,ptype=pt),
                         expath+ft+'/'+pt+'_'+root+'.pdf')
        shutil.rmtree(expath)
        
        expath = 'output/test/'
        self.assertEqual(io.get_outnom(hf5file,snap,dirf=expath),
                         expath+root+'.hdf5')
        shutil.rmtree(expath)
        

    def test_hdf5_headers(self):
        expath = 'output/test/'
        h0 = 0.7
        filenom = io.generate_header(hf5file,0,100,h0,0.4,0.3,0.6,1,1e8,
                                  outpath=expath,verbose=True)
        self.assertEqual(filenom,expath+root+'.hdf5')
        
        names = ['a','b',None]
        values = [1,'b',3]
        num = io.add2header(filenom,names,values)

        hf = h5py.File(filenom, 'r')
        self.assertEqual(hf['header'].attrs['h0'], h0)
        self.assertEqual(hf['header'].attrs[names[0]], values[0])
        self.assertEqual(hf['header'].attrs[names[1]], values[1])
        self.assertEqual(num,2)
        hf.close()        
        shutil.rmtree(expath)


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
        mm,rr = io.get_mgas_hr(hf5file,incols,[4],sel)
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
        exp[0,:] = c.re2hr*expect_r[0,:]
        exp[1,:] = c.re2hr*c.r502re*expect_r[1,:]
        assert_allclose(rr,exp,rtol=0.01)  
        
        h=2.
        mm,rr = io.get_mgas_hr(hf5file,incols,[0,3],sel,
                               h0=h,units_h0=True)
        assert_allclose(mm,expect_m/h,rtol=0.01)
        exp = np.copy(expect_r)/h
        exp[1,:] = c.re2hr*c.r502re*c.rvir2r50*exp[1,:]
        assert_allclose(rr,exp,rtol=0.01)  

        re2hr=8.; r502re=300.
        mm,rr = io.get_mgas_hr(hf5file,incols,[1,2],sel,
                               re2hr=re2hr,r502re=r502re)
        assert_allclose(mm,expect_m,rtol=0.01)
        exp = np.copy(expect_r)
        exp[0,:] = re2hr*expect_r[0,:]
        exp[1,:] = re2hr*r502re*expect_r[1,:]
        assert_allclose(rr,exp,rtol=0.01)  
        
        
if __name__ == '__main__':
    unittest.main()
