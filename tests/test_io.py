#python -m unittest tests/test_io.py 

import shutil
import unittest
import numpy as np
import h5py
from numpy.testing import assert_allclose
import numpy as np
import h5py
import src.gne_io as io
import src.gne_const as c

opath = 'data/example_data/iz61/ivol0/'
nom = 'ex'
txtfile = opath+nom+'.txt'
hf5file = opath+nom+'.hdf5'

class TestStringMethods(unittest.TestCase):

    def test_outroot(self):
        expath = 'output/iz61/'
        dirf, endf = io.get_outroot(expath,'example.txt')
        self.assertEqual(dirf,expath)
        self.assertEqual(endf,'example.hdf5')
        
    def test_plotfile(self):
        expath = 'output/iz61/ivol0/'
        dirf = io.get_plotfile(expath,nom,'uuu')
        self.assertEqual(dirf,'output/iz61/plots/uuu_ex.pdf')

    def test_outnom(self):
        outf = io.get_outnom(txtfile,dirf=None,verbose=False)
        self.assertEqual(outf,'output/iz61/ivol0/ex.hdf5')

        expath = 'uuu'
        outf = io.get_outnom(txtfile,dirf=expath,verbose=False)
        expected = expath+'/iz61/ivol0/ex.hdf5'
        self.assertEqual(outf,expected)        
        shutil.rmtree(expath)

    def test_get_param(self):
        att_config = {'albedo': 0, 'costheta': None}

        result = io.get_param(att_config, 'albedo', c.albedo)
        self.assertEqual(result, 0)

        result = io.get_param(att_config, 'costheta', c.albedo)
        self.assertEqual(result, c.albedo)
        
        att_config = {}
        result = io.get_param(att_config, 'Rv', c.Rv)
        self.assertEqual(result, c.Rv)

        att_config = None
        result = io.get_param(att_config, 'Rv', c.Rv)
        self.assertEqual(result, c.Rv)

        
    def test_hdf5_headers(self):
        expath = 'output/test'
        h0 = 0.7
        filenom = io.generate_header(hf5file,0,100,h0,0.4,0.3,0.6,1,1e8,
                                     outpath=expath,verbose=False)
        self.assertEqual(filenom,expath+'/iz61/ivol0/'+nom+'.hdf5')
        
        names = ['a','b',None]
        values = [1,'b',3]
        num = io.add2header(filenom,names,values,verbose=False)
        self.assertEqual(num,2)
        hf = h5py.File(filenom, 'r')
        self.assertEqual(hf['header'].attrs['h0'], h0)
        self.assertEqual(hf['header'].attrs[names[0]], values[0])
        self.assertEqual(hf['header'].attrs[names[1]], values[1])
        hf.close()        
        shutil.rmtree(expath)


#    def test_read_mgas_hr(self):
#        vb = False
#        sel=[0,1]
#        incols = [[6, 11]]
#        expect_m = np.array([[6.69049152e+08,2.09834368e+08]])
#        expect_r = np.array([[0.00302574,0.0017807]])
#        mm,rr = io.read_mgas_hr(txtfile,incols,sel,
#                                inputformat='txt',verbose=vb)
#        assert_allclose(mm,expect_m,rtol=0.01)  
#        assert_allclose(rr,expect_r,rtol=0.01)  
#
#        incols = [['data/mgas_disk','data/rhm_disk'],
#                  ['data/mgas_bulge','data/rhm_bulge']]
#        incols_txt = [[6,11],[9,12]]
#        expect_m = np.array([[6.69049152e+08,2.09834368e+08],
#                             [4.87330867e+09,3.95953331e+09]])
#        expect_r = np.array([[0.00302574,0.0017807],[0,0]])
#        mm,rr = io.read_mgas_hr(hf5file,incols,sel,verbose=vb)
#        assert_allclose(mm,expect_m,rtol=0.01)  
#        assert_allclose(rr,expect_r,rtol=0.01)  
#
#        m_t,r_t = io.read_mgas_hr(txtfile,incols_txt,sel,
#                                  inputformat='txt',verbose=vb)
#        assert_allclose(m_t,mm,rtol=0.01)  
#        assert_allclose(r_t,rr,rtol=0.01)  
#        
#        
#    def test_get_mgas_hr(self):
#        vb = False
#        sel=[0]
#        incols = [['data/mgas_disk','data/rhm_disk']]
#        expect_m = np.array([[6.69049152e+08]])
#        expect_r = np.array([[0.00302574]])
#        mm,rr = io.get_mgas_hr(hf5file,sel,incols,[4],verbose=vb)
#        assert_allclose(mm,expect_m,rtol=0.01)
#        assert_allclose(rr,expect_r,rtol=0.01)
#
#        sel=[0,1,3]
#        rtype=[0,0]
#        incols = [['data/mgas_disk','data/rhm_disk'],
#                  ['data/mgas_bulge','data/rhm_bulge']]
#
#        expect_m = np.array([[6.69049152e+08,2.09834368e+08,3.23166387e+09],
#                             [4.87330867e+09,3.959533e+09,3.07952046e+10]])
#        expect_r = np.array([[3.02573503e-03,0.0017807,0.00372818],
#                             [0,0,0.00125381]])
#        mm,rr = io.get_mgas_hr(hf5file,sel,incols,rtype,verbose=vb)
#        assert_allclose(mm,expect_m,rtol=0.01)  
#        assert_allclose(rr,expect_r,rtol=0.01)
#        
#        mm,rr = io.get_mgas_hr(hf5file,sel,incols,[1,2],verbose=vb)
#        assert_allclose(mm,expect_m,rtol=0.01)
#        exp = np.copy(expect_r)
#        exp[0,:] = c.re2hr*expect_r[0,:]
#        exp[1,:] = c.re2hr*c.r502re*expect_r[1,:]
#        assert_allclose(rr,exp,rtol=0.01)  
#        
#        h=2.
#        mm,rr = io.get_mgas_hr(hf5file,sel,incols,[0,3],
#                               h0=h,units_h0=True,verbose=vb)
#        assert_allclose(mm,expect_m/h,rtol=0.01)
#        exp = np.copy(expect_r)/h
#        exp[1,:] = c.re2hr*c.r502re*c.rvir2r50*exp[1,:]
#        assert_allclose(rr,exp,rtol=0.01)  
#        
#        re2hr=8.; r502re=300.
#        mm,rr = io.get_mgas_hr(hf5file,sel,incols,[1,2],
#                               re2hr=re2hr,r502re=r502re,verbose=vb)
#        assert_allclose(mm,expect_m,rtol=0.01)
#        exp = np.copy(expect_r)
#        exp[0,:] = re2hr*expect_r[0,:]
#        exp[1,:] = re2hr*r502re*expect_r[1,:]
#        assert_allclose(rr,exp,rtol=0.01)  
#        
#
#    def test_read_data(self):
#        params_txt = [0, 2] 
#        params_hdf5 = ['data/mstar_disk', 'data/SFR_disk']
#
#        sel = [0, 1, 2] # Select 3 first galaxies
#        expect_mstar = np.array([2.54168512e8,6.0441748e7,1.0511475712e10])
#        expect_sfr = np.array([4.521084e7,1.2074103e7,1.006248e9])
#
#        # Test hdf5 file            
#        result_hdf5 = io.read_data(hf5file, sel, inputformat='hdf5', 
#                                   params=params_hdf5, verbose=False)
#        self.assertEqual(result_hdf5.shape, (2, 3))
#        assert_allclose(result_hdf5[0], expect_mstar, rtol=0.01)
#        assert_allclose(result_hdf5[1], expect_sfr, rtol=0.01)
#
#        result_single_hdf5 = io.read_data(hf5file, sel, inputformat='hdf5',
#                                          params=['data/mstar_disk'], verbose=False)
#        self.assertEqual(result_single_hdf5.ndim, 1)
#
#        params_with_none = ['data/mstar_disk', None, 'data/SFR_disk']
#        result_none = io.read_data(hf5file, sel, inputformat='hdf5',
#                                   params=params_with_none, verbose=False)
#        self.assertEqual(result_none.shape, (2, 3))
#
#        with self.assertRaises(SystemExit):
#            io.read_data(hf5file, sel, inputformat='invalid', 
#                         params=['data/mstar_disk'], verbose=False)
#    
#
#        result_missing = io.read_data(hf5file, sel, inputformat='hdf5',
#                                      params=['data/nonexistent'], verbose=False)
#        assert_allclose(result_missing, np.zeros(3), rtol=0.01)
#
#        empty_sel = np.array([], dtype=int)
#        result_empty = io.read_data(hf5file, empty_sel, inputformat='hdf5',
#                                    params=['data/mstar_disk'], verbose=False)
#        self.assertEqual(len(result_empty), 0)
#    
#        single_sel = [0]
#        result_single = io.read_data(hf5file, single_sel, inputformat='hdf5',
#                                     params=['data/mstar_disk'], verbose=False)
#        self.assertEqual(len(result_single), 1)
#
#        large_sel = list(range(min(50, 100))) 
#        result_large = io.read_data(hf5file, large_sel, inputformat='hdf5',
#                                    params=['data/mstar_disk'], verbose=False)
#        self.assertEqual(len(result_large), len(large_sel))
#        
#        # Test text file
#        result_txt = io.read_data(txtfile, sel, inputformat='txt', 
#                                  params=params_txt, verbose=False)
#        self.assertEqual(result_txt.shape, (2, 3))
#        assert_allclose(result_txt[0], expect_mstar, rtol=0.01)
#        assert_allclose(result_txt[1], expect_sfr, rtol=0.01)
#
#        result_single_txt = io.read_data(txtfile, sel, inputformat='txt',
#                                         params=[0], verbose=False)
#        self.assertEqual(result_single_txt.ndim, 1)
#        
#        # HDF5 and text example files should be the same
#        sel_diff = [1, 3, 5]
#        assert_allclose(result_hdf5, result_txt, rtol=1e-10, 
#                        err_msg="HDF5 and text file results should be identical")
#        assert_allclose(result_single_hdf5, result_single_txt, rtol=1e-10)
#        result_diff_hdf5 = io.read_data(hf5file, sel_diff, inputformat='hdf5',
#                                        params=['data/mstar_disk'], verbose=False)
#        result_diff_txt = io.read_data(txtfile, sel_diff, inputformat='txt',
#                                       params=[0], verbose=False)
#        assert_allclose(result_diff_hdf5, result_diff_txt, rtol=1e-10)
#    
#        params_ordered = ['data/mstar_disk', 'data/SFR_disk', 'data/Zgas_disk']
#        params_txt_ordered = [0, 2, 4]
#   
#        result_ordered_hdf5 = io.read_data(hf5file, sel_diff, inputformat='hdf5',
#                                           params=params_ordered, verbose=False)
#        result_ordered_txt = io.read_data(txtfile, sel_diff, inputformat='txt',
#                                          params=params_txt_ordered, verbose=False)
#        assert_allclose(result_ordered_hdf5, result_ordered_txt, rtol=1e-10,
#                        err_msg="Parameter order should be preserved between formats")
#
#
#    def test_read_sfrdata(self):
#        sel = np.array([0, 1, 2, 3, 4])  # First 5 galaxies
#        cols_hdf5 = [['data/mstar_disk','data/SFR_disk','data/Zgas_disk'],
#                     ['data/mstar_stb','data/SFR_bulge','data/Zgas_bulge']]
#        cols_txt = [[0,2,4],[1,3,5]]
#        
#        cols_single_hdf5 = [['data/mstar_disk','data/SFR_disk','data/Zgas_disk']]
#        cols_single_txt = [[0,2,4]]
#        
#        sel_reverse = np.array([1, 0])
#        
#        # HDF5 file
#        ms_hdf5, sfr_hdf5, z_hdf5 = io.read_sfrdata(hf5file, cols_hdf5, sel,
#                                                    inputformat='hdf5', verbose=False)
#        self.assertEqual(ms_hdf5.shape, (2, 5))
#        self.assertEqual(sfr_hdf5.shape, (2, 5))
#        self.assertEqual(z_hdf5.shape, (2, 5))
#    
#        ms_hdf5_rev, sfr_hdf5_rev, z_hdf5_rev = io.read_sfrdata(
#            hf5file, cols_hdf5, sel_reverse, inputformat='hdf5', verbose=False)
#        assert_allclose(ms_hdf5[:, 0], ms_hdf5_rev[:, 1], rtol=1e-12)
#        assert_allclose(ms_hdf5[:, 1], ms_hdf5_rev[:, 0], rtol=1e-12)
#            
#        ms_single_hdf5, sfr_single_hdf5, z_single_hdf5 = io.read_sfrdata(
#            hf5file, cols_single_hdf5, sel, inputformat='hdf5', verbose=False)
#        self.assertEqual(ms_single_hdf5.shape, (1, 5))
#    
#        with self.assertRaises(SystemExit): # Test invalid input
#            io.read_sfrdata(hf5file, cols_hdf5, np.array([0]),
#                            inputformat='invalid', verbose=False)
#            
#        empty_sel = np.array([], dtype=int) # Test empty selection
#        ms_empty, sfr_empty, z_empty = io.read_sfrdata(
#            hf5file, cols_hdf5, empty_sel, inputformat='hdf5', verbose=False)
#        self.assertEqual(ms_empty.shape[1], 0)
#        self.assertEqual(sfr_empty.shape[1], 0) 
#        self.assertEqual(z_empty.shape[1], 0)
#    
#        # Text file
#        ms_txt, sfr_txt, z_txt = io.read_sfrdata(txtfile, cols_txt, sel,
#                                                 inputformat='txt', verbose=False)
#        self.assertEqual(ms_txt.shape, (2, 5))
#        self.assertEqual(sfr_txt.shape, (2, 5))
#        self.assertEqual(z_txt.shape, (2, 5))
#    
#        ms_txt_rev, sfr_txt_rev, z_txt_rev = io.read_sfrdata(
#            txtfile, cols_txt, sel_reverse, inputformat='txt', verbose=False)
#        assert_allclose(ms_txt[:, 0], ms_txt_rev[:, 1], rtol=1e-12)
#        assert_allclose(ms_txt[:, 1], ms_txt_rev[:, 0], rtol=1e-12)
#    
#        ms_single_txt, sfr_single_txt, z_single_txt = io.read_sfrdata(
#            txtfile, cols_single_txt, sel, inputformat='txt', verbose=False)
#        self.assertEqual(ms_single_txt.shape, (1, 5))
#        
#        # HDF5 and text example files should be the same
#        assert_allclose(ms_hdf5, ms_txt, rtol=1e-12,
#                        err_msg="Stellar mass differs between formats")
#        assert_allclose(sfr_hdf5, sfr_txt, rtol=1e-12,
#                        err_msg="SFR differs between formats")
#        assert_allclose(z_hdf5, z_txt, rtol=1e-12,
#                        err_msg="Metallicity differs between formats")
#    
#        assert_allclose(ms_single_hdf5, ms_single_txt, rtol=1e-12)
#        assert_allclose(ms_single_hdf5[0,:], ms_hdf5[0,:], rtol=1e-12,
#                        err_msg="Single component should match first one")
#    
#        assert_allclose(ms_hdf5_rev, ms_txt_rev, rtol=1e-12,
#                        err_msg="Reverse selection differs between formats")
#
        
if __name__ == '__main__':
    unittest.main()
