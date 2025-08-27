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
                                     outpath=expath,verbose=False)
        self.assertEqual(filenom,expath+root+'.hdf5')
        
        names = ['a','b',None]
        values = [1,'b',3]
        num = io.add2header(filenom,names,values,verbose=False)

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
        mm,rr = io.read_mgas_hr(hf5file,incols,sel,verbose=False)
        assert_allclose(mm,expect_m,rtol=0.01)  
        assert_allclose(rr,expect_r,rtol=0.01)  

        
    def test_get_mgas_hr(self):
        sel=[0]
        incols = [['data/mgas_disk','data/rhm_disk']]
        expect_m = np.array([[6.69049152e+08]])
        expect_r = np.array([[3.02573503e-03]])
        mm,rr = io.get_mgas_hr(hf5file,sel,incols,[4])
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
        mm,rr = io.get_mgas_hr(hf5file,sel,incols,rtype)
        assert_allclose(mm,expect_m,rtol=0.01)  
        assert_allclose(rr,expect_r,rtol=0.01)  
        
        mm,rr = io.get_mgas_hr(hf5file,sel,incols,[1,2],verbose=False)
        assert_allclose(mm,expect_m,rtol=0.01)
        exp = np.copy(expect_r)
        exp[0,:] = c.re2hr*expect_r[0,:]
        exp[1,:] = c.re2hr*c.r502re*expect_r[1,:]
        assert_allclose(rr,exp,rtol=0.01)  
        
        h=2.
        mm,rr = io.get_mgas_hr(hf5file,sel,incols,[0,3],
                               h0=h,units_h0=True,verbose=False)
        assert_allclose(mm,expect_m/h,rtol=0.01)
        exp = np.copy(expect_r)/h
        exp[1,:] = c.re2hr*c.r502re*c.rvir2r50*exp[1,:]
        assert_allclose(rr,exp,rtol=0.01)  
        
        re2hr=8.; r502re=300.
        mm,rr = io.get_mgas_hr(hf5file,sel,incols,[1,2],
                               re2hr=re2hr,r502re=r502re,verbose=False)
        assert_allclose(mm,expect_m,rtol=0.01)
        exp = np.copy(expect_r)
        exp[0,:] = re2hr*expect_r[0,:]
        exp[1,:] = re2hr*r502re*expect_r[1,:]
        assert_allclose(rr,exp,rtol=0.01)  
        

        def test_read_data(self):
            params_txt = [0, 2] 
            params_hdf5 = ['data/mstar_disk', 'data/SFR_disk']

            sel = [0, 1, 2] # Select 3 first galaxies
            expect_mstar = np.array([2.54168512e8,6.0441748e7,1.0511475712e10])
            expect_sfr = np.array([4.521084e7,1.2074103e7,1.006248e9])

            # Test hdf5 file            
            result_hdf5 = io.read_data(hf5file, sel, inputformat='hdf5', 
                                       params=params_hdf5, verbose=False)
            self.assertEqual(result_hdf5.shape, (2, 3))
            assert_allclose(result_hdf5[0], expect_mstar, rtol=0.01)
            assert_allclose(result_hdf5[1], expect_sfr, rtol=0.01)

            result_single_hdf5 = io.read_data(hf5file, sel, inputformat='hdf5',
                                              params=['data/mstar_disk'], verbose=False)
            self.assertEqual(result_single_hdf5.ndim, 1)

            params_with_none = ['data/mstar_disk', None, 'data/SFR_disk']
            result_none = io.read_data(hf5file, sel, inputformat='hdf5',
                                       params=params_with_none, verbose=False)
            self.assertEqual(result_none.shape, (2, 3))

            with self.assertRaises(SystemExit):
                io.read_data(hf5file, sel, inputformat='invalid', 
                             params=['data/mstar_disk'], verbose=False)
    

            result_missing = io.read_data(hf5file, sel, inputformat='hdf5',
                                          params=['data/nonexistent'], verbose=False)
            assert_allclose(result_missing, np.zeros(3), rtol=0.01)

            empty_sel = np.array([], dtype=int)
            result_empty = io.read_data(hf5file, empty_sel, inputformat='hdf5',
                                        params=['data/mstar_disk'], verbose=False)
            self.assertEqual(len(result_empty), 0)
    
            single_sel = [0]
            result_single = io.read_data(hf5file, single_sel, inputformat='hdf5',
                                         params=['data/mstar_disk'], verbose=False)
            self.assertEqual(len(result_single), 1)

            large_sel = list(range(min(50, 100))) 
            result_large = io.read_data(hf5file, large_sel, inputformat='hdf5',
                                        params=['data/mstar_disk'], verbose=False)
            self.assertEqual(len(result_large), len(large_sel))
            
            # Test text file
            result_txt = io.read_data(txtfile, sel, inputformat='txt', 
                                      params=params_txt, verbose=False)
            self.assertEqual(result_txt.shape, (2, 3))
            assert_allclose(result_txt[0], expect_mstar, rtol=0.01)
            assert_allclose(result_txt[1], expect_sfr, rtol=0.01)

            result_single_txt = io.read_data(txtfile, sel, inputformat='txt',
                                             params=[0], verbose=False)
            self.assertEqual(result_single_txt.ndim, 1)
            
            # Hdf5 and text example files should be the same
            sel_diff = [1, 3, 5]
            assert_allclose(result_hdf5, result_txt, rtol=1e-10, 
                            err_msg="HDF5 and text file results should be identical")
            assert_allclose(result_single_hdf5, result_single_txt, rtol=1e-10)
            result_diff_hdf5 = io.read_data(hf5file, sel_diff, inputformat='hdf5',
                                            params=['data/mstar_disk'], verbose=False)
            result_diff_txt = io.read_data(txtfile, sel_diff, inputformat='txt',
                                           params=[0], verbose=False)
            assert_allclose(result_diff_hdf5, result_diff_txt, rtol=1e-10)
    
            params_ordered = ['data/mstar_disk', 'data/SFR_disk', 'data/Zgas_disk']
            params_txt_ordered = [0, 2, 4]
   
            result_ordered_hdf5 = io.read_data(hf5file, sel_diff, inputformat='hdf5',
                                               params=params_ordered, verbose=False)
            result_ordered_txt = io.read_data(txtfile, sel_diff, inputformat='txt',
                                              params=params_txt_ordered, verbose=False)
            assert_allclose(result_ordered_hdf5, result_ordered_txt, rtol=1e-10,
                            err_msg="Parameter order should be preserved between formats")
            #---------------to remove below
            sel = [0, 1, 2, 3, 4]  # First 5 galaxies
            
            # Test the specific parameter combinations from your tutorial files
            m_sfr_z_hdf5 = [['data/mstar_disk','data/SFR_disk','data/Zgas_disk'],
                            ['data/mstar_stb','data/SFR_bulge','data/Zgas_bulge']]
            m_sfr_z_txt = [[0,2,4],[1,3,5]]
    
            # Read component 0 data
            comp0_hdf5 = io.read_data(hf5file, sel, inputformat='hdf5',
                                      params=m_sfr_z_hdf5[0], verbose=False)
            comp0_txt = io.read_data(txtfile, sel, inputformat='txt',
                                     params=m_sfr_z_txt[0], verbose=False)
    
            # Read component 1 data  
            comp1_hdf5 = io.read_data(hf5file, sel, inputformat='hdf5',
                                      params=m_sfr_z_hdf5[1], verbose=False)
            comp1_txt = io.read_data(txtfile, sel, inputformat='txt',
                                     params=m_sfr_z_txt[1], verbose=False)
    
            # These should be identical
            assert_allclose(comp0_hdf5, comp0_txt, rtol=1e-12,
                            err_msg="Component 0 data differs between formats")
            assert_allclose(comp1_hdf5, comp1_txt, rtol=1e-12,
                            err_msg="Component 1 data differs between formats")
    
            # Test Lagn parameters specifically
            Lagn_params_hdf5 = ['data/lagn', 'data/mstar_stb'] 
            Lagn_params_txt = [17, 1]
            
            lagn_hdf5 = io.read_data(hf5file, sel, inputformat='hdf5',
                                     params=Lagn_params_hdf5, verbose=False)
            lagn_txt = io.read_data(txtfile, sel, inputformat='txt',
                                    params=Lagn_params_txt, verbose=False)
    
            # This is the critical test for your specific issue
            assert_allclose(lagn_hdf5, lagn_txt, rtol=1e-12,
                            err_msg="Lagn parameters differ between formats - this could be your bug!")
        
if __name__ == '__main__':
    unittest.main()
