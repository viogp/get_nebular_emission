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
            
            # HDF5 and text example files should be the same
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


            def test_read_sfrdata(self):
                sel = np.array([0, 1, 2, 3, 4])  # First 5 galaxies
                cols_hdf5 = [['data/mstar_disk','data/SFR_disk','data/Zgas_disk'],
                             ['data/mstar_stb','data/SFR_bulge','data/Zgas_bulge']]
                cols_txt = [[0,2,4],[1,3,5]]

                # HDF5 file
                ms_hdf5, sfr_hdf5, z_hdf5 = io.read_sfrdata(hf5file, cols_hdf5, sel,
                                                            inputformat='hdf5', verbose=False)
                self.assertEqual(ms_hdf5.shape, (2, 5))
                self.assertEqual(sfr_hdf5.shape, (2, 5))
                self.assertEqual(z_hdf5.shape, (2, 5))

                # Text file
                ms_txt, sfr_txt, z_txt = io.read_sfrdata(txtfile, cols_txt, sel,
                                                         inputformat='txt', verbose=False)
                self.assertEqual(ms_txt.shape, (2, 5))
                self.assertEqual(sfr_txt.shape, (2, 5))
                self.assertEqual(z_txt.shape, (2, 5))

                # HDF5 and text example files should be the same
                assert_allclose(ms_hdf5, ms_txt, rtol=1e-12,
                                err_msg="Stellar mass differs between formats")
                assert_allclose(sfr_hdf5, sfr_txt, rtol=1e-12,
                                err_msg="SFR differs between formats")
                assert_allclose(z_hdf5, z_txt, rtol=1e-12,
                                err_msg="Metallicity differs between formats")
    
#    # Test component ordering - component 0 should be disk, component 1 should be bulge
#    # Verify that ms_hdf5[0,:] corresponds to disk masses, ms_hdf5[1,:] to bulge masses
#    
#    # Test single component
#    cols_single_hdf5 = [['data/mstar_disk','data/SFR_disk','data/Zgas_disk']]
#    cols_single_txt = [[0,2,4]]
#    
#    ms_single_hdf5, sfr_single_hdf5, z_single_hdf5 = io.read_sfrdata(
#        hf5file, cols_single_hdf5, sel, inputformat='hdf5', verbose=False)
#    ms_single_txt, sfr_single_txt, z_single_txt = io.read_sfrdata(
#        txtfile, cols_single_txt, sel, inputformat='txt', verbose=False)
#    
#    # Single component should have shape (1, 5)
#    self.assertEqual(ms_single_hdf5.shape, (1, 5))
#    assert_allclose(ms_single_hdf5, ms_single_txt, rtol=1e-12)
#    
#    # Verify that single component matches first component of multi-component
#    assert_allclose(ms_single_hdf5[0,:], ms_hdf5[0,:], rtol=1e-12,
#                   err_msg="Single component should match first component of multi-component")
#
#
#def test_read_sfrdata_data_integrity(self):
#    """Test that read_sfrdata preserves data integrity and handles edge cases"""
#    
#    sel = np.array([0, 1])
#    cols_hdf5 = [['data/mstar_disk','data/SFR_disk','data/Zgas_disk'],
#                 ['data/mstar_stb','data/SFR_bulge','data/Zgas_bulge']]
#    cols_txt = [[0,2,4],[1,3,5]]
#    
#    # Test with different selection orders
#    sel_reverse = np.array([1, 0])
#    
#    ms_hdf5_fwd, sfr_hdf5_fwd, z_hdf5_fwd = io.read_sfrdata(
#        hf5file, cols_hdf5, sel, inputformat='hdf5', verbose=False)
#    ms_hdf5_rev, sfr_hdf5_rev, z_hdf5_rev = io.read_sfrdata(
#        hf5file, cols_hdf5, sel_reverse, inputformat='hdf5', verbose=False)
#    
#    # Forward selection [0,1] should give reversed results compared to [1,0]
#    assert_allclose(ms_hdf5_fwd[:, 0], ms_hdf5_rev[:, 1], rtol=1e-12)
#    assert_allclose(ms_hdf5_fwd[:, 1], ms_hdf5_rev[:, 0], rtol=1e-12)
#    
#    # Test with text format for same selection patterns
#    ms_txt_fwd, sfr_txt_fwd, z_txt_fwd = io.read_sfrdata(
#        txtfile, cols_txt, sel, inputformat='txt', verbose=False)
#    ms_txt_rev, sfr_txt_rev, z_txt_rev = io.read_sfrdata(
#        txtfile, cols_txt, sel_reverse, inputformat='txt', verbose=False)
#    
#    # Same test for text format
#    assert_allclose(ms_txt_fwd[:, 0], ms_txt_rev[:, 1], rtol=1e-12)
#    assert_allclose(ms_txt_fwd[:, 1], ms_txt_rev[:, 0], rtol=1e-12)
#    
#    # Most importantly: forward selections should match between formats
#    assert_allclose(ms_hdf5_fwd, ms_txt_fwd, rtol=1e-12,
#                   err_msg="Forward selection differs between formats")
#    assert_allclose(ms_hdf5_rev, ms_txt_rev, rtol=1e-12,
#                   err_msg="Reverse selection differs between formats")
#
#
#def test_read_sfrdata_array_construction(self):
#    """Test the array construction logic that differs between HDF5 and text"""
#    
#    sel = np.array([0, 2, 4])  # Non-contiguous selection
#    cols_hdf5 = [['data/mstar_disk','data/SFR_disk','data/Zgas_disk'],
#                 ['data/mstar_stb','data/SFR_bulge','data/Zgas_bulge']]
#    cols_txt = [[0,2,4],[1,3,5]]
#    
#    ms_hdf5, sfr_hdf5, z_hdf5 = io.read_sfrdata(hf5file, cols_hdf5, sel,
#                                                inputformat='hdf5', verbose=False)
#    ms_txt, sfr_txt, z_txt = io.read_sfrdata(txtfile, cols_txt, sel,
#                                            inputformat='txt', verbose=False)
#    
#    # Test that the array construction preserves the correct mapping
#    # HDF5 version uses: ms = np.append(ms,[hf[cols[i][0]][:]],axis=0)
#    # Text version uses: ms = np.append(ms,[X[0]],axis=0)
#    
#    # Check each component separately
#    for comp in range(2):
#        assert_allclose(ms_hdf5[comp, :], ms_txt[comp, :], rtol=1e-12,
#                       err_msg=f"Component {comp} stellar mass differs")
#        assert_allclose(sfr_hdf5[comp, :], sfr_txt[comp, :], rtol=1e-12,
#                       err_msg=f"Component {comp} SFR differs")
#        assert_allclose(z_hdf5[comp, :], z_txt[comp, :], rtol=1e-12,
#                       err_msg=f"Component {comp} metallicity differs")
#    
#    # Test that selection is applied correctly in both versions
#    # The final selection should be: outms[i,:] = ms[i,cut]
#    
#    # Verify shapes are consistent
#    expected_shape = (2, len(sel))
#    self.assertEqual(ms_hdf5.shape, expected_shape)
#    self.assertEqual(ms_txt.shape, expected_shape)
#    
#    # Test edge case: single galaxy selection
#    single_sel = np.array([3])
#    ms_single_hdf5, _, _ = io.read_sfrdata(hf5file, cols_hdf5, single_sel,
#                                          inputformat='hdf5', verbose=False)
#    ms_single_txt, _, _ = io.read_sfrdata(txtfile, cols_txt, single_sel,
#                                         inputformat='txt', verbose=False)
#    
#    self.assertEqual(ms_single_hdf5.shape, (2, 1))
#    self.assertEqual(ms_single_txt.shape, (2, 1))
#    assert_allclose(ms_single_hdf5, ms_single_txt, rtol=1e-12)
#
#
#def test_read_sfrdata_specific_columns(self):
#    """Test specific column mappings that might cause issues"""
#    
#    sel = np.array([0, 1])
#    
#    # Test the exact mappings from your tutorial files
#    # According to generateh5data.py:
#    # cols = [0,2,4,1,3,5,6,11,19,12,17,25,27,30,8]
#    # nprop= ['mstar_disk','SFR_disk','Zgas_disk','mstar_stb','SFR_bulge',
#    #         'Zgas_bulge','mgas_disk','rhm_disk','mgas_bulge','rhm_bulge',
#    #         'lagn','magK','magR','type','MBH']
#    
#    # So: col 0->mstar_disk, col 1->mstar_stb, col 2->SFR_disk, etc.
#    
#    cols_hdf5 = [['data/mstar_disk','data/SFR_disk','data/Zgas_disk'],
#                 ['data/mstar_stb','data/SFR_bulge','data/Zgas_bulge']]
#    cols_txt = [[0,2,4],[1,3,5]]
#    
#    # Read the data
#    ms_hdf5, sfr_hdf5, z_hdf5 = io.read_sfrdata(hf5file, cols_hdf5, sel,
#                                                inputformat='hdf5', verbose=False)
#    ms_txt, sfr_txt, z_txt = io.read_sfrdata(txtfile, cols_txt, sel,
#                                            inputformat='txt', verbose=False)
#    
#    # Verify the specific column mappings
#    # Component 0: columns [0,2,4] should map to ['data/mstar_disk','data/SFR_disk','data/Zgas_disk']
#    # Component 1: columns [1,3,5] should map to ['data/mstar_stb','data/SFR_bulge','data/Zgas_bulge']
#    
#    assert_allclose(ms_hdf5[0, :], ms_txt[0, :], rtol=1e-12,
#                   err_msg="Disk stellar mass mapping incorrect")
#    assert_allclose(ms_hdf5[1, :], ms_txt[1, :], rtol=1e-12,
#                   err_msg="Bulge stellar mass mapping incorrect")
#    
#    assert_allclose(sfr_hdf5[0, :], sfr_txt[0, :], rtol=1e-12,
#                   err_msg="Disk SFR mapping incorrect") 
#    assert_allclose(sfr_hdf5[1, :], sfr_txt[1, :], rtol=1e-12,
#                   err_msg="Bulge SFR mapping incorrect")
#    
#    assert_allclose(z_hdf5[0, :], z_txt[0, :], rtol=1e-12,
#                   err_msg="Disk metallicity mapping incorrect")
#    assert_allclose(z_hdf5[1, :], z_txt[1, :], rtol=1e-12,
#                   err_msg="Bulge metallicity mapping incorrect")
#    
#    # Print some debug info if needed
#    if False:  # Set to True for debugging
#        print(f"HDF5 disk mass (first 2): {ms_hdf5[0, :]}")
#        print(f"Text disk mass (first 2): {ms_txt[0, :]}")
#        print(f"HDF5 bulge mass (first 2): {ms_hdf5[1, :]}")
#        print(f"Text bulge mass (first 2): {ms_txt[1, :]}")
#
#
#def test_read_sfrdata_error_handling(self):
#    """Test error handling and edge cases"""
#    
#    sel = np.array([0])
#    
#    # Test with invalid input format
#    cols = [['data/mstar_disk','data/SFR_disk','data/Zgas_disk']]
#    with self.assertRaises(SystemExit):
#        io.read_sfrdata(hf5file, cols, sel, inputformat='invalid', verbose=False)
#    
#    # Test with empty selection
#    empty_sel = np.array([], dtype=int)
#    ms_empty, sfr_empty, z_empty = io.read_sfrdata(
#        hf5file, cols, empty_sel, inputformat='hdf5', verbose=False)
#    
#    self.assertEqual(ms_empty.shape[1], 0)
#    self.assertEqual(sfr_empty.shape[1], 0) 
#    self.assertEqual(z_empty.shape[1], 0)
        
if __name__ == '__main__':
    unittest.main()
