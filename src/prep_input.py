"""
.. moduleauthor:: Violeta Gonzalez-Perez <violetagp@protonmail.com>

Stand alone module to prepare input files from hdf5 files
"""
import h5py
import sys
#import os
import numpy as np
#import src.gne_const as c

GP20runs = True
if GP20runs:
    # Path to files
    root = '/home/violeta/buds/emlines/gp19_iz39_ivol'
    subvols = list(range(1))

    # Cosmology and volume of the simulation
    h0     = 0.704
    omega0 = 0.307
    omegab = 0.0482
    lambda0= 0.693
    boxside= 500. #Mpc/h
    mp     = 9.35e8 #Msun/h
    # Cut in halo mass
    cut_prop = 'mhhalo' # Ensure this is read first
    cut_val = 20*mp
    
    # Define the files and their corresponding properties
    file_props = {
        'galaxies.hdf5': {
            'group': 'Output001',
            'datasets': ['mhhalo','redshift','index','type',
                         'xgal','ygal','zgal','rbulge','rcomb','rdisk','mhot','vbulge',
                         'mcold','mcold_burst','cold_metal','metals_burst',
                         'mstars_bulge','mstars_burst','mstars_disk',
                         'mstardot','mstardot_burst','mstardot_average',
                         'M_SMBH','SMBH_Mdot_hh','SMBH_Mdot_stb','SMBH_Spin'],
            'units': ['Msun/h','redshift','Host halo index','Gal. type (central=0)',
                      'Mpc/h','Mpc/h','Mpc/h','Mpc/h','Mpc/h','Mpc/h','Msun/h','km/s',
                      'Msun/h','Msun/h','Msun/h','Msun/h','Msun/h','Msun/h','Msun/h',
                      'Msun/h/Gyr','Msun/h/Gyr','Msun/h/Gyr',
                      'Msun/h','Msun/h/Gyr','Msun/h/Gyr','Spin']
        },
        'agn.hdf5': {
            'group': 'Output001', 
            'datasets': ['Lbol_AGN'],
            'units': ['1e40 h^-2 erg/s']
        },
        'tosedfit.hdf5': {
            'group': 'Output001',
            'datasets': ['mag_UKIRT-K_o_tot_ext', 'mag_SDSSz0.1-r_o_tot_ext'],
            'units': ['AB','AB']
        }
    }
   
# Loop over each subvolume
for ivol in subvols:
    path = root+str(subvols[ivol])+'/'

    # Generate a header for the output file
    outfile = path+'gne_input.hdf5'
    print(f' * Generating file: {outfile}')
    
    hf = h5py.File(outfile, 'w')
    headnom = 'header'
    head = hf.create_dataset(headnom,())
    head.attrs[u'h0'] = h0
    head.attrs[u'omega0'] = omega0
    head.attrs[u'omegab'] = omegab
    head.attrs[u'lambda0'] = lambda0
    head.attrs[u'bside_Mpch'] = boxside
    head.attrs[u'mp_Msunh'] = mp
    data_group = hf.create_group('data')
    hf.close()
    
    # Loop over files
    count_props = -1 
    for ifile, props  in file_props.items():
        filename = path+ifile
        try:
            with h5py.File(filename, 'r') as hdf_file:
                # Access the specified dataset
                group = props['group']
                if group is None:
                    hf = hdf_file
                elif group in hdf_file:
                    hf = hdf_file[group]
                else:
                    continue

                # Check that datasets are in the file
                datasets = props['datasets']
                allprops = set(datasets).issubset(list(hf.keys()))
                if not allprops:
                    missing = list(set(datasets).difference(list(hf.keys())))
                    print(f"  STOP: {missing} properties not found in {filename}")
                    sys.exit(1)

                # Check that metallicities need to be calculated
                calc_Zdisc = set(['mcold','cold_metal']).issubset(datasets)
                calc_Zbst  = set(['mcold_burst','metals_burst']).issubset(datasets)
                    
                # Extract properties
                for ii,prop in enumerate(datasets):
                    count_props += 1
                    if prop=='redshift':
                        zz = hf[prop]
                        with h5py.File(outfile, 'a') as outf:
                            outf['header'].attrs['redshift'] = zz
                    elif count_props==0:
                        if cut_prop is None:
                            vals = hf[prop][:]
                            indices = np.arange(len(vals), dtype=int)
                        else:
                            if prop!=cut_prop:
                                print(f'STOP: {cut_prop} needs to be read first')
                                sys.exit(1)
                            else:
                                cprop = hf[cut_prop][:]
                                mask = np.where(cprop > cut_val)
                                if np.shape(mask)[1] > 0:
                                    vals = cprop[mask]
                                    indices = np.arange(len(cprop))[mask]
                                else:
                                    continue
####here: define Zdisc, Zburst in the first pass, and only write indices, then write outside if
                            if calc_Zdisc:
                                Zdisc = np.zeros(len(vals), dtype=float); Zdisc.fill(1.)
                            if calc_burst:
                                Zdisc = np.zeros(len(vals), dtype=float); Zdisc.fill(1.)

                                
                        with h5py.File(outfile, 'a') as outf:
                            ids = outf['data'].create_dataset('gal_index', data=indices)
                            ids.attrs['units'] = 'Position in original file'
                                    
                            dd = outf['data'].create_dataset(prop, data=vals)
                            dd.attrs['units'] = props['units'][ii]
                    else:
                        vals = None
                        if cut_prop is not None and np.shape(mask)[1] > 0:
                            vals = hf[prop][mask]
                        elif cut_prop is None:
                            vals = hf[prop][:]                            
                        if vals is None: continue
#
#                calc_Zdisc = set(['mcold','cold_metal']).issubset(datasets)
#                calc_Zbst  = set(['mcold_burst','metals_burst']).issubset(datasets)
#
#                        if calc_Zdisc and prop=='
                        with h5py.File(outfile, 'a') as outf:
                            dd = outf['data'].create_dataset(prop, data=vals)
                            dd.attrs['units'] = props['units'][ii]
                            
        except FileNotFoundError:
            print(f"  Warning: {filename} could not be opened")
