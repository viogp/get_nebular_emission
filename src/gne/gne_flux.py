"""
.. moduleauthor:: Violeta Gonzalez-Perez <violetagp@protonmail.com>
.. contributions:: 
"""

import h5py
import numpy as np
import gne.gne_io as io
import gne.gne_const as c
import gne.gne_cosmology as cosmo

def L2flux(luminosity,zz):
    """
    Calculates line Flux(erg/s/cm^2) from Luminosity(erg/s)
    """
    # Check if input is a scalar and convert to array if needed
    is_scalar = np.isscalar(luminosity)
    luminosity = np.atleast_1d(luminosity)
    
    # Initialise the flux matrix
    flux = np.zeros(np.shape(luminosity))
    
    # Luminosity distance in cm
    d_L = cosmo.luminosity_distance(zz,cm=True)
    if d_L<c.eps:
        print(f'WARNING: no flux calculation, Dl({zz})<{c.eps}')
        return flux[0] if is_scalar else flux

    # Operate with log10, to avoid numerical issues
    ind = np.where(luminosity>0)
    if np.shape(ind)[1]<1:
        print(f'WARNING: no flux calculation, L({zz})<0')
        return flux[0] if is_scalar else flux
    den = np.log10(4.0*np.pi) + 2*np.log10(d_L)
    log_flux = np.log10(luminosity[ind]) - den

    # Flux in erg/s/cm^2
    flux[ind] = 10**log_flux

    return flux[0] if is_scalar else flux


def flux2L(flux,zz):
    """
    Calculates line Luminosity(erg/s) from Flux(erg/s/cm^2)
    """
    # Check if input is a scalar and convert to array if needed
    is_scalar = np.isscalar(flux)
    flux = np.atleast_1d(flux)
    
    # Initialise the luminosity matrix
    lum = np.zeros(np.shape(flux))
    
    # Luminosity distance in cm
    d_L = cosmo.luminosity_distance(zz,cm=True)
    if d_L<c.eps:
        print(f'WARNING: no luminosity calculated, Dl({zz})<{c.eps}')
        return lum[0] if is_scalar else lum

    # Operate with log10, to avoid numerical issues
    ind = np.where(flux>0)
    if np.shape(ind)[1]<1:
        print(f'WARNING: no luminosity calculated, F({zz})<0')
        return lum[0] if is_scalar else lum
    den = np.log10(4.0*np.pi) + 2*np.log10(d_L)
    log_lum = np.log10(flux[ind]) + den

    # Luminosity in erg/s/cm^2
    lum[ind] = 10**log_lum

    return lum[0] if is_scalar else lum



def write_flux(luminosities,dataset,filenom):
    '''
    Calculate and write down fluxes from luminosities in erg/s.

    Params
    -------
    luminositites : array of floats
        Luminosities of the lines per component (erg/s).
    dataset : list of strings
        Dataset paths
    filenom : string
        Name of file with output
    '''

    # Read redshift and cosmological parameters
    f = h5py.File(filenom, 'r')
    header = f['header']
    redshift = header.attrs['redshift']
    h0 = header.attrs['h0']
    omega0 = header.attrs['omega0']
    omegab = header.attrs['omegab']
    lambda0 = header.attrs['lambda0']
    f.close()
    
    cosmo.set_cosmology(omega0=omega0, omegab=omegab,lambda0=lambda0,h0=h0)

    # Calculate fluxes for each line
    nlines = len(dataset)

    for i in range(nlines):
        # Get luminosity for this line
        lum = luminosities[i]

        # Calculate flux
        flux = L2flux(lum, redshift)

        # Split dataset path into group and name
        group, name = dataset[i].rsplit('/', 1)

        # Write to file
        io.write_data(filenom, group=group,
                      params=[flux],
                      params_names=[name],
                      params_labels=['erg s^-1 cm^-2'])
    return


def gne_flux(infile, outpath=None, out_ending=None,
             line_names=None, verbose=True):
    '''
    Calculate fluxes from luminosities
    
    Parameters
    ----------
    infile : string
        Input file
    outpath : string
        Path to output, default is output/ 
    out_ending : string
        Name root for output file
    verbose : boolean
       If True print out messages.
    '''
    lnames = line_names

    # Read information from file
    lfile= io.get_outnom(infile,dirf=outpath,nomf=out_ending,
                         verbose=verbose)
    f = h5py.File(lfile, 'r') 
    header = f['header']
    photmod_sfr = header.attrs['photmod_sfr']
    if line_names is None: lnames = c.line_names[photmod_sfr]
    group = 'sfr_data'
    # Generate a list with all the luminosity datasets
    if 'attmod' in header.attrs:
        line_datasets = [group+'/'+line+suffix for line in lnames
                         for suffix in ['_sfr', '_sfr_att']]
    else:
        line_datasets = [group+'/'+line+'_sfr' for line in lnames]
    nlines = len(line_datasets)
    # Find the first available line dataset to get dimensions
    first_line_dataset = None
    for line_dataset in line_datasets:
        if line_dataset in f:
            first_line_dataset = line_dataset
            break
    ind_lines = []; outnames = []
    if first_line_dataset is not None:
        ncomp = np.shape(f[first_line_dataset][:])[0]
        ngal = np.shape(f[first_line_dataset][:])[1]
        neblines = np.zeros((nlines, ncomp, ngal))
        # Read nebular emission lines if in file
        for i, line_dataset in enumerate(line_datasets):
            if line_dataset not in f:
                if verbose: print(f"WARNING: No flux calculated, "
                                  f"{line_dataset} not found in {lfile}")
            else:
                ind_lines.append(i)
                outnames.append(line_dataset+'_flux')
                neblines[i, :, :] = f[line_dataset][:] # erg/s
    else:
        print(f"WARNING: No flux calculated for sfr lines as no "
              f"datasets were found in {lfile}")

    # Read AGN data if present
    ind_lines_agn = []; outnames_agn = []
    if 'agn_data' in f.keys():
        group_agn = 'agn_data'
        photmod_agn = header.attrs['photmod_NLR']
        if line_names is None: lnames = c.line_names[photmod_agn]
        # Generate a list with all the luminosity datasets
        if 'attmod' in header.attrs:
            lineagn_datasets = [group_agn+'/'+line+suffix for line in lnames
                                for suffix in ['_agn', '_agn_att']]
        else:
            lineagn_datasets = [group_agn+'/'+line+'_agn' for line in lnames]
        nlines_agn = len(lineagn_datasets)
        first_line_dataset = None
        for lineagn_dataset in lineagn_datasets:
            if lineagn_dataset in f:
                first_line_dataset = lineagn_dataset
                break
        if first_line_dataset is not None:
            ngal = np.shape(f[first_line_dataset][:])[0]
            neblines_agn = np.zeros((nlines_agn, ngal))
            # Read nebular emission lines if in file
            for i, lineagn_dataset in enumerate(lineagn_datasets):
                if lineagn_dataset not in f:
                    if verbose: print(f"WARNING: No flux calculated, "
                                      f"{lineagn_dataset} not found in {lfile}")
                else:
                    ind_lines_agn.append(i)
                    outnames_agn.append(lineagn_dataset+'_flux')
                    neblines_agn[i, :] = f[lineagn_dataset][:] # erg/s
        else:
            print(f"WARNING: No flux calculated for agn lines as no "
                  f"datasets were found in {lfile}")
    f.close()

    # Calculate the fluxes
    if len(ind_lines)>0:
        subset = neblines[ind_lines,:,:]
        write_flux(subset,outnames,lfile)
        
    if len(ind_lines_agn)>0:
        subset = neblines_agn[ind_lines_agn,:]
        write_flux(subset,outnames_agn,lfile)

    print('SUCCESS (gne_flux)')
    return 
