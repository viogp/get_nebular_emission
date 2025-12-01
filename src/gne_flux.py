"""
.. moduleauthor:: Violeta Gonzalez-Perez <violetagp@protonmail.com>
.. contributions:: 
"""

import h5py
import numpy as np
import src.gne_io as io
import src.gne_const as c
#import sys
#import warnings
from src.gne_cosmology import emission_line_flux
 
def calculate_flux(nebline,filenom,origin='sfr'):
    '''
    Get the fluxes for the emission lines given the luminosity and redshift.

    Params
    -------
    nebline : array of floats
        Luminosities of the lines per component.
        Lsun for L_AGN = 10^45 erg/s
    filenom : string
        Name of file with output
    origin : string
        Emission source (star-forming region or AGN).
      
    Returns
    -------
    fluxes : floats
        Array with the fluxes of the lines per component.
    '''
    
    if nebline.any():
        # Read redshift and cosmological parameters
        f = h5py.File(filenom, 'r')
        header = f['header']
        redshift = header.attrs['redshift']
        h0 = header.attrs['h0']
        omega0 = header.attrs['omega0']
        omegab = header.attrs['omegab']
        lambda0 = header.attrs['lambda0']
        f.close()
        
        set_cosmology(omega0=omega0, omegab=omegab,lambda0=lambda0,h0=h0)
        
        luminosities = np.zeros(nebline.shape)
        luminosities[nebline>0] = np.log10(nebline[nebline>0]*h0**2)
        if (origin=='agn') and (luminosities.shape[0]==2):
            luminosities[1] = 0
            
        fluxes = np.zeros(luminosities.shape)
        for comp in range(luminosities.shape[0]):
            for i in range(luminosities.shape[1]):
                for j in range(luminosities.shape[2]):
                    if luminosities[comp,i,j] == 0:  
                        fluxes[comp,i,j] = 0
                    else:
                        fluxes[comp,i,j] = logL2flux(luminosities[comp,i,j],redshift)
    else:
        fluxes = np.copy(nebline)
            
    return fluxes


def gne_flux(infile, outpath=None, verbose=True):
    '''
    Calculate fluxes from luminosities
    
    Parameters
    ----------
    infile : string
        Input file
    outpath : string
        Path to output, default is output/ 
    verbose : boolean
       If True print out messages.
    '''
    # Read information from file
    lfile= io.get_outnom(infile,dirf=outpath,verbose=verbose)
    f = h5py.File(lfile, 'r') 
    header = f['header']
    photmod_sfr = header.attrs['photmod_sfr']
    lnames = c.line_names[photmod_sfr]
    nlines = len(lnames)
    group = 'sfr_data'
    ncomp = np.shape(f[group+'/'+lnames[0]+'_sfr'][:])[0]
    ngal = np.shape(f[group+'/'+lnames[0]+'_sfr'][:])[1]
    neblines = np.zeros((nlines, ncomp, ngal))
    for i, line in enumerate(lnames):
        neblines[i, :, :] = f[group+'/'+line+'_sfr'][:]
    #if attmod != 'ratios':
    #    mgasr_type = io.decode_string_list(header.attrs['mgasr_type'])

    if 'agn_data' not in f.keys():
        AGN = False
    else:
        AGN = True
        group_agn = 'agn_data'
        photmod_agn = header.attrs['photmod_NLR']
        lnames_agn = c.line_names[photmod_agn]
        nlines = len(lnames_agn)
        ngal = len(f[group_agn+'/'+lnames_agn[0] +'_agn'])
        neblines_agn = np.zeros((nlines, ngal))
        for i, line in enumerate(lnames_agn):
            neblines_agn[i, :] = f[group_agn+'/'+line+'_agn'][:]
        #if attmod != 'ratios':
        #    lzgas_agn = f[group_agn+'/lz_agn'][:] # log10(Z_cold_gas)
    f.close()
    print('work in progress for the flux calculation')         
    # Calculate the fluxes
    #fluxes_sfr = calculate_flux(nebline_sfr,outfile,origin='sfr')
    #if AGN: if att:

#    io.write_data(lfile,group=group,params=neblines_att,
#                  params_names=lnames_att,params_labels=labels)
#
#    if AGN:
#        neblines_agn_att = neblines_agn
#        neblines_agn_att[ind_agn] = neblines_agn[ind_agn]*(
#            10**(-0.4*coeff_agn[ind_agn]))
#        lnames_att = [line + '_agn_att' for line in lnames_agn]
#        labels = np.full(np.shape(lnames_agn),'erg s^-1')
#
#        io.write_data(lfile,group=group_agn,params=neblines_agn_att,
#                      params_names=lnames_att,params_labels=labels)

    return 
