#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 14:36:13 2023

@Author: expox7, vgp
"""

import h5py
import numpy as np
import src.gne_io as io
import src.gne_const as c
from src.gne_io import check_file, get_param
import sys
import warnings
from src.gne_cosmology import emission_line_flux

def get_f_saito20(z):
    '''
    Calculate the dust attenuation fraction parameter
    from Eq. 5 in Saito+2020, his parameter is defined as
    f = E_stellar(B-V)/E_gas(B-V)
    and is a function of redshift (z)

    Parameters
    ----------
    z : float
      Redshift
     
    Returns
    -------
    f : float
    '''
    if (z <= 2.8):
        f = 0.44 + 0.2*z
    else:
        f = 1
    return f   


######################

def coef_att_cardelli(wavelength, Mcold_disc, rhalf_mass_disc, Z_disc, h0=0.7, costheta=0.3, albedo=0.56):
    '''
    Given the wavelength, the cold gas mass, the half-mass radius and the global metallicity of the disk,
    along with the assumed albedo and scattering angle, it gives the attenuation coefficient.

    Parameters
    ----------
    wavelength : floats
     Wavelength (A).
    Mcold_disc : floats
     Cold gas mass (Msun).
    rhalf_mass_disc : floats
     Half-mass radius of the disk (Msun).
    Z_disc : floats
     Disc's global metallicity.
    costheta : float
     Cosine of the assumed cattering angle.
    albedo : float
     Assumed albedo.
     
    Returns
    -------
    coef_att : floats
    '''
    
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    
    a_disc = 1.68
    if wavelength > 2000: #Guiderdoni & Roca-Volmerange 1987 (see De Lucia & Blaizot 2007)
        s = 1.6
    else:
        s = 1.35
    Al_Av = cardelli(wavelength)
    sectheta = 1./costheta

    mean_col_dens_disc_log = (np.log10(Mcold_disc*h0) + np.log10(c.Msun) - 
    np.log10(1.4*c.mp*np.pi)-2.*np.log10(a_disc*rhalf_mass_disc*h0*c.Mpc_to_cm))
    
    tau_disc = np.log10(Al_Av) + np.log10((Z_disc/c.zsun)**s) + mean_col_dens_disc_log - 21*np.log10(2.1)
    tau_disc = 10.**tau_disc
    
    al_disc = (np.sqrt(1.-albedo))*tau_disc
    
    A_lambda = -2.5*np.log10((1.-np.exp(-al_disc*sectheta))/(al_disc*sectheta))
    
    coef_att = 10.**(-0.4*A_lambda)
    
    return coef_att


def attenuation(nebline, att_param=None, att_ratio_lines=None,
                redshift=0, h0=0.7, attmod='cardelli89',origin='sfr',
                photmod='gutkin16', cut=None, verbose=True):
    '''
    Get the attenuated emission lines from the raw ones.

    Parameters
    ----------
    nebline : floats
     Array with the luminosity of the lines per component. (Lsun per unit SFR(Mo/yr) for 10^8yr).
    att_param : floats
     Array with the parameter values of the attenuation model.
    att_ratio_lines : strings
     Names of the lines corresponding to the values in att_params when attmod=ratios.
     They should be written as they are in the selected model (see gne_const).
    redshift : float
     Redshift of the input data.
    attmod : string
     Attenuation model.
    photmod : string
     Photoionisation model to be used for look up tables.
    cut : integers
     List of indexes of the selected galaxies from the samples.
    verbose : boolean
     If True print out messages.

    Returns
    -------
    nebline_att, coef_att : floats
      
    Notes
    -------
    It will ignore the lines for which, for any reason, attenuation can not be calculated.
    '''
    
    ncomp = len(nebline)
    coef_att = np.full(nebline.shape,c.notnum)

    if att_param[0][0] != None: ###here improve this to be more general
        if attmod not in c.attmods: ###here to be outside, as a test
            if verbose:
                print('STOP (gne_photio.attenuation): Unrecognised model for attenuation.')
                print('                Possible attmod= {}'.format(c.attmods))
            sys.exit()
        elif attmod=='ratios':
            for i, line in enumerate(c.line_names[photmod]):
                if line in att_ratio_lines:
                    ind = np.where(np.array(att_ratio_lines)==line)[0]
                else:
                    continue
                
                for comp in range(ncomp):
                    if comp==0:
                        coef_att[comp,i] = att_param[ind] * c.line_att_coef_all(redshift)
                    else:
                        if origin=='sfr':
                            coef_att[comp,i] = coef_att[0,i]
                        else:
                            coef_att[comp,i] = 1.
            coef_att[(coef_att!=coef_att)&(coef_att!=c.notnum)] = 1.
        elif attmod=='cardelli89':
            Rhm = att_param[0] ###here need to ensure that the 3 properties are properly handled
            Mcold_disc = att_param[1]
            Z_disc = att_param[2]

            coef_att = np.full(nebline.shape,c.notnum) ###here why needed again?
            for comp in range(ncomp): ###here this needs to go over components
                for i, line in enumerate(c.line_names[photmod]):
                    if comp==0:
                        coef_att[comp,i] = coef_att_cardelli(c.line_wavelength[photmod][i], 
                                                             Mcold_disc=Mcold_disc,
                                                             rhalf_mass_disc=Rhm, 
                                                             Z_disc=Z_disc, h0=h0,
                                                             costheta=0.3, albedo=0.56) * c.line_att_coef_all(redshift)
                    else:
                        coef_att[comp,i] = coef_att[0,i]
            coef_att[(coef_att!=coef_att)&(coef_att!=c.notnum)] = 1.
    
    nebline_att = np.full(nebline.shape,c.notnum)
    ind = np.where((coef_att!=c.notnum))
    nebline_att[ind] = nebline[ind]*coef_att[ind]

    return nebline_att, coef_att


# def coef_att_ratios(infile,cols_notatt,cols_att,cols_photmod,inputformat='HDF5',photmod='gutkin16',verbose=True):
#     '''
#     It reads luminosities of lines with and without attenuation
#     from line emission data and it returns the attenuation coefficients.

#     Parameters
#     ----------
#     infile : string
#      Name of the input file. 
#      - In text files (*.dat, *txt, *.cat), columns separated by ' '.
#      - In csv files (*.csv), columns separated by ','.
#     cols_notatt : list
#      Attenuated flux lines calculated by the semi-analytic model of the input data. 
#      Used to calculate attenuation coefficients for the "ratio" attenuation model.
#      - For text or csv files: list of integers with column position.
#      - For hdf5 files: list of data names.
#     cols_att : list
#      Not attenuated flux lines calculated by the semi-analytic model of the input data. 
#      Used to calculate attenuation coefficients for the "ratio" attenuation model.
#      - For text or csv files: list of integers with column position.
#      - For hdf5 files: list of data names.
#     cols_photmod : list
#      Index in the list of lines of the photoionization model of the lines for which 
#      attenuation is going to be calculated in the "ratio" attenuation model.
#     inputformat : string
#      Format of the input file.
#     photmod : string
#      Photoionisation model to be used for look up tables.
#     verbose : boolean
#      If True print out messages.

#     Returns
#     -------
#     coef_att : floats
#     '''
    
#     check_file(infile, verbose=verbose)
    
#     ncomp = len(cols_notatt)
        
#     lines = c.line_names[photmod]
#     numlines = len(lines)
    
#     cols_att = np.array(cols_att)
#     cols_notatt = np.array(cols_notatt)
#     cols_photmod = np.array(cols_photmod)
    
#     if inputformat=='HDF5':
#         with h5py.File(infile, 'r') as f:
#             hf = f['data']
            
#             coef_att = np.empty((ncomp,numlines,len(hf[cols_notatt[0,0]])))
#             coef_att.fill(c.notnum)
            
#             for i in range(len(cols_photmod)):
#                 for comp in range(ncomp):
#                     ind = np.where(hf[cols_notatt[comp,i]][:]!=0)
#                     ind2 = np.where(hf[cols_notatt[comp,i]][:]==0)
#                     coef_att[comp,cols_photmod[i]][ind] = hf[cols_att[comp,i]][ind]/hf[cols_notatt[comp,i]][ind]
#                     coef_att[comp,cols_photmod[i]][ind2] = 1
#     elif inputformat=='textfile':
#         ih = io.nheader(infile)        
#         X = np.loadtxt(infile,skiprows=ih).T
        
#         coef_att = np.empty((ncomp,numlines,len(X[0])))
#         coef_att.fill(c.notnum)
        
#         for i in range(len(cols_photmod)):
#             for comp in range(ncomp):
#                 if ncomp!=1:
#                     ind = np.where(X[cols_notatt[comp,i]]!=0)
#                     ind2 = np.where(X[cols_notatt[comp,i]]==0)
#                     coef_att[comp,cols_photmod[i]][ind] = X[cols_att[comp,i]][ind]/X[cols_notatt[comp,i]][ind]
#                     coef_att[comp,cols_photmod[i]][ind2] = 1
#                 else:
#                     ind = np.where(X[cols_notatt[comp,i]]!=0)
#                     ind2 = np.where(X[cols_notatt[comp,i]]==0)
#                     if comp==0:
#                         coef_att[comp,cols_photmod[i]][ind] = (X[cols_att[comp,i]][ind]-X[cols_att[1,i]][ind])/(X[cols_notatt[comp,i]][ind]-X[cols_notatt[1,i]][ind])
#                         coef_att[comp,cols_photmod[i]][ind2] = 1
#                     else:
#                         coef_att[comp,cols_photmod[i]][ind] = X[cols_att[comp,i]][ind]/X[cols_notatt[comp,i]][ind]
#                         coef_att[comp,cols_photmod[i]][ind2] = 1
                    
                    
#         del X        
    
#     return coef_att



def get_AlAv_cardelli89(waveAA,Rv=c.Rv):
    '''
    Given the wavelength, returns Al/Av following
    Cardelli+1989 (doi:10.1086/167900)

    Parameters
    ----------
    waveAA : array of floats
     Wavelength (A)
     
    Returns
    -------
    Al_Av : array floats
    '''
    
    wl=waveAA/10000. #microm
    x=1./wl
    
    if (x < 0.3) or (x > 10):
        print('STOP (gne_att.cardelli): ',
              'Wavelength out of range.')
        sys.exit()
        return
    elif (x <= 1.1): #IR
        ax = 0.574*(x**1.61) 
        bx = -0.527*(x**1.61)
    elif (x <= 3.3): #Optical/NIR
        y = x-1.82
        ax = (1.+0.17699*y - 0.50447*(y**2) - 0.02427*(y**3) +
              0.72085*(y**4) + 0.01979*(y**5) -
              0.77530*(y**6) + 0.32999*(y**7)) 
        bx = (1.41338*y + 2.28305*(y**2) + 1.07233*(y**3) -
              5.38434*(y**4) - 0.62251*(y**5) +
              5.30260*(y**6) - 2.09002*(y**7))
    elif (x <= 8): #UV
        if (x < 5.9):
            Fa = 0
            Fb = 0
        else: 
            Fa = -0.04473*(x-5.9)**2 - 0.009779*(x-5.9)**3
            Fb = 0.2130*(x-5.9)**2 + 0.1207*(x-5.9)**3
        ax = 1.752-0.316*x - 0.104/((x-4.67)**2 + 0.341) + Fa
        bx = -3.090+1.825*x + 1.206/((x-4.62)**2 + 0.263) + Fb
    else:
        ax = -1.073 - 0.628*(x-8) + 0.137*(x-8)**2 - 0.070*(x-8)**3
        bx = 13.670 + 4.257*(x-8) - 0.420*(x-8)**2 + 0.374*(x-8)**3
    
    Al_Av = ax+bx/Rv
    return Al_Av


def get_A_lambda(tau,costheta=c.costheta,albedo=c.albedo):
    """
    Calculate the attenuation coefficient, A_lambda.
    
    Arguments can be floats or arrays of floats.
    """
    # Vectorise
    tau = np.asarray(tau)
    costheta = np.asarray(costheta)
    albedo = np.asarray(albedo)

    # Calculate the attenuation coefficient
    al = tau*np.sqrt(1.0 - albedo)    
    x = al/costheta
    A_lambda = -2.5 * np.log10((1.0 - np.exp(-x)) / x)
    
    return A_lambda


def factor_delucia07(zgas):
    '''
    Calculate the factor for the optical depth that
    depends on the galaxy global properties,
    following De Lucia and Blaziot 2007 (0606519)
    '''
    ngal = len(zgas)
    fgal = np.full((ngal),c.notnum)

    
    return fgal

def att_favole20(wavelengths,lzgas,Rv=c.Rv,costheta=c.costheta,
                 albedo=c.albedo):
    '''
    Calculate attenuation coefficients following Favole+2020 (1908.05626)

    Parameters
    ----------
    wavelenths : array of floats
        Wavelengths (AA) at which calculate the attenuation coefficients
    lzgas : array of floats
        log10(Z_cold_gas)
    Rv : float
        Slope of the extinction curve in the optical
    costheta : float
        Cosine of the typical dust scattering angle
    albedo : float
        Dust albedo (fraction between 0 and 1)
    Return
    ------
    coeff : array of floats
    '''
    ngal = len(lzgas)
    coeff = np.full((len(wavelengths),ngal),c.notnum)

    # Calculate the factor for the optical depth of each galaxy


    # For each galaxy get the attenuation for each wavelength
    for ii,wl in enumerate(wavelengths):
        # Use the Cardelli+89 model to obtain A_l/A_V
        alav = get_AlAv_cardelli89(wl,Rv=Rv)

        #Calculate the galaxy optical depth
        if wl > 2000:
            # Following Guiderdoni & Roca-Volmerange 1987 
            s = 1.6
        else:
            s = 1.35
        nH = np.full((ngal),1)###here
        loga = (s*(lzgas - np.log10(c.zsun)) +
                nH - 21 - np.log10(2.1)) 
        tau = alav*10.**loga
        
        # For each galaxy calculate the attenuation coefficient
        coeff[ii,:] = get_A_lambda(tau,costheta=costheta,albedo=albedo)

    return coeff                                          


def gne_att(infile, outpath=None, attmod='cardelli89',
            line_att=None,att_config=None,verbose=True):
    '''
    Get the neccessary information to calculate
    dust-attenuated luminosities
    
    Parameters
    ----------
    infile : string
        Input file
    outpath : string
        Path to output, default is output/ 
    attmod : str
        Attenuation model ('cardelli89' or 'ratios')
    att_config : dict
       Model-specific parameters:
       - 'cardelli89': {'s': 1.6, 'costheta': 0.3}
       - 'ratios': {'ratios': array, 'rlines': list}
    verbose : boolean
       If True print out messages.
    '''
    # Add attenuation information to the line file
    lfile= io.get_outnom(infile,dirf=outpath,verbose=verbose)
    nattrs = io.add2header(lfile,['attmod'],[attmod],verbose=verbose)
        
    # Get emission lines
    f = h5py.File(lfile, 'r') 
    header = f['header']
    zz = header.attrs['redshift']
    photmod_sfr = header.attrs['photmod_sfr']
    lnames = c.line_names[photmod_sfr]
    nlines = len(lnames)
    group = 'sfr_data'
    ncomp = np.shape(f[group+'/'+lnames[0]+'_sfr'][:])[0]
    ngal = np.shape(f[group+'/'+lnames[0]+'_sfr'][:])[1]
    neblines = np.zeros((nlines, ncomp, ngal))
    for i, line in enumerate(lnames):
        neblines[i, :, :] = f[group+'/'+line+'_sfr'][:]
    if attmod != 'ratios':
        lzgas = f[group+'/lz_sfr'][:]  # log10(Z_cold_gas)
        lm_gas = f['data/lm_gas'][:] # log10(M/Msun)
        ###here how to use one or the other: disc/bulge
        h_gas = f['data/h_gas'][:]  #Scalelength(Mpc)
        ###here where to modify it to R1/2?

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
        if attmod != 'ratios':
            lzgas_agn = f[group_agn+'/lz_agn'][:] # log10(Z_cold_gas)
    f.close()

    # Get wavelengths and prep attenuation coefficients
    wavelengths = c.line_wavelength[photmod_sfr]
    coeff = np.full(neblines.shape,c.notnum)
    if AGN:
        wavelengths_agn = c.line_wavelength[photmod_agn]
        coeff_agn = np.full(neblines_agn.shape,c.notnum)
        
    # Get the information needed by the dust attenuation model
    if attmod not in c.attmods:
        if verbose:
            print('STOP (gne_att): Unrecognised dust model.')
            print('                Possible ones= {}'.format(c.attmods))
        sys.exit()
    elif attmod == 'ratios':
        print('Work in progress: Ratios not checked')
    #    att_ratios = att_config.get('ratios')
    #    att_rlines = att_config.get('rlines')
    else:
        Rv = get_param(att_config, 'Rv', c.Rv)
        costheta = get_param(att_config, 'costheta', c.costheta) 
        albedo = get_param(att_config, 'albedo', c.albedo)
        # Add parameters to header
        config_names  = ['Rv', 'costheta', 'albedo']
        config_values = [Rv, costheta, albedo]
        io.add2header(lfile,config_names,config_values,verbose=False)

        for icomp in range(ncomp):
            coeff[:,icomp,:] = att_favole20(wavelengths,lzgas[:,icomp],
                                            Rv=Rv,costheta=costheta,
                                            albedo=albedo)

        #gal_factor = get_delucia07(lfile)
        #gal_factor = np.full(neblines.shape[-1],1)
        #print(np.shape(gal_factor)); exit()

        
        #coeff = get_coeff_favole20(neblines_agn,wavelengths)
        #
        #if AGN:
        #    wavelengths = c.line_wavelength[photmod_agn]
        #    coeff_agn = get_coeff_favole20(neblines_agn,wavelengths)
        #    coeff_agn = np.full(neblines_agn.shape,c.notnum)

            
    # Lines with valid attenuation coefficients
    ind = (coeff > c.notnum)
    if AGN:
        ind_agn = (coeff_agn > c.notnum)

    # Extra gas attenuation on top of the stellar one
    if line_att:
        f = get_f_saito20(zz)
        coeff[ind] = coeff[ind]/f
        if AGN:
            coeff_agn[ind_agn] = coeff_agn[ind_agn]/f
            
        #Add information in header
        nattrs = io.add2header(lfile,['att_f_saito20'],[f],verbose=verbose)

    # Calculate dust-attenuated emission lines
    neblines_att = neblines
    neblines_att[ind] = neblines[ind]*(10**(-0.4*coeff[ind]))
    lnames_att = [line + '_sfr_att' for line in lnames]
    labels = np.full(np.shape(lnames),'erg s^-1')
    io.write_data(lfile,group=group,params=neblines_att,
                  params_names=lnames_att,params_labels=labels)

    if AGN:
        neblines_agn_att = neblines_agn
        neblines_agn_att[ind_agn] = neblines_agn[ind_agn]*(
            10**(-0.4*coeff_agn[ind_agn]))
        lnames_att = [line + '_agn_att' for line in lnames_agn]
        labels = np.full(np.shape(lnames_agn),'erg s^-1')

        io.write_data(lfile,group=group_agn,params=neblines_agn_att,
                      params_names=lnames_att,params_labels=labels)

    return
