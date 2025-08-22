"""
.. moduleauthor:: Violeta Gonzalez-Perez <violetagp@protonmail.com>
"""
import numpy as np
import src.gne_const as c
import src.gne_io as io
from src.gne_Z import get_lzgas

def get_sfrdata(infile,cols,selection=None,
                h0=None,units_h0=False, units_Gyr=False,
                inoh = False, mtot2mdisk=True, 
                inputformat='hdf5',testing=False,verbose=False):
    '''
    Get stellar mass as log10(M/Msun),
    sSFR as log10(SFR/M/yr) and
    gas metallicity as log10(Zgas=MZcold/Mcold)

    Parameters
    ----------
    infile : strings
     List with the name of the input files. 
     - In text files (*.dat, *txt, *.cat), columns separated by ' '.
     - In csv files (*.csv), columns separated by ','.
    inputformat : string
     Format of the input file.
    cols : list
     - [[component1_stellar_mass,sfr,Z],[component2_stellar_mass,sfr,Z],...]
     - Expected : component1 = total or disk, component2 = bulge
     - For text or csv files: list of integers with column position.
     - For hdf5 files: list of data names.
    cutcols : list
     Parameters to look for cutting the data.
     - For text or csv files: list of integers with column position.
     - For hdf5 files: list of data names.
    mincuts : strings
     Minimum value of the parameter of cutcols in the same index. All the galaxies below won't be considered.
    maxcuts : strings
     Maximum value of the parameter of cutcols in the same index. All the galaxies above won't be considered.
    attmod : string
     Attenuation model.
    inoh : boolean
       If true, the input is assumed to be 12+log10(O/H), otherwise Zgas    
    units_h0 : bool
    mtot2mdisk : boolean
      If True transform the total mass into the disk mass. disk mass = total mass - bulge mass.
    verbose : boolean
      If True print out messages
    testing : boolean
      If True only run over few entries for testing purposes

    Returns
    -------
    lms, lssfr, lzgas : array of floats
    '''
    ncomp = io.get_ncomponents(cols)

    ms,sfr,zgas = io.read_sfrdata(infile, cols, selection,
                               inputformat=inputformat, 
                               testing=testing, verbose=verbose)

    # Change units if needed
    if units_h0:
        ms = ms/h0
        sfr = sfr/h0
    if units_Gyr:
        sfr = sfr/1e9

    # Set to a default value if negative stellar masses
    for comp in range(ncomp):
        ind = np.where((ms[comp,:]<=0.) | (sfr[comp,:]<=0) | (zgas[comp,:]<=0))
        ms[comp,ind] = c.notnum
        sfr[comp,ind] = c.notnum
        zgas[comp,ind] = c.notnum

    # Calculate the disk mass if we have only the total and bulge mass
    ms_tot = ms[0,:]
    if mtot2mdisk:
        if ncomp!=2:
            if verbose:
                print('STOP (gne_io.get_data): ',
                      'mtot2mdisk can only be True with two components.')
            sys.exit()
        # Calculate the disk mass and store it as the first component
        msdisk = ms[0,:] - ms[1,:]
        ms[0,:] = msdisk
    else:
        if ncomp!=1:
            # Add mass from components if above 0
            ms_tot = np.sum(np.where(ms > 0, ms, 0), axis=0)

    # Take the log of the total stellar mass:
    lms_tot = np.zeros(len(ms_tot)); lms_tot.fill(c.notnum)
    ind = np.where(ms_tot > 0.)
    lms_tot[ind] = np.log10(ms_tot[ind])

    # Take the log of the stellar masses:
    lms = np.zeros(ms.shape); lms.fill(c.notnum)
    for comp in range(ncomp):
        ind = np.where(ms[comp,:] > 0.)
        lms[comp,ind] = np.log10(ms[comp,ind])

    # Obtain log10(sSFR/yr)
    lssfr = np.zeros(np.shape(sfr)); lssfr.fill(c.notnum)
    for comp in range(ncomp):
        ind = np.where((sfr[comp,:] > 0.) & (lms[comp,:] > 0.))
        lssfr[comp,ind] = np.log10(sfr[comp,ind]) - lms[comp,ind]
    #print(lssfr.shape,sfr.shape,zgas.shape); exit() ###here
    #if ncomp!=1: #Total SFR
    #    lssfr_tot = np.zeros(len(lssfr))
    #    ssfr = np.zeros(lssfr.shape)
    #    for comp in range(ncomp):
    #        ind = np.where(lssfr[comp,:]!=c.notnum)
    #        ssfr[comp,ind] = 10.**(lssfr[comp,ind])
    #
    #    ins = np.sum(ssfr,axis=1)
    #    ind = np.where(ins>0)
    #    lssfr_tot[ind] = np.log10(ins[ind])
            
    # Obtain log10(Zgas=MZcold/Mcold)
    lzgas = get_lzgas(zgas,inoh=inoh)

    return lms.T,lssfr.T,lzgas.T

