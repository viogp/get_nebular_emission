"""
.. moduleauthor:: Violeta Gonzalez-Perez <violetagp@protonmail.com>
.. contributions:: Olivia Vidal <ovive.pro@gmail.com>
.. contributions:: Julen Expósito-Márquez <expox7@gmail.com>
"""
import time
import numpy as np
import src.gne_io as io
from src.gne_une import get_une_sfr, get_une_agn
from src.gne_Z import correct_Z,get_zgasagn
from src.gne_m_sfr import get_sfrdata
from src.gne_Lagn import bursttobulge,get_Lagn
import src.gne_const as c
from src.gne_stats import components2tot
from src.gne_photio import get_lines, get_limits, calculate_flux
from src.gne_att import attenuation
from src.gne_plots import make_testplots

def gne(infile,redshift,snap,h0,omega0,omegab,lambda0,vol,mp,
        inputformat='hdf5',outpath=None,
        units_h0=False,units_Gyr=False,units_L40h2=False,
        une_sfr_nH='kashino19',une_sfr_U='kashino19',
        photmod_sfr='gutkin16',
        q0=c.q0_orsi, z0=c.Z0_orsi, gamma=c.gamma_orsi,
        T=10000,xid_sfr=0.3,co_sfr=1,
        m_sfr_z=[None],mtot2mdisk=True,
        inoh=False,LC2sfr=False,
        IMF=['Kroupa','Kroupa'],imf_cut_sfr=100,
        AGN=False,une_agn_nH=None,une_agn_spec='feltre16',
        une_agn_U='panuzzo03',photmod_agn='feltre16',
        xid_agn=0.5,alpha_agn=-1.7,
        agn_nH_params=None,AGNinputs='Lagn', Lagn_params=[None],
        Zgas_NLR=None,Z_correct_grad=False,
        zeq=None,infile_z0=None,
        att=False,attmod='cardelli89',
        att_params=[None], att_ratio_lines=[None],
        flux=False,
        extra_params=[None], extra_params_names=[None],
        extra_params_labels=[None],
        cutcols=[None], mincuts=[None], maxcuts=[None],
        cutlimits=False, 
        testing=False,verbose=True):
    '''
    Calculate emission lines given the properties of model galaxies

    Parameters
    ----------
    infile : strings
     List with the name of the input files. 
     - In text files (*.dat, *txt, *.cat), columns separated by ' '.
     - In csv files (*.csv), columns separated by ','.
    m_sfr_z : list
     - [[component1_stellar_mass,sfr/LC,Z],[component2_stellar_mass,sfr/LC,Z],...]
     - For text or csv files: list of integers with column position.
     - For hdf5 files: list of data names.
    inputformat : string
        Format of the input file: 'hdf5' or 'txt'.
    infile_z0 : strings
     List with the name of the input files with the galaxies at redshift 0. 
     - In text files (*.dat, *txt, *.cat), columns separated by ' '.
     - In csv files (*.csv), columns separated by ','.
    h0 : float
      If not None: value of h, H0=100h km/s/Mpc.
    redshift : float
     Redshift of the input data.
    snap: integer
        Simulation snapshot number
    cutcols : list
     Parameters to look for cutting the data.
     - For text or csv files: list of integers with column position.
     - For hdf5 files: list of data names.
    mincuts : floats
     Minimum value of the parameter of cutcols in the same index. All the galaxies below won't be considered.
    maxcuts : floats
     Maximum value of the parameter of cutcols in the same index. All the galaxies above won't be considered.
    att : boolean
     If True calculates attenuated emission.
    att_params : list
     Parameters to look for calculating attenuation. See gne_const to know what each model expects.
     - For text or csv files: list of integers with column position.
     - For hdf5 files: list of data names.
    att_ratio_lines : strings
     Names of the lines corresponding to the values in att_params when attmod=ratios.
     They should be written as they are in the selected model (see gne_const).
    flux : boolean
     If True calculates flux of the emission lines based on the given redshift.
    IMF : array of strings
       Assumed IMF for the input data of each component, [[component1_IMF],[component2_IMF],...]
    q0 : float
     Ionization parameter constant to calibrate Orsi 2014 model for nebular regions. q0(z/z0)^-gamma
    z0 : float
     Ionization parameter constant to calibrate Orsi 2014 model for nebular regions. q0(z/z0)^-gamma
    gamma : float
     Ionization parameter constant to calibrate Orsi 2014 model for nebular regions. q0(z/z0)^-gamma
    T : float
     Typical temperature of ionizing regions.
    AGN : boolean
     If True calculates emission from the narrow-line region of AGNs.
    AGNinputs : string
     Type of inputs for AGN's bolometric luminosity calculations.
    Lagn_params : list
     Inputs for AGN's bolometric luminosity calculations.
     - For text or csv files: list of integers with column position.
     - For hdf5 files: list of data names.
    Zgas_NLR : list of integer (text file) or strings (hdf5 file)
        Location of the central metallicity in input files
    Z_correct_gradrection : boolean
        If True, corrects Zgas_NLR using gradients from the literature
    agn_nH_params : list
     Inputs for the calculation of the volume-filling factor.
     - For text or csv files: list of integers with column position.
     - For hdf5 files: list of data names.
    extra_params : list
     Parameters from the input files which will be saved in the output file.
     - For text or csv files: list of integers with column position.
     - For hdf5 files: list of data names.
    extra_params_names : strings
     Names of the datasets in the output files for the extra parameters.
    extra_params_labels : strings
     Description labels of the datasets in the output files for the extra parameters.
    attmod : string
     Attenuation model.
    une_sfr_nH : string
        Model to go from galaxy properties to Hydrogen (or e) number density.
    une_sfr_U : string
        Model to go from galaxy properties to ionising parameter.
    une_agn_nH : list of 2 strings
        Profile assumed for the gas around NLR AGN and provided radii.
    une_agn_spec : string
        Model for the spectral distribution for AGNs.
    une_sfr_U : string
        Model to go from galaxy properties to AGN ionising parameter.
    photmod_sfr : string
     Photoionisation model to be used for look up tables.
    photmod_agn : string
     Photoionisation model to be used for look up tables.
    inoh : boolean
       If true, the input is assumed to be 12+log10(O/H), otherwise Zgas
    LC2sfr : boolean
     If True magnitude of Lyman Continuum photons expected as input for SFR.
    cutlimits : boolean
     If True the galaxies with U, ne and Z outside the photoionization model's grid limits won't be considered.
    mtot2mdisk : boolean
     If True transform the total mass into the disk mass. disk mass = total mass - bulge mass.
    xid_agn : float
     Dust-to-metal ratio for the AGN photoionisation model.
    alpha_agn : float
     Alpha value for the AGN photoionisation model.
    xid_sfr : float
     Dust-to-metal ratio for the SF photoionisation model.
    co_sfr : float
     C/O ratio for the SF photoionisation model.
    imf_cut_sfr : float
     Solar mass high limit for the IMF for the SF photoionisation model.
    units_h0: boolean
        True if input units with h
    units_Gyr: boolean
        True if input units with */Gyr
    units_L40h2: boolean
        True if input units with 1e40erg/s
    testing : boolean
        If True only run over few entries for testing purposes
    verbose : boolean
        If True print out messages

    Notes
    -------
    This code returns an .hdf5 file with the mass, specific star formation rate,
    electron density, metallicity, ionization parameter, and the emission lines.

    '''

    # Generate header in the output file from input
    outfile = io.generate_header(infile,redshift,snap,
                                 h0,omega0,omegab,lambda0,vol,mp,
                                 units_h0,outpath=outpath,
                                 une_sfr_nH=une_sfr_nH,une_sfr_U=une_sfr_U,
                                 photmod_sfr=photmod_sfr,
                                 une_agn_nH=une_agn_nH,
                                 une_agn_spec=une_agn_spec,
                                 une_agn_U=une_agn_U,
                                 photmod_agn=photmod_agn,
                                 attmod=attmod,verbose=verbose)

    #----------------HII region calculation------------------------
    # Number of components
    ncomp = io.get_ncomponents(m_sfr_z)
    
    # Time variables
    start_total_time = time.perf_counter()
    start_time = time.perf_counter()

    # Get indexes for selection
    cut = io.get_selection(infile,outfile,inputformat=inputformat,
                           cutcols=cutcols,mincuts=mincuts,maxcuts=maxcuts,
                           testing=testing,verbose=verbose)

    # Read the input data and correct it to the adequate units, etc.
    lms, lssfr, lzgas = get_sfrdata(infile,m_sfr_z,selection=cut,
                                    h0=h0,units_h0=units_h0,
                                    units_Gyr=units_Gyr,
                                    inoh = inoh,LC2sfr=LC2sfr,
                                    mtot2mdisk=mtot2mdisk,
                                    inputformat=inputformat,
                                    testing=testing,verbose=verbose)

    epsilon_param_z0 = [None]
    if infile_z0 is not None:
        epsilon_param_z0 = io.read_data(infile_z0,cut,
                                        inputformat=inputformat,
                                        params=agn_nH_params,
                                        testing=testing,
                                        verbose=verbose)
    
    # Modification of the stellar mass-metallicity relation
    if zeq is not None:
        minZ, maxZ = get_limits(propname='Z', photmod=photmod_sfr)
        ###here this does not work due to Lagn_param, need to change this dependency
        lzgas = correct_Z(zeq,lms,lzgas,minZ,maxZ,Lagn_param)

    # Characterise the HII regions from galaxy global properties
    lu_sfr, lnH_sfr = get_une_sfr(lms, lssfr, lzgas, outfile,
                                  q0=q0, z0=z0,gamma=gamma, T=T,
                                  epsilon_param_z0=epsilon_param_z0,
                                  IMF=IMF,une_sfr_nH=une_sfr_nH,
                                  une_sfr_U=une_sfr_U,verbose=verbose)
    if verbose:
        print('SF:')
        print(' U and nH calculated.')
            
    lu_o_sfr = np.copy(lu_sfr)
    lnH_o_sfr = np.copy(lnH_sfr)
    lzgas_o_sfr = np.copy(lzgas)

    # Obtain spectral emission lines from HII regions
    nebline_sfr = get_lines(lu_sfr.T,lnH_sfr.T,lzgas.T,photmod=photmod_sfr,
                            xid_phot=xid_sfr, co_phot=co_sfr,
                            imf_cut_phot=imf_cut_sfr,verbose=verbose)

    # Change units into erg/s  ###here ???
    if (photmod_sfr == 'gutkin16'):
        # Units: Lbolsun per unit SFR(Msun/yr) for 10^8yr, assuming Chabrier
        sfr = np.zeros(shape=np.shape(lssfr))
        for comp in range(ncomp):
            sfr[:,comp] = 10**(lms[:,comp]+lssfr[:,comp])
            nebline_sfr[comp,:,:] = nebline_sfr[comp,:,:]*c.Lbolsun*sfr[:,comp]

    if verbose:
        print(' Emission lines calculated.')
            
    if att:
        att_param = io.read_data(infile,cut,
                                 inputformat=inputformat,
                                 params=att_params,
                                 testing=testing,
                                 verbose=verbose)

        nebline_sfr_att, coef_sfr_att = attenuation(nebline_sfr, att_param=att_param, 
                                      att_ratio_lines=att_ratio_lines,
                                                    redshift=redshift,h0=h0,
                                      origin='sfr',
                                      cut=cut, attmod=attmod, photmod=photmod_sfr,verbose=verbose)
        
        if verbose:
            print(' Attenuation calculated.')
    else:
        nebline_sfr_att = np.array(None)

    if flux:
        fluxes_sfr = calculate_flux(nebline_sfr,outfile,origin='sfr')
        fluxes_sfr_att = calculate_flux(nebline_sfr_att,outfile,origin='sfr')
        if verbose:
            print(' Flux calculated.')
    else:
        fluxes_sfr = np.array(None)
        fluxes_sfr_att = np.array(None)

    # Read any extra parameters and write them in the output file
    # together with the HII spectral lines
    extra_param = io.read_data(infile,cut,
                               inputformat=inputformat,
                               params=extra_params,
                               testing=testing,
                               verbose=verbose)
        
    io.write_sfr_data(outfile,lms,lssfr,lu_o_sfr,lnH_o_sfr,lzgas_o_sfr,
                      nebline_sfr,nebline_sfr_att,
                      fluxes_sfr,fluxes_sfr_att,
                      extra_param=extra_param,
                      extra_params_names=extra_params_names,
                      extra_params_labels=extra_params_labels,
                      verbose=verbose)
    
    del lu_sfr, lnH_sfr
    del lu_o_sfr, lnH_o_sfr, lzgas_o_sfr
    del nebline_sfr, nebline_sfr_att

    #----------------NLR AGN calculation------------------------
    if AGN:
        # Get total mass for Z corrections 
        lm_tot = components2tot(lms)

        # Get the central metallicity
        if Z_correct_grad:
            lzgas_agn1 = get_zgasagn(infile,Zgas_NLR,selection=cut,inoh=inoh,
                                    Z_correct_grad=True,lm_tot=lm_tot,
                                    inputformat=inputformat,
                                    testing=testing,verbose=verbose)
        else:
            lzgas_agn1 = get_zgasagn(infile,Zgas_NLR,selection=cut,
                                    inoh=inoh,inputformat=inputformat,
                                    testing=testing,verbose=verbose)

        ###here Duplicate lzgas_agn to the SFR components at the moment
        lzgas_agn = np.repeat(lzgas_agn1[:, np.newaxis], ncomp, axis=1)
        ###here to be removed once the NLR is consistently done on 1 component
        ####here to be removed from here
        #Lagn_param = io.read_data(infile,cut,
        #                          inputformat=inputformat,
        #                          params=Lagn_params,
        #                          testing=testing,
        #                          verbose=verbose)        
        #if ncomp>1:
        #    bursttobulge(lms, Lagn_param)
        ####here to be removed until here (affecting to tremonti aprox)
        
        
        agn_nH_param = None
        if une_agn_nH is not None:
            agn_nH_param = io.get_data_agnnH(infile,une_agn_nH[1],
                                             agn_nH_params,selection=cut,
                                             h0=h0,units_h0=units_h0,
                                             inputformat=inputformat,
                                             testing=testing,verbose=verbose)
            
        Lagn = get_Lagn(infile,cut,inputformat=inputformat,
                        params=Lagn_params,AGNinputs=AGNinputs,
                        h0=h0,units_h0=units_h0,
                        units_Gyr=units_Gyr,units_L40h2=units_L40h2,
                        testing=testing,verbose=verbose)
        
        Q_agn, lu_agn, lnH_agn, epsilon_agn, ng_ratio = \
            get_une_agn(lms,lssfr,lzgas_agn, outfile,
                        Lagn=Lagn, T=T,
                        agn_nH_param=agn_nH_param,
                        une_agn_nH=une_agn_nH,
                        une_agn_spec=une_agn_spec,
                        une_agn_U=une_agn_U,verbose=verbose)

        if verbose:
            print('AGN:')
            print(' U and nH calculated.')
        
        lu_o_agn = np.copy(lu_agn)
        lnH_o_agn = np.copy(lnH_agn)
        lzgas_o_agn = np.copy(lzgas_agn) 

        nebline_agn = get_lines(lu_agn.T,lnH_agn.T,lzgas_agn.T,photmod=photmod_agn,
                                xid_phot=xid_agn,alpha_phot=alpha_agn,
                                verbose=verbose)
        
        # Change units into erg/s
        if (photmod_agn == 'feltre16'):
            # Units: erg/s for a central Lacc=10^45 erg/s
            nebline_agn[0] = nebline_agn[0]*Lagn/1e45
       
        if verbose:
            print(' Emission lines calculated.')

        if att:
            nebline_agn_att, coef_agn_att = attenuation(nebline_agn, att_param=att_param, 
                                                        att_ratio_lines=att_ratio_lines,redshift=redshift,h0=h0,
                                          origin='agn',
                                          cut=cut, attmod=attmod, photmod=photmod_agn,verbose=verbose)
            if verbose:
                print(' Attenuation calculated.')     
        else:
            nebline_agn_att = np.array(None)
            
        if flux:
            fluxes_agn = calculate_flux(nebline_agn,outfile,origin='agn')
            fluxes_agn_att = calculate_flux(nebline_agn_att,outfile,origin='agn')
            if verbose:
                print(' Flux calculated.')
        else:
            fluxes_agn = np.array(None)
            fluxes_agn_att = np.array(None)

        io.write_agn_data(outfile,Lagn,
                          lu_o_agn,lnH_o_agn,lzgas_o_agn,
                          nebline_agn,nebline_agn_att,
                          fluxes_agn,fluxes_agn_att,
                          epsilon_agn,
                          verbose=verbose)             
        del lu_agn, lnH_agn, lzgas_agn 
        del lu_o_agn, lnH_o_agn, lzgas_o_agn
        del nebline_agn, nebline_agn_att


    del lms, lssfr, cut
    
    if verbose:
        print('* Total time: ', round(time.perf_counter() - start_total_time,2), 's.')
