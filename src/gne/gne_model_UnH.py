"""
.. moduleauthor:: Violeta Gonzalez-Perez <violetagp@protonmail.com>
.. contributions:: Olivia Vidal <ovive.pro@gmail.com>
.. contributions:: Julen Expósito-Márquez <expox7@gmail.com>
"""
import sys
import numpy as np
import h5py
import gne.gne_const as c
import gne.gne_io as io
import gne.gne_stats as st
import gne.gne_epsilon as e

def get_alphaB(T):
    
    '''
    Calculate the hydrogen case B recombination coefficient,
    for a given temperature of the ionising region.
    Interpolate the values saved in the constant file.

    Parameters
    ----------
    T : float
        Nebular region's temperature (K).
     
    Returns
    -------
    alphaB : float
    '''

    temps = list(c.alphaB.keys())
    vals = list(c.alphaB.values())
    
    alphaB = np.interp(T,temps,vals)
    return alphaB
  

def phot_rate_sfr(lssfr=None, lms=None, IMF=None, Lagn=None):
    '''
    Given log10(Mstar), log10(sSFR), log10(Z), Lagn and the assumed IMF,
    get the rate of ionizing photons in photon per second.

    Parameters
    ----------
    lssfr : floats
     sSFR of the galaxies per component (log10(SFR/M*) (1/yr)).
    lms : floats
     Masses of the galaxies per component (log10(M*) (Msun)).
    lzgas : floats
     Metallicity of the galaxies per component (log10(Z)).
    IMF : array of strings
     Assumed IMF for the input data of each component.
    Lagn : floats
     Bolometric luminosity of the AGN (Lsun).
     
    Returns
    -------
    Q : floats
    '''
    
    Q = np.zeros(np.shape(lssfr))
    for comp in range(Q.shape[1]):
        ###here ref. missing
        Q[:,comp] = 10**(lssfr[:,comp] + lms[:,comp]) * c.IMF_SFR[IMF[comp]] * c.phot_to_sfr_kenn
        # lssfr[:,comp] = np.log10(Q[:,comp]/(c.IMF_SFR[IMF[comp]] * c.phot_to_sfr_kenn)) - lms[:,comp]
    #Q[ind,0] = Lagn[ind]*((3.28e15)**-1.7)/(1.7*8.66e-11*c.planck) # Feltre 2016 (Julen's)               
    return Q


def get_Q_agn(Lagn,alpha,model_spec='feltre16',verbose=True):
    '''
    Obtain the rate of ionizing photons in photon per second, Q,
    given the bolometric luminosity and spectral index of the AGN.

    Parameters
    ----------
    Lagn : array of floats
       Bolometric luminosity of the AGNs (erg/s).
    alpha_NLR : array of floats
        Spectral index assumed for the AGN.
    model_spec_agn : string
        Model for the spectral distribution for AGNs.
    verbose : boolean
       If True print out messages.

    Returns
    -------
    Q : array of floats
    '''

    Q = np.zeros(np.shape(Lagn))
    nul = c.h_nul*c.eV/c.h   # Hz

    if model_spec not in c.model_spec_agn:
        if verbose:
            print('STOP (gne_model_UnH): Unrecognised spectral AGN model.')
            print('                Possible options= {}'.format(c.model_spec_agn))
        sys.exit()
        
    elif (model_spec == 'feltre16'):
        hnu = c.agn_spec_limits[model_spec]
        nu1 = hnu[0]*c.eV/c.h
        nu2 = hnu[1]*c.eV/c.h
        nu3 = hnu[2]*c.eV/c.h      

        a1a3   = (nu1**2.5)/(nu2**(0.5+alpha))
        a1_den = (nu1**3)/3 + 2*(nu1**2.5)*(nu2**0.5-nu1**0.5) +\
            a1a3*(nu3**(alpha+1) - nu2**(alpha+1))/(alpha+1)
        int_SL = (nu3**alpha - nul**alpha)/alpha

        mask = Lagn > 0.
        Q[mask] = (Lagn[mask]/c.h_erg)*(a1a3/a1_den)*int_SL
    return Q


def get_UnH_kashino20(lms1, lssfr1, lzgas, IMF=['Kroupa','Kroupa'],nhout=True):
    '''
    Calculate the ionising parameter, log10(Us), and
    electron density, log10(nH), for ionising region(s),
    connecting those to galactic properties with the model from
    Kashino & Inoue 2019 (https://arxiv.org/abs/1812.06939).

    Parameters
    ----------
    lms : array of floats
        Masses of the galaxies per component (log10(M*) (Msun)).
    lssfr : array of floats
        sSFR of the galaxies per component (log10(SFR/M*) (1/yr)).
    lzgas : array of floats
        Metallicity of the galaxies per component (log10(Z)).
    ng_ratio : floats
     Ratio between the mean particle number density of the cold gas of the 
     input sample and the sample at redshift 0.
    IMF : array of strings
        Assumed IMF for the input data of each component.
    nhout : bool
        True to output the hydrogen number density, False for the ionising parameter
    
    Returns
    -------
    lu or lnH : floats
    '''

    # Initialise vectors
    lms, lssfr, lu, lnH, loh4 = [np.full(np.shape(lms1), c.notnum) for i in range(5)]

    # In Kashino+2020 a Kroupa IMF is assumed
    ncomp = io.get_ncomponents(lms1.T)
    for i in range(ncomp):
        iimf = IMF[i]
        
        ind = np.where((lms1[:,i]>c.notnum) & (lssfr1[:,i]>c.notnum))
        if (np.shape(ind)[1]>1):
            lms[ind,i] = np.log10(c.IMF_M[iimf]/c.IMF_M['Kroupa']) + lms1[ind,i]
            lssfr[ind,i] = lssfr1[ind,i] + \
                np.log10(c.IMF_SFR[iimf]*c.IMF_M['Kroupa']/c.IMF_SFR['Kroupa']/c.IMF_M[iimf])

    # Perform calculation where there is adequate input data
    ind = np.where((lssfr > c.notnum)&(lms > c.notnum)&(lzgas > c.notnum))
    if (np.shape(ind)[1]>1):
        # Transform log10(Zgas) into 4+log10(O/H)
        loh4[ind] = lzgas[ind] - np.log10(c.zsunK20) + c.ohsun - 8. 

        # Apply equation Table 2
        lnH[ind] = 2.066 + 0.310*(lms[ind]-10.) + 0.492*(lssfr[ind] + 9.)

        # Eq. 12 (and Table 2) from Kashino & Inoue 2019
        lu[ind] =  -2.316 - 0.360*loh4[ind] - 0.292*lnH[ind] + 0.428*(lssfr[ind] + 9.)

    ###here EV: this is an application of Panuzzo, not Kashino
    #ind = np.where((lssfr > c.notnum) &
    #               (lms > 0) &
    #               (lzgas > c.notnum))
    #ind_comp = []   
    #for comp in range(len(Q[0])):
    #    ind_comp.append(np.where((lssfr[:,comp] > c.notnum) &
    #                   (lms[:,comp] > 0) &
    #                   (lzgas[:,comp] > c.notnum) &
    #                   (Q[:,comp] > 0))[0])
    #    
    #epsilon = np.full(np.shape(lssfr),c.notnum)
    #cte = np.zeros(np.shape(lssfr))
    #
    #for comp in range(len(Q[0])):
    #    epsilon[:,comp][ind_comp[comp]] = ((1/get_alphaB(T)) * ((4*c.c_cm*(10**lu[:,comp][ind_comp[comp]]))/3)**(3/2) * 
    #                          ((4*np.pi)/(3*Q[:,comp][ind_comp[comp]]*(10**lnH[:,comp][ind_comp[comp]])))**(1/2))
    #    
    #    if ng_ratio != None:
    #        epsilon[:,comp][ind_comp[comp]] = epsilon[:,comp][ind_comp[comp]] * ng_ratio
    #    
    #    cte[:,comp][ind_comp[comp]] = 3*(get_alphaB(T)**(2/3)) * (3*epsilon[:,comp][ind_comp[comp]]**2*(10**lnH[:,comp][ind_comp[comp]])/(4*np.pi))**(1/3) / (4*c.c_cm)    
    #
    #lu[ind] = np.log10(cte[ind] * Q[ind]**(1/3))

    # Assuming the ionising parameter from Kashino+202 to be U(Rs)
    output = lu
    ## Assuming the ionising parameter from Kashino+202 to be <U(R)>
    #output = lu - np.log10(3.) 
    if nhout: output = lnH
    
    return output


def get_UnH_orsi14(lzgas, q0, z0, gamma):
    '''
    Given log10(Zgas) and the values for the free parameters,
    get the ionizing parameter, logU, using the model from Orsi 2014.

    Parameters
    ----------
    lzgas : floats
     Metallicity of the galaxies per component (log10(Z)).
    q0 : float
     Ionization parameter constant to calibrate Orsi 2014 model for nebular regions. q0(z/z0)^-gamma
    z0 : float
     Ionization parameter constant to calibrate Orsi 2014 model for nebular regions. q0(z/z0)^-gamma
    gamma : float
     Ionization parameter constant to calibrate Orsi 2014 model for nebular regions. q0(z/z0)^-gamma

    Returns
    -------
    lu : floats
       As nothing is stated in the paper, we assume this is U(Rs)
    '''
    
    lu = np.full(np.shape(lzgas), c.notnum)

    ind = np.where(lzgas > c.notnum)
    if (np.shape(ind)[1]>0):
        lu[ind] = np.log10((q0*((10**lzgas[ind])/z0)**(-gamma)) /c.c_cm)

    ###here Evolution part to be done externally
    #ind = np.where((lssfr > c.notnum) &
    #               (lms > 0) &
    #               (lzgas > c.notnum))
    #ind_comp = []   
    #for comp in range(len(Q[0])):
    #    ind_comp.append(np.where((lssfr[:,comp] > c.notnum) &
    #                   (lms[:,comp] > 0) &
    #                   (lzgas[:,comp] > c.notnum) &
    #                   (Q[:,comp] > 0))[0])
    #    
    #epsilon = np.full(np.shape(lssfr),c.notnum)
    #cte = np.zeros(np.shape(lssfr))
    #
    #for comp in range(len(Q[0])):
    #    epsilon[:,comp][ind_comp[comp]] = ((1/get_alphaB(T)) * ((4*c.c_cm*(10**lu[:,comp][ind_comp[comp]]))/3)**(3/2) * 
    #                          ((4*np.pi)/(3*Q[:,comp][ind_comp[comp]]*(10**lnH[:,comp][ind_comp[comp]])))**(1/2))
    #    
    #    if ng_ratio != None:
    #        epsilon[:,comp][ind_comp[comp]] = epsilon[:,comp][ind_comp[comp]] * ng_ratio
    #    
    #    cte[:,comp][ind_comp[comp]] = 3*(get_alphaB(T)**(2/3)) * (3*epsilon[:,comp][ind_comp[comp]]**2*(10**lnH[:,comp][ind_comp[comp]])/(4*np.pi))**(1/3) / (4*c.c_cm)    
    #
    #lu[ind] = np.log10(cte[ind] * Q[ind]**(1/3))

    return lu

# def get_UnH_carton17(lms, lssfr, lzgas):
#     '''
#     Given log10(Mstar), log10(sSFR), log10(Z),
#     get the ionizing parameter, logU, and the electron density, logne,
#     using the model from Carton 2017.

#     Parameters
#     ----------
#     lms : floats
#      Masses of the galaxies per component (log10(M*) (Msun)).
#     lssfr : floats
#      sSFR of the galaxies per component (log10(SFR/M*) (1/yr)).
#     lzgas : floats
#      Metallicity of the galaxies per component (log10(Z)).

#     Returns
#     -------
#     lu, lnH, lzgas : floats
#     '''
    
#     lu, lnH = [np.full(np.shape(lms), c.notnum) for i in range(2)]

#     ind = np.where((lssfr > c.notnum) &
#                    (lms > 0) &
#                    (lzgas > c.notnum))
    
#     if (np.shape(ind)[1]>1):
#         lnH[ind] = 2.066 + 0.310*(lms[ind]-10) + 0.492*(lssfr[ind] + 9.)
#         lu[ind] = -0.8*np.log10(10**lzgas[ind]/c.zsun) - 3.58   

#     return lu, lnH, lzgas


def get_U_panuzzo03_sfr(Q, lms, lssfr, lzgas, T, epsilon0, ng_ratio, origin, IMF):
    '''
    Given the rate of ionizing photons, log10(Mstar), log10(sSFR), log10(Z),
    the assumed temperature for the ionizing regions, the volume filling-factor
    and the assumed IMF,
    get the ionizing parameter, logU, and the electron density, logne,
    using the model from Panuzzo 2003.

    Parameters
    ----------
    Q : floats
     Rate of ionizing photons (phot/s)
    lms : floats
     Masses of the galaxies per component (log10(M*) (Msun)).
    lssfr : floats
     sSFR of the galaxies per component (log10(SFR/M*) (1/yr)).
    lzgas : floats
     Metallicity of the galaxies per component (log10(Z)).
    T : float
     Typical temperature of ionizing regions.
    epsilon0 : floats
     Volume filling-factor of the galaxies.
    ng_ratio : floats
     Ratio between the mean particle number density of the cold gas of the 
     input sample and the sample at redshift 0.
    origin : string
     Source of the ionizing photons.
    IMF : array of strings
     Assumed IMF for the input data of each component.

    Returns
    -------
    lu, lnH, lzgas : floats
    '''
    
    lzgas_all = np.copy(lzgas)
    
    lu, lnH, lzgas = [np.full(np.shape(lms), c.notnum) for i in range(3)]

    ind = np.where((lssfr > c.notnum) &
                   (lms > 0) &
                   (lzgas_all > c.notnum) &
                   (Q > 0))
    
    # ind1 = np.where((lssfr[:,0] > c.notnum) &
    #                (lms[:,0] > 0) &
    #                (lzgas[:,0] > c.notnum) &
    #                (Q[:,0] > 0))[0]
    # ind2 = np.where((lssfr[:,1] > c.notnum) &
    #                (lms[:,1] > 0) &
    #                (lzgas[:,1] > c.notnum) &
    #                (Q[:,1] > 0))[0]
    
    # ind_comp = [ind1,ind2]
    
    ind_comp = []   
    for comp in range(len(Q[0])):
        ind_comp.append(np.where((lssfr[:,comp] > c.notnum) &
                       (lms[:,comp] > 0) &
                       (lzgas_all[:,comp] > c.notnum) &
                       (Q[:,comp] > 0))[0])
    
    if (np.shape(ind)[1]>1):
        
        epsilon = np.full(np.shape(lssfr),c.notnum)
        cte = np.zeros(np.shape(lssfr))
        
        if origin=='sfr':
            for comp in range(len(Q[0])):
                epsilon[:,comp][ind_comp[comp]] = ((1/get_alphaB(T)) * ((4*c.c_cm*(10**lu[:,comp][ind_comp[comp]]))/3)**(3/2) * 
                                      ((4*np.pi)/(3*Q[:,comp][ind_comp[comp]]*(10**lnH[:,comp][ind_comp[comp]])))**(1/2))
                
                if ng_ratio != None:
                    epsilon[:,comp][ind_comp[comp]] = epsilon[:,comp][ind_comp[comp]] * ng_ratio
                
                cte[:,comp][ind_comp[comp]] = 3*(get_alphaB(T)**(2/3)) * (3*epsilon[:,comp][ind_comp[comp]]**2*(10**lnH[:,comp][ind_comp[comp]])/(4*np.pi))**(1/3) / (4*c.c_cm)    
            
            lu[ind] = np.log10(cte[ind] * Q[ind]**(1/3))
        
        if origin=='agn':
            lnH[ind] = 3
            lzgas[ind] = lzgas_all[ind]
            
            for comp in range(len(Q[0])):
                
                epsilon[:,comp][ind_comp[comp]] = epsilon0[ind_comp[comp]]
                
                cte[:,comp][ind_comp[comp]] = ( (3*(get_alphaB(T)**(2/3)) / (4*c.c_cm)) 
                 * (3*epsilon[:,comp][ind_comp[comp]]**2*(10**lnH[:,comp][ind_comp[comp]])/(4*np.pi))**(1/3) )

            # Get the Ionising Parameters at the Stromgren Radius: Us~<U>/3
            cte[cte==0] = 1e-50
            lu[ind] = np.log10(cte[ind] * Q[ind]**(1/3) / 3)
            lu[cte==1e-50] = c.notnum
    

    return lu, lnH, lzgas


def get_U_panuzzo03(Q,filenom,epsilon=None,nH=None, origin='NLR'):
    '''
    Calculate the ionising parameter, as log10(Us), from Eq.B.6, Panuzzo+2003.
    This requires as input the rate of ionising photons, Q,
    the filling factor, epsilon, the electron density, nH,
    and the temperature of the ionising region, read from the given file.

    Parameters
    ----------
    Q : array of floats
        Rate of ionizing photons (photons/s)
    filenome : string
        Name of the file with relevant information
    epsilon : array of floats (or None)
        Volume filling-factor of the ionising region
    nH : array of floats (or None)
        Hydrogen or electron density (cm^-3)
    origin : string
        Type of ionising region.

    Returns
    -------
    lu : floats
    '''
    # Read the relevant ionising region constants
    f = h5py.File(filenom, 'r')
    header = f['header']
    temp = header.attrs['T_'+origin+'_K']
    # If epsilon and nH not provided, get constant values
    if epsilon is None:
        epsilon = header.attrs['epsilon_'+origin]
    if nH is None:
        nH = header.attrs['nH_'+origin+'_cm3']
    f.close()

    # Obtain the recombination coefficient
    alphaB = get_alphaB(temp)

    # Calculate the constant part of the equation
    const = (3/(4*c.c_cm))*np.power(3*alphaB*alphaB/(4*np.pi),1/3)
    
    # Calculate the average ionising parameter
    uu = const*np.power(Q*nH*epsilon*epsilon,1/3)

    # Get the Ionising Parameters at the Stromgren Radius: Us~<U>/3
    lu = np.zeros(uu.shape); lu.fill(c.notnum)
    lu[uu>0.] = np.log10(uu[uu>0.]) - np.log10(3)

    return lu


def get_UnH_sfr(lms, lssfr, lzgas, filenom,
                q0=c.q0_orsi, z0=c.Z0_orsi, gamma=1.3, T=10000,
                ng_ratio=None, epsilon_param=[None], epsilon_param_z0=[None],epsilon=0.01,
                IMF=['Kroupa','Kroupa'],
                model_nH_sfr='kashino20',model_U_sfr='kashino20',verbose=True):
    '''
    Calculate the ionising parameter, log10(Us),
    electron density, log10(nH), and
    filling factor, epsilon, if required,
    for HII reions, given global properties of galaxy regions,
    (log10(Mstar), log10(sSFR) and log10(Zgas)),

    Parameters
    ----------
    lms : array of floats
        Masses of the galaxies per component (log10(M*) (Msun)).
    lssfr : array of floats
        sSFR of the galaxies per component (log10(SFR/M*) (1/yr)).
    lzgas : array of floats
        Metallicity of the galaxies per component (log10(Z)).
    q0 : float
        Ionization parameter constant for Orsi 2014 model: q=q0(z/z0)^-gamma
    z0 : float
        Metallicity constant for Orsi 2014 model: q=q0(z/z0)^-gamma
    gamma : float
        Exponent for Orsi 2014 model: q=q0(z/z0)^-gamma
    T : float
        Typical temperature of HII ionizing regions.
    epsilon_param : floats
     Parameters for epsilon calculation.
    epsilon_param_z0 : floats
     Parameters for epsilon calculation in the sample of galaxies at redshift 0.
    epsilon : floats
     Volume filling-factor of the galaxy.
    IMF : array of strings
     Assumed IMF for the input data of each component.
    model_nH_sfr : string
        Model to go from galaxy properties to Hydrogen (or e) number density.
    model_U_sfr : string
        Model to go from galaxy properties to ionising parameter.
    verbose : boolean
        True for printing out messages.

    Returns
    -------
    lu, lnH, epsilon : array of floats
    '''

    ## Read redshift
    #f = h5py.File(filenom, 'r')
    #header = f['header']
    #redshift = header.attrs['redshift']
    #f.close()
    #
    ## ncomp = len(lms[0])
    #
    #epsilon = None
    #if epsilon_param_z0 is not None:
    #    # ng = calculate_ng_hydro_eq(2*epsilon_param[1],epsilon_param[0],epsilon_param[1],profile='exponential',verbose=True)
    #    # epsilon = ng/c.nH_gal
    #    # epsilon[epsilon>1] = 1
    #        
    #    
    #    # ng_z0 = calculate_ng_hydro_eq(2*epsilon_param_z0[1],epsilon_param_z0[0],epsilon_param_z0[1],profile='exponential',verbose=True)
    #    # ng_ratio = n_ratio(ng,ng_z0)
    #    if redshift==0.8:
    #        ng_ratio = 1.74
    #    elif redshift==1.5:
    #        ng_ratio = 1.58
    #    else:
    #        ng_ratio = 1.
                        
    if model_nH_sfr not in c.model_nH_sfr:
        if verbose:
            print('STOP (gne_model_UnH): Unrecognised model to get nH (HII).')
            print('                Possible options= {}'.format(c.model_nH_sfr))
        sys.exit()
    elif (model_nH_sfr == 'kashino20'):
        lnH = get_UnH_kashino20(lms,lssfr,lzgas,IMF,nhout=True)

    epsilon = None
    if model_U_sfr not in c.model_U_sfr:
        if verbose:
            print('STOP (gne_model_UnH): Unrecognised model to get U (HII).')
            print('                Possible options= {}'.format(c.model_nH_sfr))
        sys.exit()
    elif (model_U_sfr == 'kashino20'):
        lu = get_UnH_kashino20(lms,lssfr,lzgas,IMF,nhout=False)
    elif (model_U_sfr == 'orsi14'):
        lu = get_UnH_orsi14(lzgas,q0,z0,gamma)
    ####here to be adapted to new function
    #elif (model_U_sfr == 'panuzzo03_sfr'):
    #    Q = phot_rate_sfr(lssfr=lssfr,lms=lms,IMF=IMF)
    #    lu, lnH, lzgas = get_U_panuzzo03_sfr(Q,lms,lssfr,lzgas,T,epsilon,ng_ratio,'sfr',IMF)
        
    return lu, lnH #, epsilon


def get_UnH_agn(Lagn, mgas, hr, filenom,
                mgasr_type=None,verbose=True):
    '''
    Calculate the ionising parameter, U, and the
    filling factor, epsilon, if needed, for a given
    AGN bolometric luminosity, Lagn, and gas mass, mgas,
    and scalelenght, hr, that can be given per component.

    Parameters
    ----------
    Lagn : array of floats
       Bolometric luminosity of the AGNs (erg/s).
    mgas : array of floats (or None)
       Gas mass of a galaxy (component) (Msun).
    hr : array of floats (or None)
       Scalelenght of the galaxy (component) (Mpc).
    filenom : string
        File with information relevant for the calculation
    mgasr_type : list of strings per component
       'disc', 'bulge' or None
    verbose : boolean
       If True print out messages.

    Returns
    -------
    lu, epsilon : array of floats (or None)
    '''
    # Read model for the calculation of the ionising parameter
    f = h5py.File(filenom, 'r')
    header = f['header']
    model_U_agn = header.attrs['model_U_NLR']
    f.close()

    if model_U_agn not in c.model_U_agn:
        if verbose:
            print('STOP (gne_model_UnH): Unrecognised model to get U (AGN).')
            print('                Possible options= {}'.format(c.model_U_agn))
        sys.exit()
    elif (model_U_agn == 'panuzzo03'):
        # Calculate the number of ionising photons
        f = h5py.File(filenom, 'r')
        header = f['header']
        model_spec_agn = header.attrs['model_spec_NLR']
        alpha_NLR = header.attrs['alpha_NLR']
        f.close()

        Q = get_Q_agn(Lagn,alpha_NLR,model_spec=model_spec_agn,verbose=verbose)

        # Obtain the filling factor
        if (mgas is None or hr is None):
            epsilon = None
            nattrs = io.add2header(filenom,['epsilon_NLR'],[c.epsilon_NLR],
                                   verbose=verbose)
        else: 
            epsilon = e.calculate_epsilon(mgas,hr,filenom,
                                          rmax=[c.radius_NLR],nH=c.nH_NLR_cm3,
                                          mgasr_type=mgasr_type,
                                          verbose=verbose)
            epsilon = st.ensure_2d(epsilon)

        # Calculate the ionising factor
        lu = get_U_panuzzo03(Q,filenom,epsilon=epsilon,origin='NLR') 
        lu = st.ensure_2d(lu)
        
    return lu, epsilon
