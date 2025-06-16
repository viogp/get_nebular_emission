"""
.. moduleauthor:: Violeta Gonzalez-Perez <violetagp@protonmail.com>
.. contributions:: Olivia Vidal <ovive.pro@gmail.com>
.. contributions:: Julen Expósito-Márquez <expox7@gmail.com>
"""
import sys
import numpy as np
import h5py
import src.gne_const as c
import src.gne_io as io
import src.gne_stats as st

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


def surface_density(x,M,reff,profile='exponential',verbose=True):
    '''
    Given the mass of a disk, M, its effective radius, reff,
    the surface density is calculated at a given distance from the center, x.

    Parameters
    ----------
    x : array of floats
     Distance to the center in which surface density is going to be calculated (Mpc).
    M : array of floats
     Mass of the desired component of the galaxy (Msun).
    reff : floats
     Effective radius of the galaxy (Mpc)
    profile : string
     Assumed density profile form for the surface density.
    verbose : boolean
     If True print out messages.
     
    Returns
    -------
    surf_den : floats
    '''
    
    profiles = ['exponential']
    
    if profile not in profiles:
        if verbose:
            print('STOP (gne_model_UnH): Unrecognised profile for the surface density.')
            print('                Possible profiles= {}'.format(profiles))
        sys.exit()
    elif profile=='exponential':
        central = M/(2*np.pi*(reff**2))
        
        surf_den = central*np.exp(-x/reff)
        return surf_den

    
def enclosed_mass_disk(x,M,reff,profile='exponential',verbose=True):
    '''
    Given the mass of the desired component of the galaxy, the disk effective radius
    and a distance to the center, it calculates the surface density at that distance.

    Parameters
    ----------
    x : floats
     Distance to the center in which surface density is going to be calculated (Mpc).
    M : floats
     Mass of the desired component of the galaxy (Msun).
    reff : floats
     Effective radius of the galaxy (Mpc)
    profile : string
     Assumed density profile form for the surface density.
    verbose : boolean
     If True print out messages.
     
    Returns
    -------
    surf_den : floats
    '''
    
    profiles = [profile]
    
    if profile not in profiles:
        if verbose:
            print('STOP (gne_model_UnH): Unrecognised profile for the surface density.')
            print('                Possible profiles= {}'.format(profiles))
        sys.exit()
    elif profile=='exponential':
        ind = np.where((M>1e-5)&(reff>1e-5))[0]  ###here limitis to 0
        mass_enclosed = np.zeros(M.shape)
        if len(x) > 1:
            mass_enclosed[ind] = (M[ind]/reff[ind])*(reff[ind] - np.exp(-x[ind]/reff[ind])*reff[ind] - x[ind]*np.exp(-x[ind]/reff[ind]))
        else:
            mass_enclosed[ind] = (M[ind]/reff[ind])*(reff[ind] - np.exp(-x[0]/reff[ind])*reff[ind] - x[0]*np.exp(-x[0]/reff[ind]))
            
        return mass_enclosed
    

    
def enclosed_mass_sphere(x,M,reff,profile='exponential',verbose=True):
    '''
    Given the mass of the desired component of the galaxy, the disk effective radius
    and a distance to the center, it calculates the surface density at that distance.

    Parameters
    ----------
    x : floats
     Distance to the center in which surface density is going to be calculated (Mpc).
    M : floats
     Mass of the desired component of the galaxy (Msun).
    reff : floats
     Effective radius of the galaxy (Mpc)
    profile : string
     Assumed density profile form for the surface density.
    verbose : boolean
     If True print out messages.
     
    Returns
    -------
    surf_den : floats
    '''
    
    profiles = ['exponential']
    
    if profile not in profiles:
        if verbose:
            print('STOP (gne_model_UnH): Unrecognised profile for the surface density.')
            print('                Possible profiles= {}'.format(profiles))
        sys.exit()
    elif profile=='exponential':
        ind = np.where((M>1e-5)&(reff>1e-5))[0] ###here limits, shouldn't be 0?
        mass_enclosed = np.zeros(M.shape)
        if len(x) > 1:
            mass_enclosed[ind] = (M[ind]/(2*reff[ind]**3))*(2*reff[ind]**3 - np.exp(-x[ind]/reff[ind])*reff[ind]*(2**reff[ind]**2 + 2*reff[ind]*x[ind] + x[ind]**2))
        else:
            mass_enclosed[ind] = (M[ind]/(2*reff[ind]**3))*(2*reff[ind]**3 - np.exp(-x[0]/reff[ind])*reff[ind]*(2**reff[ind]**2 + 2*reff[ind]*x[0] + x[0]**2)) ###here different eqs from mine
        
        return mass_enclosed

    
def vol_sphere(r):
    '''
    Given the radius of a sphere, returns its value.

    Parameters
    ----------
    r : float
     Radius of the sphere.
     
    Returns
    -------
    V : float
    '''
    
    V = (4./3.)*np.pi*r**3
    return V


def mean_density(x,M,r_hm,profile='exponential',bulge=False,verbose=True):
    '''
    Given the mass of the desired component of the galaxy, the disk effective radius
    and a distance to the center, it calculates the particle density at that distance.

    Parameters
    ----------
    x : floats
     Distance to the center in which surface density is going to be calculated (Mpc).
    Ms : floats
     Stellar mass of the galaxy (Msun).
    Mg : floats
     Cold gas mass of the galaxy (Msun).
    r_hm : floats
     Half-mass radius of the galaxy (Mpc)
    profile : string
     Assumed density profile form for the surface density.
    bulge : boolean
     True if the calculation is being applied to a bulge.
     False if the calculation is being applied to a disk.
    verbose : boolean
     If True print out messages.
     
    Returns
    -------
    n : floats
    '''
    
    profiles = ['exponential']
    
    if profile not in profiles:
        if verbose:
            print('STOP (gne_model_UnH): Unrecognised profile for the surface density.')
            print('                Possible profiles= {}'.format(profiles))
        sys.exit()
    elif profile=='exponential':
        reff = c.halfmass_to_reff*r_hm # GALFORM ###here not to be hardwired
        
        if bulge:
            M_enclosed = enclosed_mass_sphere(x,M,reff,profile=profile,verbose=verbose)
        else:
            M_enclosed = enclosed_mass_disk(x,M,reff,profile=profile,verbose=verbose)

        n = np.zeros(M_enclosed.shape)
        
        if len(x) > 1:
            ind = np.where((M_enclosed>0)&(x>0))[0]
            n[ind] = M_enclosed[ind]/vol_sphere(x[ind]) / (c.mp*c.kg_to_Msun*c.Mpc_to_cm**3)
        else:
            ind = np.where((M_enclosed>0))[0]
            n[ind] = M_enclosed[ind]/vol_sphere(x[0]) / (c.mp*c.kg_to_Msun*c.Mpc_to_cm**3)
        
        return n


def gamma_gas_func():
    '''
    Calculates the velocity dispersion of the gas component (see Lagos et. al. 2011).
     
    Returns
    -------
    gamma_gas : float
    '''
    gamma_gas = 10 #km s^-1, Lagos et al. 2011
    
    return gamma_gas


def gamma_star_func(h_star,den_star):
    '''
    Calculates the velocity disparsion of the star component (see Lagos et. al. 2011).
    
    Parameters
    ----------
    h_star : float
     Stellar scaleheight.
    den_star : float
     Stellar surface density.
     
    Returns
    -------
    gamma_gas : float
    '''
    
    gamma_star = np.sqrt(np.pi*c.G*h_star*den_star) # GALFORM
    
    return gamma_star
    

    
def particle_density(x,M,r_hm,T=10000,profile='exponential',verbose=True):
    '''
    Given the mass of the desired component of the galaxy, the disk effective radius
    and a distance to the center, it calculates the particle density at that distance.

    Parameters
    ----------
    x : floats
      Distance to the center in which surface density is going to be calculated (Mpc).
    Ms : floats
      Stellar mass of the galaxy (Msun).
    Mg : floats
      Cold gas mass of the galaxy (Msun).
    r_hm : floats
      Half-mass radius of the galaxy (Mpc)
    T : float
     Typical temperature of ionizing regions.
    profile : string
      Assumed density profile form for the surface density.
    verbose : boolean
      If True print out messages.
     
    Returns
    -------
    n : floats
    '''
    
    reff = c.halfmass_to_reff*r_hm 
    
    den_gas = surface_density(x,M,reff,profile=profile,verbose=verbose)
    
    # h_star = c.reff_to_scale_high*reff
    # den_star = surface_density(x,Ms,reff,profile=profile,verbose=verbose)
    # gamma_gas = gamma_gas_func()
    # gamma_star = gamma_star_func(h_star,den_star)
    # Pext = 0.5*np.pi*c.G*den_gas*(den_gas + (gamma_gas/gamma_star)*den_star) * 1e10/(c.Mpc_to_cm**2)
    
    Pext = 0.5*np.pi*c.G*den_gas**2 * 1e10/(c.Mpc_to_cm**2)
    
    # P = nkT
    n = Pext/(T*c.boltzmann) / c.Mpc_to_cm**3 # cm^-3
    
    return n


def mean_density_hydro_eq(max_r,M,r_hm,profile='exponential',verbose=True):
    '''
    Given the mass of the desired component of the galaxy, the disk effective radius
    and a distance to the center, it calculates the mean particle density within that distance.

    Parameters
    ----------
    max_r : floats
      Distance to the center within the surface density is going to be calculated (Mpc).
    Ms : floats
      Stellar mass of the galaxy (Msun).
    Mg : floats
      Cold gas mass of the galaxy (Msun).
    r_hm : floats
      Half-mass radius of the galaxy (Mpc)
    profile : string
      Assumed density profile form for the surface density.
    verbose : boolean
      If True print out messages.
     
    Returns
    -------
    n : floats
    '''
    n = 0
    
    if max_r>0:
        x = np.arange(0,max_r,max_r/100)
        for xbin in x:
            n += particle_density(xbin,M,r_hm,profile=profile,verbose=verbose)
        n = n/len(x)
        return n # cm^-3
    else:
        return n # cm^-3

    
def calculate_ng_hydro_eq(max_r,M,r_hm,profile='exponential',verbose=True):
    '''
    Given the mass of the desired component of the galaxy, the disk effective radius
    and a distance to the center, it calculates the mean particle density within that distance.

    Parameters
    ----------
    max_r : floats
      Distance to the center within the surface density is going to be calculated (Mpc).
    Ms : floats
      Stellar mass of the galaxy (Msun).
    Mg : floats
      Cold gas mass of the galaxy (Msun).
    r_hm : floats
      Half-mass radius of the galaxy (Mpc)
    profile : string
      Assumed density profile form for the surface density.
    verbose : boolean
      If True print out messages.
     
    Returns
    -------
    n : floats
    '''
    
    ng = np.zeros(M.shape)
    if len(max_r) > 1:
        for i in range(len(M)):
            ng[i] = mean_density_hydro_eq(max_r[i],M[i],r_hm[i],profile=profile,verbose=verbose)
    else:
        for i in range(len(M)):
            ng[i] = mean_density_hydro_eq(max_r[0],M[i],r_hm[i],profile=profile,verbose=verbose)
            
    return ng # cm^-3
        

def epsilon_simplemodel(max_r,Mg,r_hm,nH=1000,profile='exponential',bulge=False,verbose=True):
    '''
    Given the mass of the desired component of the galaxy, the disk effective radius
    and a distance to the center, it calculates the volume filling-factor within that distance.

    Parameters
    ----------
    max_r : floats
     Distance to the center within the surface density is going to be calculated (Mpc).
    Ms : floats
     Stellar mass of the galaxy (Msun).
    Mg : floats
     Cold gas mass of the galaxy (Msun).
    r_hm : floats
     Half-mass radius of the galaxy (Mpc).
    nH : float
     Assumed hydrogen density in the ionizing regions.
    profile : string
     Assumed density profile form for the surface density.
    bulge : boolean
     True if the calculation is being applied to a bulge.
     False if the calculation is being applied to a disk.
    verbose : boolean
     If True print out messages.
     
    Returns
    -------
    epsilon : floats
    '''
    
    n = mean_density(max_r,Mg,r_hm,profile=profile,bulge=bulge,verbose=verbose)
    epsilon = n/nH
    
    return n, epsilon


def calculate_epsilon(mgas,hr,max_r,filenom,nH=c.nH_NLR,
                      profile='exponential',verbose=True):
    '''
    It reads the relevant parameters in the input file and calculates 
    the volume filling-factor within that distance.

    Parameters
    ----------
    epsilon_param : array of floats
       Parameters for epsilon calculation.
    max_r : array of floats
       Distance to the center within the surface density is going to be calculated (Mpc).
    filenom : string
       File with output
    nH : float
     Assumed hydrogen density in the ionizing regions.
    profile : string
     Assumed density profile form for the surface density.
    verbose : boolean
     If True print out messages.
     
    Returns
    -------
    epsilon : array of floats
    '''
    ncomp = io.get_ncomponents(mgas)
    Mg = mgas[0,:]
    r = hr[0,:]
#    if epsilon_param.shape[0] == 2: #2
#        Mg, r = epsilon_param
    if ncomp == 1:
        # Mg = Mg + Mg_bulge
        ind_epsilon = np.where((Mg>5e-5)&(r>5e-5)) ###here why this arbitrary values?
        epsilon = np.zeros(Mg.shape)
        ng = np.zeros(Mg.shape)
        if len(max_r) > 1:
            max_r = max_r[ind_epsilon]
        ng[ind_epsilon], epsilon[ind_epsilon]=epsilon_simplemodel(max_r,
                                                                  Mg[ind_epsilon],r[ind_epsilon],nH=nH,verbose=verbose)
    else:
        #        Mg, r, Mg_bulge, r_bulge = epsilon_param
        Mg_bulge = mgas[1,:]
        r_bulge = hr[1,:]
        ind_epsilon = np.where((Mg>5e-5)&(r>5e-5))
        epsilon = np.zeros(Mg.shape)
        ng = np.zeros(Mg.shape)
        if len(max_r) > 1:
            max_r = max_r[ind_epsilon]
        ng_disk, ep_disk = epsilon_simplemodel(max_r,
                                               Mg[ind_epsilon],r[ind_epsilon],nH=nH,verbose=verbose)
        ng_bulge, ep_bulge = epsilon_simplemodel(max_r,
                                                 Mg_bulge[ind_epsilon],r_bulge[ind_epsilon],nH=nH,
                                                 bulge=True,verbose=verbose)
        epsilon[ind_epsilon]= ep_disk + ep_bulge
        ng[ind_epsilon]= ng_disk + ng_bulge
    
    epsilon[epsilon>1] = 1
    return epsilon


def n_ratio(n,n_z0):
    '''
    Estimates the metallicity of the AGN from the global metallicity.

    Parameters
    ----------
    n : floats
     Particle number density.
    n_z0 : floats
     Particle number density of the galaxies from the sample at redshift 0.
     
    Returns
    -------
    ratio : floats
    '''
    
    ratio = np.full(n.shape,1.)
    ind = np.where((n>0)&(n_z0>0))[0]
    
    mean = np.mean(n[ind])
    mean_0 = np.mean(n_z0[ind])
    
    ratio = mean/mean_0
    
    return ratio
    


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
    #    
    #    int_S = np.power(nu1,3)/3. +\
    #        (np.power(nu2,0.5) - np.power(nu1,0.5))/0.5 +\
    #        (np.power(nu3,alpha+1) - np.power(nu2,alpha+1))/(alpha+1)
    #    
    #    int_SL = (np.power(nu1,2) - np.power(nuL,2))/2. -\
    #        (np.power(nu2,-0.5) - np.power(nu1,-0.5))/0.5 +\
    #        (np.power(nu3,alpha) - np.power(nu2,alpha))/alpha
    #    
        mask = Lagn > 0.
    #    Q[mask] = Lagn[mask]*Lagn[mask]*int_SL/(c.h_erg*int_S)
    #
        Q[mask] = -Lagn[mask]*(nul**alpha)/(alpha*8.66e-11*c.h_erg) ###here Julen's eq 
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

    # As nothing is stated in the paper, we assume this is U(Rs)
    output = lu
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
                lms_o, lssfr_o,lzgas_o,###here line to be removed when  adapted
                mgasr_type=None,verbose=True):
    '''
    Given the AGN bolometric luminosity,
    gas mass and scalelenght (per component)
    get the ionizing parameter, U, and the
    filling factor, epsilon, if needed.

    Parameters
    ----------
    Lagn : array of floats
       Bolometric luminosity of the AGNs (erg/s).
    mgas : array of floats (or None)
       Central gas mass or per galaxy component
    hr : array of floats (or None)
       Scalelenght of the central region or per component
    filenom : string
        File with information relevant for the calculation
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
        #Q = np.repeat(Q[np.newaxis,...], 2, axis=0).T ###here to be removed (match vectors)

        #Q = phot_rate_agn(lssfr=lssfr_o,lms=lms_o,Lagn=Lagn)
        #Q = get_Q_agn(Lagn)#,lssfr=lssfr_o,lms=lms_o)

        # Obtain the filling factor
        if (mgas is None or hr is None):
            epsilon = None
            nattrs = io.add2header(filenom,['epsilon_NLR'],[c.epsilon_NLR])
        else: ###here to check the epsilon calculation   
            epsilon = np.full(np.shape(lzgas_o)[0],c.epsilon_NLR)
            model_nH_agn = ['exponential','reff'] ###here to be removed
            if model_nH_agn is not None:
                epsilon = calculate_epsilon(mgas,hr,[c.radius_NLR],
                                            filenom,nH=c.nH_NLR,
                                            profile=model_nH_agn,
                                            verbose=verbose)
            #epsilon = st.ensure_2d(epsilon)

        # Calculate the ionising factor
        lu = get_U_panuzzo03(Q,filenom,epsilon=epsilon,origin='NLR') 
        lu = st.ensure_2d(lu)
        
    return lu, epsilon
