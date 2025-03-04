"""
.. moduleauthor:: Violeta Gonzalez-Perez <violetagp@protonmail.com>
.. contributions:: Olivia Vidal <ovive.pro@gmail.com>
.. contributions:: Julen Expósito-Márquez <expox7@gmail.com>
"""
import sys
import numpy as np
import h5py
import src.gne_const as c
from src.gne_stats import perc_2arrays
from src.gne_io import get_ncomponents

def alpha_B(T):
    '''
    Given the temperature of the ionizing region, it interpolates from Osterbrock & Ferland 2006
    tables the corresponding value of the hydrogen case B recombination coefficient.

    Parameters
    ----------
    T : floats
     Typical temperature considered for nebular regions.
     
    Returns
    -------
    alphaB : floats
    '''
    
    temps = [5000, 10000, 20000]
    values = [4.54e-13, 2.59e-13, 1.43e-13] # Osterbrock & Ferland 2006, Second Edition, page 22
    
    alphaB = np.interp(T,temps,values)
    
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
            print('STOP (gne_une): Unrecognised profile for the surface density.')
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
            print('STOP (gne_une): Unrecognised profile for the surface density.')
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
            print('STOP (gne_une): Unrecognised profile for the surface density.')
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
            print('STOP (gne_une): Unrecognised profile for the surface density.')
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


def calculate_epsilon(epsilon_param,max_r,filenom,nH=c.nH_agn,
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

    if epsilon_param.shape[0] == 2: #2
        Mg, r = epsilon_param
        # Mg = Mg + Mg_bulge
        ind_epsilon = np.where((Mg>5e-5)&(r>5e-5)) ###here why this arbitrary values?
        epsilon = np.zeros(Mg.shape)
        ng = np.zeros(Mg.shape)
        if len(max_r) > 1:
            max_r = max_r[ind_epsilon]
        ng[ind_epsilon], epsilon[ind_epsilon]=epsilon_simplemodel(max_r,
                                                                  Mg[ind_epsilon],r[ind_epsilon],nH=nH,verbose=verbose)
    else:
        Mg, r, Mg_bulge, r_bulge = epsilon_param
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


def phot_rate_agn(lssfr=None, lms=None, IMF=None, Lagn=None):
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
    ind = np.where(Lagn>0)[0]
    # Q[ind,0] = Lagn[ind]*2.3e10 # Panda 2022
    Q[ind,0] = Lagn[ind]*((3.28e15)**-1.7)/(1.7*8.66e-11*c.planck) # Feltre 2016
    # This implies that Lion = Lbol/5 aprox.
            
    return Q


def get_une_kashino20(lms1, lssfr1, lzgas, IMF=['Kroupa','Kroupa'],nhout=True):
    '''
    Characterise the SF ionising region from global galactic properties,
    using the model from
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
    ncomp = get_ncomponents(lms1.T)
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
    #    epsilon[:,comp][ind_comp[comp]] = ((1/alpha_B(T)) * ((4*c.c_cm*(10**lu[:,comp][ind_comp[comp]]))/3)**(3/2) * 
    #                          ((4*np.pi)/(3*Q[:,comp][ind_comp[comp]]*(10**lnH[:,comp][ind_comp[comp]])))**(1/2))
    #    
    #    if ng_ratio != None:
    #        epsilon[:,comp][ind_comp[comp]] = epsilon[:,comp][ind_comp[comp]] * ng_ratio
    #    
    #    cte[:,comp][ind_comp[comp]] = 3*(alpha_B(T)**(2/3)) * (3*epsilon[:,comp][ind_comp[comp]]**2*(10**lnH[:,comp][ind_comp[comp]])/(4*np.pi))**(1/3) / (4*c.c_cm)    
    #
    #lu[ind] = np.log10(cte[ind] * Q[ind]**(1/3))

    output = lu
    # As nothing is stated in the paper, we assume this is U(Rs)
    if nhout: output = lnH
    
    return output


def get_une_orsi14(lzgas, q0, z0, gamma):
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
    #    epsilon[:,comp][ind_comp[comp]] = ((1/alpha_B(T)) * ((4*c.c_cm*(10**lu[:,comp][ind_comp[comp]]))/3)**(3/2) * 
    #                          ((4*np.pi)/(3*Q[:,comp][ind_comp[comp]]*(10**lnH[:,comp][ind_comp[comp]])))**(1/2))
    #    
    #    if ng_ratio != None:
    #        epsilon[:,comp][ind_comp[comp]] = epsilon[:,comp][ind_comp[comp]] * ng_ratio
    #    
    #    cte[:,comp][ind_comp[comp]] = 3*(alpha_B(T)**(2/3)) * (3*epsilon[:,comp][ind_comp[comp]]**2*(10**lnH[:,comp][ind_comp[comp]])/(4*np.pi))**(1/3) / (4*c.c_cm)    
    #
    #lu[ind] = np.log10(cte[ind] * Q[ind]**(1/3))

    return lu

# def get_une_carton17(lms, lssfr, lzgas):
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


def get_une_panuzzo03_sfr(Q, lms, lssfr, lzgas, T, epsilon0, ng_ratio, origin, IMF):
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
            # lu, lnH, lzgas = get_une_orsi14(Q, lms, lssfr, lzgas, T, q0=c.q0_orsi, z0=c.Z0_orsi, gamma=1.3)
            lu, lnH, lzgas = get_une_kashino20(Q,lms,lssfr,lzgas_all,T,ng_ratio,IMF)
            
            for comp in range(len(Q[0])):
                epsilon[:,comp][ind_comp[comp]] = ((1/alpha_B(T)) * ((4*c.c_cm*(10**lu[:,comp][ind_comp[comp]]))/3)**(3/2) * 
                                      ((4*np.pi)/(3*Q[:,comp][ind_comp[comp]]*(10**lnH[:,comp][ind_comp[comp]])))**(1/2))
                
                if ng_ratio != None:
                    epsilon[:,comp][ind_comp[comp]] = epsilon[:,comp][ind_comp[comp]] * ng_ratio
                
                cte[:,comp][ind_comp[comp]] = 3*(alpha_B(T)**(2/3)) * (3*epsilon[:,comp][ind_comp[comp]]**2*(10**lnH[:,comp][ind_comp[comp]])/(4*np.pi))**(1/3) / (4*c.c_cm)    
            
            lu[ind] = np.log10(cte[ind] * Q[ind]**(1/3))
        
        if origin=='agn':
            lnH[ind] = 3
            lzgas[ind] = lzgas_all[ind]
            
            for comp in range(len(Q[0])):
                
                epsilon[:,comp][ind_comp[comp]] = epsilon0[ind_comp[comp]]
                
                cte[:,comp][ind_comp[comp]] = ( (3*(alpha_B(T)**(2/3)) / (4*c.c_cm)) 
                 * (3*epsilon[:,comp][ind_comp[comp]]**2*(10**lnH[:,comp][ind_comp[comp]])/(4*np.pi))**(1/3) )

            # Get the Ionising Parameters at the Stromgren Radius: Us~<U>/3
            cte[cte==0] = 1e-50
            lu[ind] = np.log10(cte[ind] * Q[ind]**(1/3) / 3)
            lu[cte==1e-50] = c.notnum
    

    return lu, lnH, lzgas


def get_une_panuzzo03(Q, lms, lssfr, lzgas, T, epsilon0, ng_ratio, origin, IMF):
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
            # lu, lnH, lzgas = get_une_orsi14(Q, lms, lssfr, lzgas, T, q0=c.q0_orsi, z0=c.Z0_orsi, gamma=1.3)
            lu, lnH, lzgas = get_une_kashino20(Q,lms,lssfr,lzgas_all,T,ng_ratio,IMF)
            
            for comp in range(len(Q[0])):
                epsilon[:,comp][ind_comp[comp]] = ((1/alpha_B(T)) * ((4*c.c_cm*(10**lu[:,comp][ind_comp[comp]]))/3)**(3/2) * 
                                      ((4*np.pi)/(3*Q[:,comp][ind_comp[comp]]*(10**lnH[:,comp][ind_comp[comp]])))**(1/2))
                
                if ng_ratio != None:
                    epsilon[:,comp][ind_comp[comp]] = epsilon[:,comp][ind_comp[comp]] * ng_ratio
                
                cte[:,comp][ind_comp[comp]] = 3*(alpha_B(T)**(2/3)) * (3*epsilon[:,comp][ind_comp[comp]]**2*(10**lnH[:,comp][ind_comp[comp]])/(4*np.pi))**(1/3) / (4*c.c_cm)    
            
            lu[ind] = np.log10(cte[ind] * Q[ind]**(1/3))
        
        if origin=='agn':
            lnH[ind] = np.log10(c.nH_agn)
            lzgas[ind] = lzgas_all[ind]
            
            for comp in range(len(Q[0])):
                
                epsilon[:,comp][ind_comp[comp]] = epsilon0[ind_comp[comp]]
                
                cte[:,comp][ind_comp[comp]] = ( (3*(alpha_B(T)**(2/3)) / (4*c.c_cm)) 
                 * (3*epsilon[:,comp][ind_comp[comp]]**2*(10**lnH[:,comp][ind_comp[comp]])/(4*np.pi))**(1/3) )

            # Get the Ionising Parameters at the Stromgren Radius: Us~<U>/3
            cte[cte==0] = 1e-50
            lu[ind] = np.log10(cte[ind] * Q[ind]**(1/3) / 3)
            lu[cte==1e-50] = c.notnum
    

    return lu, lnH, lzgas


def get_une_sfr(lms, lssfr, lzgas, filenom,
                q0=c.q0_orsi, z0=c.Z0_orsi, gamma=1.3, T=10000,
                ng_ratio=None, epsilon_param=[None], epsilon_param_z0=[None],epsilon=0.01,
                IMF=['Kroupa','Kroupa'],
                une_sfr_nH='kashino20',une_sfr_U='kashino20',verbose=True):
    '''
    Given the global properties of a galaxy or a region
    (log10(Mstar), log10(sSFR) and log10(Zgas)),
    characterise the HII region with its
    ionising parameter, U, and electron density, ne.

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
    une_sfr_nH : string
        Model to go from galaxy properties to Hydrogen (or e) number density.
    une_sfr_U : string
        Model to go from galaxy properties to ionising parameter.
    verbose : boolean
        True for printing out messages.

    Returns
    -------
    lu, lnH : floats
    '''

    # Read redshift
    f = h5py.File(filenom, 'r')
    header = f['header']
    redshift = header.attrs['redshift']
    f.close()
    
    # ncomp = len(lms[0])
    
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
    #        ng_ratio = c.med_to_low
    #    elif redshift==1.5:
    #        ng_ratio = c.high_to_low
    #    else:
    #        ng_ratio = 1.
                        
    if une_sfr_nH not in c.une_sfr_nH:
        if verbose:
            print('STOP (gne_une): Unrecognised model to get nH (HII).')
            print('                Possible options= {}'.format(c.une_sfr_nH))
        sys.exit()
    elif (une_sfr_nH == 'kashino20'):
        lnH = get_une_kashino20(lms,lssfr,lzgas,IMF,nhout=True)

    if une_sfr_U not in c.une_sfr_U:
        if verbose:
            print('STOP (gne_une): Unrecognised model to get U (HII).')
            print('                Possible options= {}'.format(c.une_sfr_nH))
        sys.exit()
    elif (une_sfr_U == 'kashino20'):
        lu = get_une_kashino20(lms,lssfr,lzgas,IMF,nhout=False)
    elif (une_sfr_U == 'orsi14'):
        lu = get_une_orsi14(lzgas,q0,z0,gamma)
    elif (une_sfr_U == 'panuzzo03_sfr'):
        Q = phot_rate_sfr(lssfr=lssfr,lms=lms,IMF=IMF)
        lu, lnH, lzgas = get_une_panuzzo03_sfr(Q,lms,lssfr,lzgas,T,epsilon,ng_ratio,'sfr',IMF)
        
    return lu, lnH # epsilon, ng_ratio


def get_une_agn(lms_o, lssfr_o, lzgas_o, filenom, agn_nH_param=None,
                Lagn=None, ng_ratio=None,IMF=['Kroupa','Kroupa'],
                T=10000, epsilon_param_z0=[None],
                une_agn_nH=None,une_agn_spec='feltre16',
                une_agn_U='panuzzo03', verbose=True):
    '''
    Given the global properties of a galaxy or a region
    (log10(Mstar), log10(sSFR) and 12+log(O/H)),
    get the characteristics of the ionising region
    (ionizing parameter, U, and the electron density, ne).

    Parameters
    ----------
    lms : floats
     Masses of the galaxies per component (log10(M*) (Msun)).
    lssfr : floats
     sSFR of the galaxies per component (log10(SFR/M*) (1/yr)).
    lzgas : floats
     Metallicity of the galaxies per component (log10(Z)).
    Lagn : floats
     Bolometric luminosity of the AGNs (erg/s).
    T : float
     Typical temperature of ionizing regions.
    agn_nH_param : floats
     Parameters for epsilon calculation.
    epsilon_param_z0 : floats
     Parameters for epsilon calculation in the sample of galaxies at redshift 0.
    epsilon : floats
     Volume filling-factor of the galaxy.
    une_agn_nH : string
        Profile assumed for the distribution of gas around NLR AGN.
    une_agn_spec : string
        Model for the spectral distribution for AGNs.
    verbose : boolean
     Yes = print out messages.

    Returns
    -------
    Q, lu, lnH, lzgas : floats
    '''

    # Read redshift
    f = h5py.File(filenom, 'r')
    header = f['header']
    redshift = header.attrs['redshift']
    f.close()
    
    # ncomp = len(lms[0])
    Q = phot_rate_agn(lssfr=lssfr_o,lms=lms_o,IMF=IMF,Lagn=Lagn)
    
    epsilon = np.full(np.shape(lzgas_o)[0],c.eNGC1976)
    if une_agn_nH is not None:
        epsilon = calculate_epsilon(agn_nH_param,[c.radius_NLR],
                                    filenom,nH=c.nH_agn,
                                    profile=une_agn_nH,verbose=verbose)

    if une_agn_U not in c.une_agn_U:
        if verbose:
            print('STOP (gne_une): Unrecognised model to get U (AGN).')
            print('                Possible options= {}'.format(c.une_agn_U))
        sys.exit()
    elif (une_agn_U == 'panuzzo03'):
        lu, lnH, lzgas = get_une_panuzzo03(Q,lms_o,lssfr_o,lzgas_o,T,epsilon,ng_ratio,'agn',IMF=IMF)
        
    return Q, lu, lnH, epsilon, ng_ratio
