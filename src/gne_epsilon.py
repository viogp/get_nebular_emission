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


def surface_density_disc(R,MD,hD,profile='exponential',verbose=True):
    '''
    Calculate the surface density at radius R, for a disc
    with mass MD, and scalelength hD.

    Parameters
    ----------
    R : float
       Radius at which to calculate the surface density (Mpc).
    MD : array of floats
       Mass of the disc (Msun).
    hD : array of floats
       Scalelength of the disc (Mpc)
    profile : string
       Assumed density profile type, 'exponential'.
    verbose : boolean
       If True print out messages.
     
    Returns
    -------
    surface_density : array of float (kg/m^2)
    '''
    surface_density = np.zeros(MD.shape)
    
    ind = np.where((MD>0)&(hD>0))[0]
    a = R[0]/hD[ind]
    if len(R) > 1:
        a = R[ind]/hD[ind]
    
    if profile not in 'exponential':
        if verbose:
            print('WARNING gne_model_UnH is only set to handle')
            print('                      exponential profiles.')
        profile == 'exponential'

    m_kg = MD[ind]*c.Msun
    h_m = hD[ind]*c.mega*c.parsec
    surface_density[ind] = np.exp(-a)*m_kg/(2*np.pi*h_m*h_m)
    
    return surface_density #kg/m^2



def enclosed_mass(R,M,h,mgasr_type='disc',verbose=True):
    '''
    Calculate the mass enclosed within a radius x, for a
    disc or a bulge component, with a total mass M,
    and scalelength hD.

    Parameters
    ----------
    R : float
       Radius at which to calculate the enclosed mass (Mpc).
    M : array of floats
       Total mass of the component (Msun).
    h : array of floats
       Scalelength of the component (Mpc)
    mgasr_type : string
       'disc', 'bulge' or None
    verbose : boolean
       If True print out messages.
     
    Returns
    -------
     mass_enclosed : array of floats (same units as M)
    '''
    mass_enclosed = np.zeros(M.shape)    
    ind = np.where((M>0)&(h>0))[0]
    a = R[0]/h[ind]
    if len(R) > 1:
        a = R[ind]/h[ind]

    if mgasr_type == 'bulge':
        mass_enclosed[ind] = M[ind] - 0.5*M[ind]*np.exp(-a)*(a*a+2*a+2)
    else:
        mass_enclosed[ind] = M[ind]*(1 - np.exp(-a)*(1+a))

    return mass_enclosed 


def number_density(R,M,h,mgasr_type='disc',verbose=True):
    '''
    Calculate the number density (cm^-3) within a radius R,
    for either a disc or a bulge, with mass M and
    scalelength h.

    Parameters
    ----------
    R : array of floats
       Radius at which number densities will be calculated (Mpc).
    M : array of floats (or None)
       Mass of the galaxy component (Msun)
    h : array of floats (or None)
       Scalelenght of the component (Mpc)
    mgasr_type : string
       'disc', 'bulge' or None
    verbose : boolean
       If True print out messages.
     
    Returns
    -------
    n : array of floats (cm^-3)
    '''

    M_enclosed = enclosed_mass(R,M,h,mgasr_type=mgasr_type,verbose=verbose)

    n = np.zeros(M_enclosed.shape)
        
    if len(R) > 1:
        ind = np.where((M_enclosed>0)&(R>0))[0]
        R_cm = R[ind]*c.Mpc_to_cm
    else:
        ind = np.where((M_enclosed>0))[0]
        R_cm = R[0]*c.Mpc_to_cm
    
    n[ind] = (M_enclosed[ind]*c.Msun/c.mp)/st.vol_sphere(R_cm)

    return n #cm^-3


def number_density_hydro_eq(max_r,M,r_hm,verbose=True):
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


#ni[ind] = number_density_hydro(rmax,Mi[ind],hi[ind],
#                               mgasr_type=mgasr_type[ii],
#                               profile=profile,verbose=verbose)
def number_density_hydro(max_r,M,r_hm,verbose=True):
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
            ng[i] = number_density_hydro_eq(max_r[i],M[i],r_hm[i],profile=profile,verbose=verbose)
    else:
        for i in range(len(M)):
            ng[i] = number_density_hydro_eq(max_r[0],M[i],r_hm[i],profile=profile,verbose=verbose)
            
    return ng # cm^-3
        

def calculate_epsilon(mgas,hr,filenom,rmax=[c.radius_NLR],nH=c.nH_NLR_cm3,
                      mgasr_type=None,n_model='simple',
                      pressure=None,verbose=True):
    '''
    Calculate the volume filling-factor within a radius, rmax, for
    a galaxy component with mass mgas, and scalelength hr.

    Parameters
    ----------
    mgas : array of floats (or None)
       Central gas mass or per galaxy component (Msun)
    hr : array of floats (or None)
       Scalelenght of the central region or per component (Mpc)
    filenom : string
        File with information relevant for the calculation
    rmax : array of floats
       Radius at which number densities will be calculated (Mpc).
    nH : float
       Assumed hydrogen density in the ionizing regions.
    mgasr_type : list of strings per component
       'disc', 'bulge' or None
    nmodel : list of strings
       'simple', 'hydro' or None
    pressure : string
       If True make the calculation from the hydrostatic pressure
    verbose : boolean
       If True print out messages.
     
    Returns
    -------
    epsilon : array of floats
    '''
    epsilon = np.zeros(mgas.shape[1])
    
    ncomp = io.get_ncomponents(mgas)
    for ii in range(ncomp):
        Mi = mgas[ii,:]
        hi = hr[ii,:]
    
        ni = np.zeros(Mi.shape)
        epi = np.zeros(Mi.shape)
    
        ind = np.where((Mi>0)&(hi>0))
        if len(rmax) > 1:
            rmax = rmax[ind]

        if pressure is None:
            ni[ind] = number_density(rmax,Mi[ind],hi[ind],mgasr_type=mgasr_type[ii],
                                     verbose=verbose)
        else: ###here needs to be tested
            ni[ind] = number_density_hydrop(rmax,Mi[ind],hi[ind],mgasr_type=mgasr_type[ii],
                                            verbose=verbose)

        epi[ind] = ni[ind]/nH
    
        epsilon[ind] = epsilon[ind] + epi[ind]
    
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
    

