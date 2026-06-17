import os
import numpy as np
import gne.gne_stats as st
import gne.gne_const as c

def get_obs_bpt(redshift,bpt):
    '''
    Get observational data for BPT diagrams at a given redshift
    
    Parameters
    ----------
    redshift : float
       Redshift of interest
    bpt: string
        Type of BPT diagram: 'NII' (OIII/Hbeta vs N2/Ha)
        or 'SII' (OIII/Hbeta vs S2/Ha)

    Returns
    -------
    xobs, yobs : array of floats
       Ratios for each observed spectral emission line
    obsdata : boolean
       True if there is any observational data at the given redshift
    '''

    xobs = -999.; yobs = -999.; obsdata = False

    # Use different data sets for different redshifts
    if redshift <= 0.2:
        obsdata = True
        obsfile = os.path.join(c.obs_data_dir,'favole2024.txt')
        l1,l2 = np.loadtxt(obsfile,skiprows=1,usecols=(15,9),unpack=True)
        xx, yy = [np.zeros(len(l1)) for i in range(2)]
        ind = np.where((l1>0.) & (l2>0.))
        if (np.shape(ind)[1]>0): #O3/Hb
            yy[ind] = np.log10(l1[ind]/l2[ind])
            
        if bpt=='NII': #N2/Ha
            l1,l2 = np.loadtxt(obsfile,skiprows=1,usecols=(18,6),unpack=True)
            ind = np.where((l1>0.) & (l2>0.))
            if (np.shape(ind)[1]>0):
                xx[ind] = np.log10(l1[ind]/l2[ind]) 
        elif bpt=='SII': #S2/Ha
            l1,l2 = np.loadtxt(obsfile,skiprows=1,usecols=(21,6),unpack=True)
            ind = np.where((l1>0.) & (l2>0.))
            if (np.shape(ind)[1]>0):
                xx[ind] = np.log10(l1[ind]/l2[ind]) 

    elif 1.45 <= redshift <= 1.75:
        obsdata = True
        if bpt=='NII':
            obsfile = os.path.join(c.obs_data_dir,'NII_Kashino.txt')
            yy = np.loadtxt(obsfile,skiprows=18,usecols=(6)) #O3/Hb
            xx = np.loadtxt(obsfile,skiprows=18,usecols=(3)) #N2/Ha
                
        elif bpt=='SII':
            obsfile = os.path.join(c.obs_data_dir,'SII_Kashino.txt')
            yy = np.loadtxt(obsfile,skiprows=18,usecols=(6)) #O3/Hb
            xx = np.loadtxt(obsfile,skiprows=18,usecols=(3)) #N2/Ha

    if obsdata:
        ind = np.where((xx>c.notnum) & (yy>c.notnum))
        if (np.shape(ind)[1]>0):
            xobs = xx[ind]
            yobs = yy[ind]
        else:
            obsdata = False

    return xobs,yobs,obsdata


def get_pozzetti(redshift,outpath=None):
    xobs = -999.; yobs = -999.; pozzetti_mod = False
    nomtab = 'pozzetti_tab3.txt'
    pozzetti_table = os.path.join(c.obs_data_dir,nomtab)
    nomtabMpc = nomtab.replace('tab3','tab3_Mpc')
    
    # Read table 3 form Pozzetti+2018
    data = np.loadtxt(pozzetti_table)
    zmin = data[:,0]
    zmax = data[:,1]
    nzz  = len(zmin)

    # Find if redshift within the intervals
    edges = np.insert(zmax,0,zmin[0])
    izz = st.locate_interval(redshift,edges,side='left')
    if (izz<0) or (izz>nzz-1):
        return xobs,yobs,pozzetti_mod
    pozzetti_mod = True

    # Check if Pozzeti's table in Mpc exists for this cosmology
    if outpath is None:
        opath = os.path.join(c.repo_dir, 'output')
    else:
        opath = outpath
    print(opath)    
    
    # Check if Pozzetti's table needs converting
    #if not outpath:
    #    outfile = os.path.join('output',pozzetti_table)
    #outfile = outfile
    #file_exists = io.check_file(outfile)
    #
    #if file_exists:
    #    data = np.loadtxt(pozzetti_table)
    #    m1 = data[izz,2:7]
    #    m2 = data[izz,7:12]
    #    m3 = data[izz,12:]
    return xobs,yobs,pozzetti_mod
