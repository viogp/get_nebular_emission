"""
.. moduleauthor:: Julen Expósito-Márquez <expox7@gmail.com>
.. contributions:: Violeta Gonzalez-Perez <violetagp@protonmail.com>
"""
import numpy as np
import src.gne_const as c
import src.gne_stats as st
import src.gne_io as io

def get_lzgas(zz,inoh=False):
    '''
    Get log10(Zgas=MZcold/Mcold) from input

    Parameters
    ----------
    zz : numpy array
        Input metallicities
    inoh : boolean
        If true, the input is assumed to be 12+log10(O/H), otherwise Zgas    

    Returns
    -------
    lzgas : array of floats
    '''
    
    lzgas = np.zeros(zz.shape); lzgas.fill(c.notnum)
    if inoh: 
        # Obtain log10(Zgas) from an input of 12+log10(O/H)
        lzgas = np.log10(c.zsun) - c.ohsun + zz
    else: 
        ind = np.where(zz>0)
        lzgas[ind] = np.log10(zz[ind])

    lzgas = st.ensure_2d(lzgas)
    return lzgas


def get_Ztremonti(logM,logZ,Lagn_param):
    # Ms and Z scale relation from Tremonti et. al. 2004    
    try:
        if logZ.shape[1] > 1:
            if Lagn_param[-1][0] != None:
                logMt = np.log10(10**logM[:,0] + Lagn_param[-1])
            else:
                logMt = np.log10(10**logM[:,0] + 10**logM[:,1])
            logZ[:,0] = -1.492 + 1.847*logMt - 0.08026*logMt**2
            # logZ[:,1] = -1.492 + 1.847*logMt - 0.08026*logMt**2
    except:
        logZ = -1.492 + 1.847*logM - 0.08026*logM**2
    
    logZ = logZ - c.ohsun + np.log10(c.zsun) # We leave it in log(Z)
    
    return logMt, logZ


def get_Ztremonti2(logM,logZ,minZ,maxZ,Lagn_param):
    # Correction in bins to Z values using the Ms and Z scale relation from Tremonti et. al. 2004
    
    logZt = np.copy(logZ)    

    logMtot, logZt = get_Ztremonti(logM,logZt,Lagn_param)
    # logZt = Z_blanc(logM)
    
    logZtot = logZ[:,0]
    logZt = logZt[:,0]
    
    # ind_lims = np.where((logZtot > np.log10(minZ)) & (logZtot < np.log10(maxZ)))[0]
    ind_lims = np.where((logZtot > -900) & (logZtot < 900))[0]
    
    smin = 7
    smax = 12
    ds = 0.05
    sbins = np.arange(smin, (smax + ds), ds)
    sbinsH = np.arange(smin, smax, ds)
    shist = sbinsH + ds * 0.5

    median = perc_2arrays(sbins, logMtot[ind_lims], logZtot[ind_lims], 0.5)
    median_t = perc_2arrays(sbins, logMtot[ind_lims], logZt[ind_lims], 0.5)
    ind_med = np.where(median != -999.)[0]

    shist = shist[ind_med]
    median = median[ind_med]
    median_t = median_t[ind_med]
    
    final_bin = sbins[ind_med[-1]+1]
    sbins = sbins[ind_med]
    sbins = np.append(sbins,final_bin)
    
    dif = median_t - median

    for i in range(len(sbins)-1):
        ind = np.where((logMtot>sbins[i])&(logMtot<sbins[i+1]))
        logZ[:,0][ind] = logZ[:,0][ind] + dif[i]
        logZ[:,1][ind] = logZ[:,1][ind] + dif[i]
        
    # smin = 7
    # smax = 12
    # ds = 0.05
    # sbins = np.arange(smin, (smax + ds), ds)
    # sbinsH = np.arange(smin, smax, ds)
    # shist = sbinsH + ds * 0.5

    # median2 = perc_2arrays(sbins, logMtot[ind_lims], logZ[:,0][ind_lims], 0.5)
    # ind_med = np.where(median2 != -999.)[0]
    # median2 = median2[ind_med]
        
    # print(median2)
    # print()
    # print(median_t)
    
    return logZ

    
def get_Zblanc(logM_or):
    logZ = np.zeros(logM_or.shape)
    logM = logM_or - 9.35
    for comp in range(logM_or.shape[1]):
        for i in range(logM_or.shape[0]):
            if logM[i,comp]<(8.7-9.35):
                logZ[i,comp] = 8.37
            elif logM[i,comp]<(9.5-9.35):
                logZ[i,comp] = 8.37 + 0.14*logM[i,comp] - 0.14*(8.7-9.35)
            elif logM[i,comp]<(10.5-9.35):
                logZ[i,comp] = 8.37 + 0.14*(9.5-9.35) - 0.14*(8.7-9.35) + 0.37*logM[i,comp] - 0.37*(9.5-9.35)
            else:
                logZ[i,comp] = 8.37 + 0.14*(9.5-9.35) - 0.14*(8.7-9.35) + 0.37*(10.5-9.35) - 0.37*(9.5-9.35) + 0.03*logM[i,comp] - 0.03*(10.5-9.35)
            
    logZ = logZ - c.ohsun + np.log10(c.zsun) # We leave it in log(Z)
    
    return logZ


def correct_Z(zeq,lms,lzgas,minZ,maxZ,Lagn_param):
    '''
    Correct the metallicity using equations from the literature

    Parameters
    ----------
    zeq: string
         Equation from the literature to folow.    
    Lagn_agn : array of float
         
    Returns
    -------
    znew : array of float
    '''

    znew = np.copy(lzgas)
    
    if zeq not in c.zeq:
        if verbose:
            print('WARNING (gne_Z): No correction done to Z.')
            print('                 Possible zeq= {}'.format(c.zeq))
    elif (zeq == 'tremonti2004'):
        znew = get_Ztremonti(lms,lzgas,Lagn_param)[1]
    elif (zeq == 'tremonti2004b'):
        znew = get_Ztremonti2(lms,lzgas,minZ,maxZ,Lagn_param)
    elif (zeq == 'leblanc'):        
        znew = get_Zblanc(lms)
        
    return znew



def correct_Zagn(lm_tot,lz):
    '''
    Corrects the global gas metallicity with gradients from Belfiore+2017

    Parameters
    ----------
    lm_tot : array of floats
       Stellar mass of the galaxy (log10(M*/Msun)).
    lz : array of floats
       Cold gas metallicity, log10(Z_gas).
     
    Returns
    -------
    lzout : array of floats
        Central metallicity
    '''

    lzout = np.copy(lz)
    
    # Numbers approximated from Fig. 3 in
    # Belfiore+2017 (https://arxiv.org/pdf/1703.03813)
    mask = (lm_tot>9.5) & (lm_tot<=10) & (lz>c.notnum)
    lzout[mask] = lzout[mask] + 0.1

    mask = (lm_tot>10.) & (lm_tot<=11) & (lz>c.notnum)
    lzout[mask] = lzout[mask] + 0.2

    return lzout


def get_zgasagn(infile,cols,selection=None,inoh=False,
                Z_correct_grad=False,lm_tot=None,
                inputformat='hdf5',testing=False,verbose=False):
    '''
    Get the central gas metallicity as log10(Zgas=MZcold/Mcold)

    Parameters
    ----------
    infile : list of strings
        Input file names.
    cols : list of integer (text file) or strings (hdf5 file)
        Location of the (central) metallicity in input files
    selection : list of integers
        Indices of selected galaxies from input files
    inoh : boolean
        If true, the input is assumed to be 12+log10(O/H), otherwise Zgas    
    Z_correct_gradrection : boolean
        If True, corrects Zgas_NLR using gradients from the literature
    lm_tot : numpy array
        Total mass of the galaxy to apply gradient corrections if needed
    inputformat : string
        Format of the input file: 'hdf5' or 'txt'.
    testing : boolean
      If True only run over few entries for testing purposes
    verbose : boolean
        If True print out messages    

    Returns
    -------
    lzgas_agn : array of floats
    '''
    ncomp = len(cols)
    data = io.read_data(infile,selection,inputformat=inputformat,
                     params=cols,testing=testing,verbose=verbose)
    if ncomp>1: # Add components
        zz = np.sum(data, axis=0)
    else:
        zz = data

    lzgas = get_lzgas(zz,inoh=inoh)

    if Z_correct_grad:
        lzgas = correct_Zagn(lm_tot,lzgas)

    lzgas = st.ensure_2d(lzgas)
    return lzgas
