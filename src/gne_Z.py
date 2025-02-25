"""
.. moduleauthor:: Julen Expósito-Márquez <expox7@gmail.com>
.. contributions:: Violeta Gonzalez-Perez <violetagp@protonmail.com>
"""
import numpy as np
import src.gne_const as c

def correct_Zagn(lm,lzgas):
    '''
    Estimates the metallicity of the AGN from the global gas metallicity.

    Parameters
    ----------
    lm : array of floats
       Stellar mass of the galaxy or its components (log10(M*/Msun)).
    lzgas : array of floats
       Cold gas metallicity, log10(Z_gas).
     
    Returns
    -------
    lzout : array of floats
        Metallicity of the AGN
    '''

    lzout = np.copy(lzgas)

    n_comp = lzout.shape[1]
    if n_comp > 1:
        ms = 10**lm
        lm_tot = np.log10(np.sum(ms,axis=1))
    else:
        lm_tot = np.copy(lm)


    for ii in range(n_comp):
        lz = lzgas[:,ii]

        # Numbers approximated from Fig. 3 in
        # Belfiore+2017 (https://arxiv.org/pdf/1703.03813)
        mask = (lm_tot>9.5) & (lm_tot<=10) & (lz>c.notnum)
        lzout[mask,ii] = lzout[mask,ii] + 0.1

        mask = (lm_tot>10.) & (lm_tot<=11) & (lz>c.notnum)
        lzout[mask,ii] = lzout[mask,ii] + 0.2

    return lzout


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
