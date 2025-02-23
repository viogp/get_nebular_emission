"""
.. Moduleauthor:: Violeta Gonzalez-Perez <violetagp@protonmail.com>
.. contributions:: Olivia Vidal <ovive.pro@gmail.com>
.. contributions:: Julen Expósito-Márquez <expox7@gmail.com>
"""
import h5py
import sys
import os
import numpy as np
import src.gne_const as c
import math


def stop_if_no_file(infile):
    '''
    It stops the program if a file does not exists

    Parameters
    -------
    infile : string
        Input file
    '''
    
    if (not os.path.isfile(infile)):
        print('STOP: no input file {}'.format(infile)) 
        sys.exit()
    return


def check_file(infile,verbose=False):
    '''
    It checks if a file exists

    Parameters
    -------
    infile : string
        Input file
    verbose : boolean
        If True print out messages

    Returns
    -------
    file_fine : boolean
        True when the file exists.
    '''
    
    file_fine = True  
    if (not os.path.isfile(infile)):
        file_fine = False
        if verbose:
            print('WARNING (gne_io.check_file): file not found {}'.
                  format(infile))

    return file_fine


def create_dir(outdir):
    '''
    Return True if directory already exists or it has been created
    '''
    if not os.path.exists(outdir):
        try:
            os.makedirs(outdir)
        except:
            print('WARNING (iotools.create_dir): problem creating directory ',outdir)
            return False
    return True


def get_outroot(root,snap,outpath=None,verbose=False):
    '''
    Get path to output line data and the file root name

    Parameters
    -------
    root : string
        Root for input files
    snap: integer
        Simulation snapshot number
    outpath : string
        Path to output
    verbose : boolean
        If True print out messages

    Returns
    -------
    outroot : string
        Path to output line data files
    '''

    nom = os.path.splitext(root.split('/')[-1])[0]

    if outpath is None:
        dirf = 'output/iz' + str(snap) + '/'
    else:
        dirf = outpath + '/iz' + str(snap) + '/'

    create_dir(dirf)    
    outroot = dirf + nom 

    if verbose:
        print(f'* Root to output: {outroot}')
    return outroot


def get_plotpath(root,verbose=False):
    '''
    Get path to plots given the output data

    Parameters
    -------
    root : string
        Root to data to use for plotting
    verbose : boolean
        If True print out messages

    Returns
    -------
    plotpath : string
        Path to plots
    '''

    if ('/' in root):
        index = root.rfind('/')
        plotpath = root[:index]+'/plots/'
    else:
        plotpath = 'plots/'
    create_dir(plotpath)    

    if verbose:
        print(f'* Path to plots: {plotpath}')
    return plotpath



def get_outnom(filenom,snap,dirf=None,ftype='line_data',ptype='bpt',verbose=False):
    '''
    Get output from a given filename

    Parameters
    -------
    filenom : string
        Name of file
    snap: integer
        Simulation snapshot number
    dirf : string
        Path to output
    ftype : string
        Type of the file: sample, line_data, plots
    ptype : string
        Type of plot: bpt
    verbose : boolean
        If True print out messages

    Returns
    -------
    outfile : string
        Path to output file
    '''

    nom = os.path.splitext(filenom.split('/')[-1])[0]

    if dirf is None:
        dirf = 'output/iz' + str(snap) + '/'
        if ftype == 'plots': dirf = dirf + ftype + '/'
        create_dir(dirf)    

    if ftype == 'line_data':
        outfile = dirf + nom + '.hdf5'
    elif ftype == 'plots':
        outfile = dirf + ptype + '_' + nom + '.pdf'

    if verbose:
        print(f'* Output {ftype}: {outfile}')
    return outfile



def print_h5attr(infile,inhead='header'):
    """
    Print out the group attributes of a hdf5 file

    Parameters
    ----------
    infile : string
      Name of input file (this should be a hdf5 file)
    inhead : string
      Name of the group to read the attributes from

    Example
    -------
    >>> import h2s_io as io
    >>> infile = '/hpcdata0/simulations/BAHAMAS/AGN_TUNED_nu0_L100N256_WMAP9/Data/Snapshots/snapshot_026/snap_026.27.hdf5'
    >>> io.print_h5attr(infile,inhead='Units')
    """

    filefine = check_file(infile) #print(filefine)
    if (not filefine):
        print('WARNING (h2s_io.printh5attr): Check that the file provided is correct')
        return ' '
    
    f = h5py.File(infile, 'r')
    header = f[inhead]
    for hitem in list(header.attrs.items()): 
        print(hitem)
    f.close()

    return ' '


def get_nheader(infile,firstchar=None):
    '''
    Given a text file with a structure: header+data, 
    counts the number of header lines

    Parameters
    -------
    infile : string
        Input file

    Returns
    -------
    ih : integer
        Number of lines with the header text
    '''


    ih = 0
    with open(infile,'r') as ff:
        for line in ff:
            if not line.strip():
                # Count any empty lines in the header
                ih += 1
            else:
                sline = line.strip()
                
                # Check that the first character is not a digit
                char1 = sline[0]
                word1 = sline.split()[0]
                if not firstchar:
                    if (not char1.isdigit()):
                        if (char1 != '-'):
                            ih += 1
                        else:
                            try:
                                float(word1)
                                return ih
                            except:
                                ih += 1
                    else:
                        return ih
                else:
                    if char1 == firstchar:
                        ih+=1
    return ih
        


def get_ncomponents(cols):
    '''
    Get the number of components to estimate the emission lines from

    Parameters
    ----------
    cols : list
      List of columns with properties per components

    Returns
    -------
    ncomp : integer
      Number of components (for example 2 for bulge and disk)
    '''
    
    ncomp = 1
    
    try:
        dum = np.shape(cols)[1]
        ncomp = np.shape(cols)[0]
    except:
        ncomp = 1
        print('STOP (gne_io.get_ncomponents): ',
              'Columns should be given as m_sfr_z=[[0,1,2]]')
        sys.exit()
        
    return ncomp


def get_selection(infile, outfile, inputformat='hdf5',
                  cutcols=None, mincuts=[None], maxcuts=[None],
                  testing=False,verbose=False):
    '''
    Get indexes of selected galaxies

    Parameters
    ----------
    infile : strings
     List with the name of the input files. 
     - In text files (*.dat, *txt, *.cat), columns separated by ' '.
     - In csv files (*.csv), columns separated by ','.
    inputformat : string
     Format of the input file.
    cutcols : list
     Parameters to look for cutting the data.
     - For text or csv files: list of integers with column position.
     - For hdf5 files: list of data names.
    mincuts : strings
     Minimum value of the parameter of cutcols in the same index. All the galaxies below won't be considered.
    maxcuts : strings
     Maximum value of the parameter of cutcols in the same index. All the galaxies above won't be considered.
    verbose : boolean
      If True print out messages
    testing : boolean
      If True only run over few entries for testing purposes

    Returns
    -------
    selection : array of integers
    '''

    selection = None
    
    check_file(infile, verbose=verbose)

    if testing:
        limit = c.testlimit
    else:
        limit = None    

    if inputformat not in c.inputformats:
        if verbose:
            print('STOP (gne_io): Unrecognised input format.',
                  'Possible input formats = {}'.format(c.inputformats))
        sys.exit()
    elif inputformat=='hdf5':
        with h5py.File(infile, 'r') as hf:
            ind = np.arange(len(hf[cutcols[0]][:]))
            for i in range(len(cutcols)):
                if cutcols[i]:
                    param = hf[cutcols[i]][:]
                    mincut = mincuts[i]
                    maxcut = maxcuts[i]

                    if mincut and maxcut:
                        ind = np.intersect1d(ind,np.where((mincut<param)&(param<maxcut))[0])
                    elif mincut:
                        ind = np.intersect1d(ind,np.where(mincut<param)[0])
                    elif maxcut:
                        ind = np.intersect1d(ind,np.where(param<maxcut)[0])
            selection = ind[:limit]
    elif inputformat=='txt':
        ih = get_nheader(infile)
        ind = np.arange(len(np.loadtxt(infile,usecols=cutcols[0],skiprows=ih)))
        for i in range(len(cutcols)):
            if cutcols[i]:
                param = np.loadtxt(infile,usecols=cutcols[i],skiprows=ih)
                mincut = mincuts[i]
                maxcut = maxcuts[i]

                if mincut and maxcut:
                    ind = np.intersect1d(ind,np.where((mincut<param)&(param<maxcut))[0])
                elif mincut:
                    ind = np.intersect1d(ind,np.where(mincut<param)[0])
                elif maxcut:
                    ind = np.intersect1d(ind,np.where(param<maxcut)[0])
        selection = ind[:limit]
    else:
        if verbose:
            print('STOP (gne_io.get_selection): ',
                  'Input file has not been found.')
        sys.exit()

    return selection


def read_data(infile, cut, inputformat='hdf5', params=[None],
              testing=False, verbose=True):    
    '''
    Read input data per column/dataset
    
    Parameters
    ----------
    infile : string
       Name of the input file. 
    cut : array of integers
       List of indexes of the selected galaxies from the samples.
    inputformat : string
       Format of the input file.
    params : list of either integers or strings
       Inputs columns for text files or dataset name for hdf5 files.
    testing : boolean
       If True only run over few entries for testing purposes
    verbose : boolean
       If True print out messages.
     
    Returns
    -------
    outparams : array of floats
    '''

    check_file(infile, verbose=verbose)

    if inputformat not in c.inputformats:
        if verbose:
            print('STOP (gne_io): Unrecognised input format.',
                  'Possible input formats = {}'.format(c.inputformats))
        sys.exit()
    elif inputformat=='hdf5':
        with h5py.File(infile, 'r') as hf:
            ii = 0
            for nomparam in params:
                if (nomparam is not None):
                    prop = np.zeros(np.shape(cut))
                    try:
                        ###here what if Pos/vel as matrix?
                        prop = hf[nomparam][cut]
                    except:
                        print('\n WARNING (gne_io): no {} found in {}'.format(
                            nomparam,infile))

                    if (ii == 0):
                        outparams = prop
                    else:
                        outparams = np.vstack((outparams,prop))
                    ii += 1

    elif inputformat=='txt': ###need to adapt to the generalisation and test
        ih = get_nheader(infile)
        outparams = np.loadtxt(infile,skiprows=ih,usecols=params)[cut].T

    return outparams



def read_sfrdata(infile, cols, cut, inputformat='hdf5',
                 testing=False, verbose=True):    
    '''
    Read input M, SFR and Z data for each component
    
    Parameters
    ----------
    infile : string
       Name of the input file.
    cols : list of either integers or strings
       Inputs columns for text files or dataset name for hdf5 files.
    cut : array of integers
       List of indexes of the selected galaxies from the samples.
    inputformat : string
       Format of the input file.
    testing : boolean
       If True only run over few entries for testing purposes
    verbose : boolean
       If True print out messages.
     
    Returns
    -------
    outms, outssfr, outzgas : array of floats
    '''

    check_file(infile, verbose=verbose)
    
    ncomp = get_ncomponents(cols)
    
    if inputformat not in c.inputformats:
        if verbose:
            print('STOP (gne_io): Unrecognised input format.',
                  'Possible input formats = {}'.format(c.inputformats))
        sys.exit()
    elif inputformat=='hdf5':
        with h5py.File(infile, 'r') as hf:
            for i in range(ncomp):
                if i==0:
                    ms = np.array([hf[cols[i][0]][:]])
                    ssfr = np.array([hf[cols[i][1]][:]])
                    zgas = np.array([hf[cols[i][2]][:]])
                else:
                    ms = np.append(ms,[hf[cols[i][0]][:]],axis=0)
                    ssfr = np.append(ssfr,[hf[cols[i][1]][:]],axis=0)
                    zgas = np.append(zgas,[hf[cols[i][2]][:]],axis=0)
    elif inputformat=='txt':
        ih = get_nheader(infile)            
        for i in range(ncomp):
            X = np.loadtxt(infile,usecols=cols[i],skiprows=ih).T
            
            if i==0:
                ms = np.array([X[0]])
                ssfr = np.array([X[1]])
                zgas = np.array([X[2]])
            else:
                ms = np.append(ms,[X[0]],axis=0)
                ssfr = np.append(ssfr,[X[1]],axis=0)
                zgas = np.append(zgas,[X[2]],axis=0)
    else:
        if verbose:
            print('STOP (gne_io.read_sfrdata): ',
                  'Input file has not been found.')
        sys.exit()
        
    ms = ms.T
    ssfr = ssfr.T
    zgas = zgas.T

    outms = ms[cut]
    outssfr = ssfr[cut]
    outzgas = zgas[cut]
    
    return outms, outssfr, outzgas        



def get_sfrdata(infile,cols,selection=None,
                h0=None,units_h0=False, units_Gyr=False,
                inoh = False, LC2sfr=False, mtot2mdisk=True, 
                inputformat='hdf5',testing=False,verbose=False):
    '''
    Get Mstars, sSFR and Z in the adecuate units.

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
    LC2sfr : boolean
      If True magnitude of Lyman Continuum photons expected as input for SFR.
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

    ms,sfr,zgas = read_sfrdata(infile, cols, selection,
                               inputformat=inputformat, 
                               testing=testing, verbose=verbose)

    if units_h0:
        ms = ms/h0
        sfr = sfr/h0
    if units_Gyr:
        sfr = sfr/1e9
    
    # Set to a default value if negative stellar masses
    ind = np.where((ms<=1.) | (sfr<0) | (zgas<=0))
    ms[ind] = c.notnum
    sfr[ind] = c.notnum
    zgas[ind] = c.notnum
    
    ####here is this correct? does not seem to make sense
    #if LC2sfr: # Avoid positive magnitudes of LC photons
    #    ind = np.where(lssfr>0)
    #    lssfr[ind] = c.notnum ; lzgas[ind] = c.notnum
    # np.log10(Q[i,comp]/(c.IMF_SFR[IMF[comp]] * c.phot_to_sfr_kenn)) - lms[i,comp]

    ncomp = get_ncomponents(cols)
        
    # Calculate the disk mass if we have only the total and bulge mass
    if mtot2mdisk:
        if ncomp!=2:
            if verbose:
                print('STOP (gne_io.get_data): ',
                      'mtot2mdisk can only be True with two components.')
            sys.exit()
                
        ms_tot = ms[:,0]

        # Calculate the disk mass :
        msdisk = ms[:,0] - ms[:,1]
        ms = np.column_stack((msdisk,ms[:,1]))     

        # Take the log of the total stellar mass
        lms_tot = np.zeros(len(ms_tot)); lms_tot.fill(c.notnum)
        ind = np.where(ms_tot > 0.)
        lms_tot[ind] = np.log10(ms_tot[ind])

        # Take the log of the stellar mass:
        lms = np.zeros(np.shape(ms)); lms.fill(c.notnum)
        ind = np.where(ms > 0.)
        lms[ind] = np.log10(ms[ind])

    else:
        if ncomp!=1:
            ms_tot = np.sum(ms,axis=1)
    
            # Take the log of the total stellar mass:
            lms_tot = np.zeros(len(ms_tot)); lms_tot.fill(c.notnum)
            ind = np.where(ms_tot > 0.)
            lms_tot[ind] = np.log10(ms_tot[ind])
    
        # Take the log of the stellar mass:
        lms = np.zeros(np.shape(ms)); lms.fill(c.notnum)
        ind = np.where(ms > 0.)
        lms[ind] = np.log10(ms[ind])

    # Obtain log10(sSFR) in 1/yr and calculate SFR from LC photons if necessary
    lssfr = np.zeros(np.shape(sfr)); lssfr.fill(c.notnum)
    if LC2sfr:
        for comp in range(ncomp):
            ins = np.zeros(len(sfr))
            ind = np.where(sfr[:, comp] != c.notnum)
            ins[ind] = 1.02*(10.**(-0.4*sfr[ind,comp]-4.))

            ind = np.where(ins > 0)
            lssfr[ind,comp] = np.log10(ins[ind]) - lms[ind,comp]
    
            ind = np.where(lssfr[:, comp] == c.notnum)
            lssfr[ind,comp] = c.notnum
            
    else: # If SFR as input
        for comp in range(ncomp):
            # Take the log of the ssfr:
            ind = np.where(sfr[:,comp] > 0.)[0]
            lssfr[ind,comp] = np.log10(sfr[ind,comp]) - lms[ind,comp]

    if ncomp!=1:
        lssfr_tot = np.zeros(len(lssfr))
        ssfr = np.zeros(lssfr.shape)
        for comp in range(ncomp):
            ind = np.where(lssfr[:,comp]!=c.notnum)
            ssfr[ind,comp] = 10.**(lssfr[ind,comp])

        ins = np.sum(ssfr,axis=1)
        ind = np.where(ins>0)
        lssfr_tot[ind] = np.log10(ins[ind])

            
    if ncomp!=1:
        lsfr = lssfr_tot+lms_tot
    else:
        lsfr = lssfr + lms

    # Obtain log10(Zgas=MZcold/Mcold)        
    lzgas = np.zeros(zgas.shape); lzgas.fill(c.notnum)
    if inoh: 
        # Obtain log10(Zgas) from an input of 12+log10(O/H)
        lzgas = np.log10(c.zsun) - c.ohsun + zgas
    else: 
        ind = np.where(zgas>0)
        lzgas[ind] = np.log10(zgas[ind])

    if ncomp!=1:
        oh12 = np.zeros(lzgas.shape)
        lzgas_tot = np.zeros(len(lzgas)); lzgas_tot.fill(c.notnum)
        for comp in range(ncomp):
            ind = np.where(lzgas[:,comp] != c.notnum)
            oh12[ind,comp] = 10. ** (lzgas[ind,comp])
    
        ins = np.sum(oh12,axis=1)
        ind = np.where(ins>0)
        lzgas_tot[ind] = np.log10(ins[ind])
                    
    return lms,lssfr,lzgas


def get_data_agnnH(infile,rtype,cols,selection=None,
                   h0=None,units_h0=False,inputformat='hdf5',
                   testing=False,verbose=False):
    '''
    Get Mgas and R50 in the adecuate units.

    Parameters
    ----------
    infile : string
       Name of the input file
    inputformat : string
     Format of the input file.
    cols : list
      Expected : component1 = total or disk, component2 = bulge
    units_h0 : bool
    verbose : boolean
      If True print out messages
    testing : boolean
      If True only run over few entries for testing purposes

    Returns
    -------
    outparams : array of floats
    '''

    vals = read_data(infile,selection,inputformat=inputformat,
                     params=cols,testing=testing,verbose=verbose)
    if units_h0:
        vals = vals/h0

    outparams = vals

    if rtype == 'reff':
        # Transform  reff (exponential): rscale = reff/1.678
        outparams[1] = outparams[1]/c.re2rs_exp
        if np.shape(outparams)[0]>2: outparams[3] = outparams[3]/c.re2rs_exp
    elif rtype == 'r':
        # Transform  r: rscale = r/2./1.678
        outparams[1] = outparams[1]/2./c.re2rs_exp
        if np.shape(outparams)[0]>2: outparams[3] = outparams[3]/2./c.re2rs_exp
        
    return outparams



def generate_header(infile,redshift,snap,
                    h0,omega0,omegab,lambda0,vol,mp,
                    units_h0=False,outpath=None,
                    une_sfr_nH=None, une_sfr_U=None,
                    photmod_sfr=None,
                    une_agn_nH=None, une_agn_spec=None,
                    une_agn_U=None, photmod_agn=None,
                    attmod=None,verbose=True):
    """
    Generate the header of the file with the line data

    Parameters
    -----------
    infile : string
        Path to input
    zz: float
        Redshift of the simulation snapshot
    snap: integer
        Simulation snapshot number
    h0 : float
        Hubble constant divided by 100
    omega0 : float
        Matter density at z=0
    omegab : float
        Baryonic density at z=0
    lambda0 : float
        Cosmological constant z=0
    vol : float
        Simulation volume
    mp : float
        Simulation resolution, particle mass
    units_h0: boolean
        True if input units with h    
    outpath : string
        Path to output
    une_sfr_nH : string
        Model to go from galaxy properties to Hydrogen (or e) number density.
    une_sfr_U : string
        Model to go from galaxy properties to ionising parameter.
    photmod_sfr : string
        Photoionisation model to be used for look up tables.
    une_agn_nH : list of 2 strings
        Profile assumed for the gas around NLR AGN and type of radii.
    une_agn_spec : string
        Model for the spectral distribution for AGNs.
    une_sfr_U : string
        Model to go from galaxy properties to AGN ionising parameter.
    photmod_agn : string
        Photoionisation model to be used for look up tables.
    attmod : string
        Attenuation model.
    verbose : bool
        True for messages
 
    Returns
    -----
    filenom : string
       Full path to the output file
    """

    # Get the file name
    filenom = get_outnom(infile,snap,dirf=outpath,ftype='line_data',verbose=verbose)

    # Change units if required
    if units_h0:
        vol = vol/(h0*h0*h0)
        mp = mp/h0
    
    # Generate the output file (the file is rewrtitten)
    hf = h5py.File(filenom, 'w')

    # Generate a header
    headnom = 'header'
    head = hf.create_dataset(headnom,(100,))
    head.attrs[u'redshift'] = redshift
    head.attrs[u'h0'] = h0
    head.attrs[u'omega0'] = omega0
    head.attrs[u'omegab'] = omegab
    head.attrs[u'lambda0'] = lambda0
    head.attrs[u'vol_Mpc3'] = vol
    head.attrs[u'mp_Msun'] = mp

    if une_sfr_nH is not None: head.attrs[u'une_sfr_nH'] = une_sfr_nH
    if une_sfr_U is not None: head.attrs[u'une_sfr_U'] = une_sfr_U    
    if photmod_sfr is not None: head.attrs[u'photmod_sfr'] = photmod_sfr
    if une_agn_nH is not None: head.attrs[u'une_agn_nH'] = une_agn_nH[0]
    if une_agn_spec is not None: head.attrs[u'une_agn_spec'] = une_agn_spec
    if une_agn_U is not None: head.attrs[u'une_agn_U'] = une_agn_U
    if photmod_agn is not None: head.attrs[u'photmod_agn'] = photmod_agn
    if attmod is not None: head.attrs[u'attmod'] = attmod
    hf.close()
    
    return filenom
    

def write_sfr_data(filenom,lms,lssfr,lu_sfr,lne_sfr,lzgas_sfr,
               nebline_sfr,nebline_sfr_att=None,fluxes_sfr=None,fluxes_sfr_att=None,
               extra_param=[[None]],extra_params_names=None,extra_params_labels=None,
               verbose=True):
    '''
    Write line data from star forming regions

    Parameters
    ----------
    filenom : string
       Full path to the output file
    lms_sfr : floats
     Masses of the galaxies per component (log10(M*) (Msun)).
    lssfr_sfr : floats
     sSFR of the galaxies per component (log10(SFR/M*) (1/yr)).
    lu_sfr : floats
     U of the galaxies per component.
    lne_sfr : floats
     ne of the galaxies per component (cm^-3).
    lzgas_sfr : floats
     Metallicity of the galaxies per component (12+log(O/H))
    nebline_sfr : floats
      Array with the luminosity of the lines per component. (Lsun per unit SFR(Mo/yr) for 10^8yr)
    nebline_sfr_att : floats
      Array with the luminosity of the attenuated lines per component. (Lsun per unit SFR(Mo/yr) for 10^8yr)    
    extra_params : list
     Parameters from the input files which will be saved in the output file.
     - For text or csv files: list of integers with column position.
     - For hdf5 files: list of data names.
    extra_params_names : strings
     Names of the datasets in the output files for the extra parameters.
    extra_params_labels : strings
     Description labels of the datasets in the output files for the extra parameters.
    '''

    # Read information on models
    f = h5py.File(filenom, 'r')   
    header = f['header']
    photmod_sfr = header.attrs['photmod_sfr']
    f.close()

    # Output data
    with h5py.File(filenom,'a') as hf:
        # Global data
        gdat = hf.create_group('data')
        
        gdat.create_dataset('lms', data=lms, maxshape=(None,None))
        gdat['lms'].dims[0].label = 'log10(M*/Msun)'
        
        gdat.create_dataset('lssfr', data=lssfr, maxshape=(None,None))
        gdat['lssfr'].dims[0].label = 'log10(SFR/M*/yr)'

        if extra_param[0][0] != None:
            for i in range(len(extra_param)):
                gdat.create_dataset(extra_params_names[i], data=extra_param[i][:,None], maxshape=(None,None))
                if extra_params_labels:
                    gdat[extra_params_names[i]].dims[0].label = extra_params_labels[i]

        # SF data
        hfdat = hf.create_group('sfr_data')
        hfdat.create_dataset('lu_sfr', data=lu_sfr, maxshape=(None,None))
        hfdat['lu_sfr'].dims[0].label = 'log10(U) (dimensionless)'
    
        hfdat.create_dataset('lne_sfr',data=lne_sfr, maxshape=(None,None))
        hfdat['lne_sfr'].dims[0].label = 'log10(nH) (cm**-3)'
    
        hfdat.create_dataset('lz_sfr', data=lzgas_sfr, maxshape=(None,None))
        hfdat['lz_sfr'].dims[0].label = 'log10(Z) (dimensionless)'

        for i in range(len(c.line_names[photmod_sfr])):           
            hfdat.create_dataset(c.line_names[photmod_sfr][i] + '_sfr', 
                                 data=nebline_sfr[:,i], maxshape=(None,None))
            hfdat[c.line_names[photmod_sfr][i] + '_sfr'].dims[0].label = \
                'Lines units: [Lsun = 3.826E+33egr s^-1 per unit SFR(Mo/yr) for 10^8yr]'
            
            if fluxes_sfr.any():
                hfdat.create_dataset(c.line_names[photmod_sfr][i] + '_sfr_flux', 
                                     data=fluxes_sfr[:,i], maxshape=(None,None))
                hfdat[c.line_names[photmod_sfr][i] + '_sfr_flux'].dims[0].label = 'Lines units: egr s^-1 cm^-2'
                
            if fluxes_sfr_att.any():
                hfdat.create_dataset(c.line_names[photmod_sfr][i] + '_sfr_flux_att', 
                                     data=fluxes_sfr_att[:,i], maxshape=(None,None))
                hfdat[c.line_names[photmod_sfr][i] + '_sfr_flux_att'].dims[0].label = 'Lines units: egr s^-1 cm^-2'

            
            if nebline_sfr_att.any():
                if nebline_sfr_att[0,i,0] > 0:
                    hfdat.create_dataset(c.line_names[photmod_sfr][i] + '_sfr_att', 
                                         data=nebline_sfr_att[:,i], maxshape=(None,None))
                    hfdat[c.line_names[photmod_sfr][i] + '_sfr_att'].dims[0].label = 'Lines units: [Lsun = 3.826E+33egr s^-1 per unit SFR(Mo/yr) for 10^8yr]'
    
    return 


def write_agn_data(filenom,lu_agn,lne_agn,lzgas_agn,
                   nebline_agn,nebline_agn_att=None,fluxes_agn=None,fluxes_agn_att=None,
                   epsilon_agn=None,
                   ew_notatt=None,ew_att=None,
                   verbose=True):
    '''
    Write line data from AGNs in output file

    Parameters
    ----------
    filenom : string
      Name of the output file.
    lu_agn : floats
     U of the galaxies per component.
    lne_agn : floats
     ne of the galaxies per component (cm^-3).
    lzgas_agn : floats
     Metallicity of the galaxies per component (12+log(O/H))
    nebline_agn : array of floats
       Luminosities (erg/s)
    nebline_agn_att : array of floats
       Dust attenuated luminosities (erg/s)
    '''
    
    # Read information on models
    f = h5py.File(filenom, 'r')   
    header = f['header']
    photmod_agn = header.attrs['photmod_agn']
    f.close()

    with h5py.File(filenom,'a') as hf:
        # AGN data
        hfdat = hf.create_group('agn_data')

        hfdat.create_dataset('lu_agn', data=lu_agn, maxshape=(None,None))
        hfdat['lu_agn'].dims[0].label = 'log10(U) (dimensionless)'
    
        hfdat.create_dataset('lne_agn',data=lne_agn, maxshape=(None,None))
        hfdat['lne_agn'].dims[0].label = 'log10(nH) (cm**-3)'
    
        hfdat.create_dataset('lz_agn', data=lzgas_agn, maxshape=(None,None))
        hfdat['lz_agn'].dims[0].label = 'log10(Z)'
        
        hfdat.create_dataset('epsilon_agn', data=epsilon_agn[None,:], maxshape=(None,None))
        hfdat['epsilon_agn'].dims[0].label = 'NLRs volume filling factor (dimensionless)'

        for i in range(len(c.line_names[photmod_agn])):
            hfdat.create_dataset(c.line_names[photmod_agn][i] + '_agn', 
                                 data=nebline_agn[0,i][None,:], maxshape=(None,None))
            hfdat[c.line_names[photmod_agn][i] + '_agn'].dims[0].label = 'Lines units: egr s^-1'
            
            if fluxes_agn.any():
                hfdat.create_dataset(c.line_names[photmod_agn][i] + '_agn_flux', 
                                     data=fluxes_agn[0,i][None,:], maxshape=(None,None))
                hfdat[c.line_names[photmod_agn][i] + '_agn_flux'].dims[0].label = 'Lines units: egr s^-1 cm^-2'
                
            if fluxes_agn_att.any():
                if fluxes_agn_att[0,i,0] >= 0:
                    hfdat.create_dataset(c.line_names[photmod_agn][i] + '_agn_flux_att', 
                                         data=fluxes_agn_att[0,i][None,:], maxshape=(None,None))
                    hfdat[c.line_names[photmod_agn][i] + '_agn_flux_att'].dims[0].label = 'Lines units: egr s^-1 cm^-2'
            
            if nebline_agn_att.any():
                if nebline_agn_att[0,i,0] >= 0:
                    hfdat.create_dataset(c.line_names[photmod_agn][i] + '_agn_att', 
                                         data=nebline_agn_att[0,i][None,:], maxshape=(None,None))
                    hfdat[c.line_names[photmod_agn][i] + '_agn_att'].dims[0].label = 'Lines units: egr s^-1'

    return 

