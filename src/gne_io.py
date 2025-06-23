"""
.. moduleauthor:: Violeta Gonzalez-Perez <violetagp@protonmail.com>
.. contributions:: Olivia Vidal <ovive.pro@gmail.com>
.. contributions:: Julen Expósito-Márquez <expox7@gmail.com>
"""
import h5py
import sys
import os
import numpy as np
import src.gne_const as c

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
        


def generate_header(infile,redshift,snap,
                    h0,omega0,omegab,lambda0,vol,mp,
                    units_h0=False,outpath=None,verbose=True):
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
    hf.close()
    
    return filenom


def add2header(filenom,names,values,verbose=True):
    """
    Add attributes to header

    Parameters
    -----------
    filenom : string
        Path to file 
    names : list of strings
        Atribute names
    values: list
        Values of attributes
    verbose : bool
        True for messages
    """
    
    # Open the file header
    hf = h5py.File(filenom, 'a')
    head = hf['header']
    
    # Append attributes
    count = 0
    for ii, nom in enumerate(names):
        if nom is not None:
            head.attrs[nom] = values[ii]
            count += 1
    hf.close()

    if verbose: print(f'* gne_io.add2header: Appended {count} attributes out of {len(names)}')
    
    return count


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
              'Columns should be given as [[0,1,..][...]]')
        sys.exit()
        
    return ncomp


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
    # Initialise output
    outms, outssfr, outzgas = [np.zeros((ncomp,cut.size)) for i in range(3)]

    # Read input data
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

    for i in range(ncomp):
        outms[i,:] = ms[i,cut]
        outssfr[i,:] = ssfr[i,cut]
        outzgas[i,:] = zgas[i,cut]

    return outms, outssfr, outzgas        



def read_mgas_hr(infile, cols, selection, inputformat='hdf5',
                 testing=False, verbose=True):    
    '''
    Read input Mgas and scalelenght for each component
    
    Parameters
    ----------
    infile : string
       Name of the input file.
    cols : list of either integers or strings
       Inputs columns for text files or dataset name for hdf5 files.
    selection : array of integers
       List of indexes of the selected galaxies from the samples.
    inputformat : string
       Format of the input file.
    testing : boolean
       If True only run over few entries for testing purposes
    verbose : boolean
       If True print out messages.
     
    Returns
    -------
    mgas, hr : array of floats
    '''

    check_file(infile, verbose=verbose)

    # Initialise output
    ncomp = get_ncomponents(cols)
    outm, outr = [np.zeros((ncomp,len(selection))) for i in range(2)]

    # Read input data
    if inputformat not in c.inputformats:
        if verbose:
            print('STOP (gne_io): Unrecognised input format.',
                  'Possible input formats = {}'.format(c.inputformats))
        sys.exit()
    elif inputformat=='hdf5':
        with h5py.File(infile, 'r') as hf:
            for i in range(ncomp):
                if i==0:
                    mgas = np.array([hf[cols[i][0]][:]])
                    hr = np.array([hf[cols[i][1]][:]])
                else:
                    mgas = np.append(mgas,[hf[cols[i][0]][:]],axis=0)
                    hr = np.append(hr,[hf[cols[i][1]][:]],axis=0)
    elif inputformat=='txt':
        ih = get_nheader(infile)            
        for i in range(ncomp):
            X = np.loadtxt(infile,usecols=cols[i],skiprows=ih).T
            
            if i==0:
                mgas = np.array([X[0]])
                hr = np.array([X[1]])
            else:
                mgas = np.append(mgas,[X[0]],axis=0)
                hr = np.append(hr,[X[1]],axis=0)

    for i in range(ncomp):
        outm[i,:] = mgas[i,selection]
        outr[i,:] = hr[i,selection]

    return outm, outr        


          
def get_mgas_hr(infile,selection,cols,r_type,
                h0=None,units_h0=False,
                re2hr=c.re2hr,r502re=c.r502re,rvir2r50=c.rvir2r50,
                inputformat='hdf5',
                testing=False,verbose=False):
    '''
    Get Mgas and scalelength in the adecuate units.

    Parameters
    ----------
    infile : string
       Name of the input file.
    cols : list of either integers or strings
       Inputs columns for text files or dataset name for hdf5 files.
    r_type : list of integers per component
       0 for scalelength; 1 for R50 or Reff; and 2 for a full radius
    selection : array of integers
       List of indexes of the selected galaxies from the samples.
    h0 : float
       Hubble constant divided by 100 (only needed if units_h0 is True)
    units_h0: boolean
       True if input units with h
    re2hr: float
       Constant, a, to get the scalength, hr, from
       an effective radius (3D), Re, as hr=a*Re
    r502re: float
       Constant, a, to get the effective radius (3D), Re, from
       a half-mass(light) radius (2D), R50, as Re=a*R50
    rvir2r50: float
       Constant, a, to get the half-mass(light) radius (2D), R50, from
       the halo radius, Rvir, as R50=a*Rvir
    inputformat : string
       Format of the input file.
    testing : boolean
       If True only run over few entries for testing purposes
    verbose : boolean
       If True print out messages.

    Returns
    -------
    mgas, hr : array of floats (Msun, Mpc)
    '''

    # Read Mgas and hr
    mgas, hr = read_mgas_hr(infile,cols,selection,
                            inputformat=inputformat,
                            testing=testing,verbose=verbose)
    if units_h0:
        mgas = mgas/h0
        hr = hr/h0

    # Initialise output with change of units
    outm = np.copy(mgas)
    outr = np.copy(hr)

    # Check that rtype is adequate
    ncomp = np.shape(outr)[0]
    if (min(r_type)<0 or max(r_type)>3 or len(r_type)!=ncomp):
        print('WARNING! Input r_type should be 0, 1, 2 or 3, per component.')

    # Correct scalelenght for each component
    for i in range(ncomp):
        if r_type[i] == 1:
            # Get the scalelenght from an effective (2D, projected) radius
            outr[i,:] = re2hr*hr[i,:]
        elif r_type[i] == 2:
            # Get the scalelenght from the half-mass(light) 3D radius
            outr[i,:] = re2hr*r502re*hr[i,:]
        elif r_type[i] == 3:
            # Get the scalelenght from the radius of the halo
            outr[i,:] = re2hr*r502re*rvir2r50*hr[i,:]

    return outm, outr


def write_sfr_data(filenom,lms,lssfr,lu_sfr,lnH_sfr,lzgas_sfr,
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
    lnH_sfr : floats
     nH of the galaxies per component (cm^-3).
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
                gdat.create_dataset(extra_params_names[i], data =\
                                    extra_param[i][:,None], maxshape=(None,None))
                if extra_params_labels:
                    gdat[extra_params_names[i]].dims[0].label = extra_params_labels[i]

        # SF data
        hfdat = hf.create_group('sfr_data')
        hfdat.create_dataset('lu_sfr', data=lu_sfr, maxshape=(None,None))
        hfdat['lu_sfr'].dims[0].label = 'log10(U) (dimensionless)'
    
        hfdat.create_dataset('lnH_sfr',data=lnH_sfr, maxshape=(None,None))
        hfdat['lnH_sfr'].dims[0].label = 'log10(nH) (cm**-3)'
    
        hfdat.create_dataset('lz_sfr', data=lzgas_sfr, maxshape=(None,None))
        hfdat['lz_sfr'].dims[0].label = 'log10(Z_cold_gas) (dimensionless)'

        for i in range(len(c.line_names[photmod_sfr])):           
            hfdat.create_dataset(c.line_names[photmod_sfr][i] + '_sfr', 
                                 data=nebline_sfr[:,i], maxshape=(None,None))
            hfdat[c.line_names[photmod_sfr][i] + '_sfr'].dims[0].label = \
                'Lines units: [Lsun = 3.826E+33egr s^-1 per unit SFR(Mo/yr) for 10^8yr]'
            
            if fluxes_sfr.any():
                hfdat.create_dataset(c.line_names[photmod_sfr][i] + '_sfr_flux', 
                                     data=fluxes_sfr[:,i], maxshape=(None,None))
                hfdat[c.line_names[photmod_sfr][i] + '_sfr_flux'].dims[0].label = \
                    'Lines units: egr s^-1 cm^-2'
                
            if fluxes_sfr_att.any():
                hfdat.create_dataset(c.line_names[photmod_sfr][i] + '_sfr_flux_att', 
                                     data=fluxes_sfr_att[:,i], maxshape=(None,None))
                hfdat[c.line_names[photmod_sfr][i] + '_sfr_flux_att'].dims[0].label = \
                    'Lines units: egr s^-1 cm^-2'

            
            if nebline_sfr_att.any():
                if nebline_sfr_att[0,i,0] > 0:
                    hfdat.create_dataset(c.line_names[photmod_sfr][i] + '_sfr_att', 
                                         data=nebline_sfr_att[:,i], maxshape=(None,None))
                    hfdat[c.line_names[photmod_sfr][i] + '_sfr_att'].dims[0].label = \
                        'Lines units: [Lsun = 3.826E+33egr s^-1 per unit SFR(Mo/yr) for 10^8yr]'
    
    return 


def write_agn_data(filenom,Lagn,lu_agn,lzgas_agn,
                   nebline_agn,nebline_agn_att=None,
                   fluxes_agn=None,fluxes_agn_att=None,
                   epsilon_agn=None,
                   ew_notatt=None,ew_att=None,
                   verbose=True):
    '''
    Write line data from AGNs in output file

    Parameters
    ----------
    filenom : string
       Name of the output file.
    Lagn : numpy array
       Bolometric luminosity (erg/s)
    lu_agn : floats
     U of the galaxies per component.
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
    photmod_agn = header.attrs['photmod_NLR']
    f.close()

    with h5py.File(filenom,'a') as hf:
        # AGN data
        hfdat = hf.create_group('agn_data')

        hfdat.create_dataset('Lagn', data=Lagn, maxshape=(None))
        hfdat['Lagn'].dims[0].label = 'L_bol (erg/s)'
        
        hfdat.create_dataset('lu_agn', data=lu_agn, maxshape=(None,None))
        hfdat['lu_agn'].dims[0].label = 'log10(U) (dimensionless)'

        hfdat.create_dataset('lz_agn', data=lzgas_agn, maxshape=(None,None))
        hfdat['lz_agn'].dims[0].label = 'log10(Z)'

        if epsilon_agn is not None:
            hfdat.create_dataset('epsilon_NLR', data=epsilon_agn, maxshape=(None))
            hfdat['epsilon_NLR'].dims[0].label = \
                'AGN NLRs volume filling factor (dimensionless)'

        for i in range(len(c.line_names[photmod_agn])):
            hfdat.create_dataset(c.line_names[photmod_agn][i] + '_agn', 
                                 data=nebline_agn[0,i][None,:], maxshape=(None,None))
            hfdat[c.line_names[photmod_agn][i] + '_agn'].dims[0].label = \
                'Lines units: egr s^-1'
            
            if fluxes_agn.any():
                hfdat.create_dataset(c.line_names[photmod_agn][i] + '_agn_flux', 
                                     data=fluxes_agn[0,i][None,:], maxshape=(None,None))
                hfdat[c.line_names[photmod_agn][i] + '_agn_flux'].dims[0].label = \
                    'Lines units: egr s^-1 cm^-2'
                
            if fluxes_agn_att.any():
                if fluxes_agn_att[0,i,0] >= 0:
                    hfdat.create_dataset(c.line_names[photmod_agn][i] + '_agn_flux_att', 
                                         data=fluxes_agn_att[0,i][None,:], maxshape=(None,None))
                    hfdat[c.line_names[photmod_agn][i] + '_agn_flux_att'].dims[0].label = \
                        'Lines units: egr s^-1 cm^-2'
            
            if nebline_agn_att.any():
                if nebline_agn_att[0,i,0] >= 0:
                    hfdat.create_dataset(c.line_names[photmod_agn][i] + '_agn_att', 
                                         data=nebline_agn_att[0,i][None,:], maxshape=(None,None))
                    hfdat[c.line_names[photmod_agn][i] + '_agn_att'].dims[0].label = \
                        'Lines units: egr s^-1'

    return 

