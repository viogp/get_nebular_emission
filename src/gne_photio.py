"""
.. moduleauthor:: Violeta Gonzalez-Perez <violetagp@protonmail.com>
.. contributions:: Olivia Vidal <ovive.pro@gmail.com>
.. contributions:: Julen Expósito-Márquez <expox7@gmail.com>
"""
import h5py
import numpy as np
import sys
#import warnings
import src.gne_io as io
import src.gne_const as c
import src.gne_stats as st
from src.gne_cosmology import emission_line_flux, logL2flux, set_cosmology


def get_zfile(zmet_str, photmod='gutkin16'):
    '''
    Given a metallicity string get the name of the corresponding table

    Parameters
    ----------
    zmet_str : string
        Metallicity name in files.
    photomod : string
        Name of the considered photoionisation model.

    Returns
    -------
    zfile : string
        Name of the model file with data for the given metallicity.
    '''

    root = 'data/nebular_data/' + photmod + '_tables/nebular_emission_Z'
    if len(zmet_str)<3:
        zmet_str = zmet_str+'0'
    zfile = root + zmet_str + '.txt'
    # Si son 2 numeros que le añada un cero
    file_fine = io.check_file(zfile)
    if (not file_fine):
        zfile = None

    return zfile


def clean_photarray(lms, lssfr, lu, lnH, lzgas, photmod='gutkin16', verbose=True):

    '''
    Given the model, take the values outside the limits and give them the apropriate
    value inside the limits depending on the model.

    Parameters
    ----------
    lms : floats
     Masses of the galaxies per component (log10(M*) (Msun)).
    lssfr : floats
     sSFR of the galaxies per component (log10(SFR/M*) (1/yr)).
    lu : floats
     U of the galaxies per component.
    lnH : floats
     ne of the galaxies per component (cm^-3).
    lzgas : floats
     Metallicity of the galaxies per component (log10(Z)).
    photomod : string
     Name of the considered photoionisation model.
    verbose : boolean
     If True print out messages.

    Returns
    -------
    lms,lssfr,lu,lnH,lzgas : floats
    '''

    minU, maxU = get_limits(propname='logUs', photmod=photmod)
    minnH, maxnH = get_limits(propname='nH', photmod=photmod)
    minZ, maxZ = get_limits(propname='Z', photmod=photmod)
    
    for i in range(lu.shape[1]):        
        lu[:,i][(lu[:,i] > maxU)&(lu[:,i] != c.notnum)] = maxU
        lu[:,i][(lu[:,i] < minU)&(lu[:,i] != c.notnum)] = minU
        
        lnH[:,i][(lnH[:,i] > np.log10(maxnH))&(lnH[:,i] != c.notnum)] = np.log10(maxnH)
        lnH[:,i][(lnH[:,i] < np.log10(minnH))&(lnH[:,i] != c.notnum)] = np.log10(minnH)
        
        lzgas[:,i][(lzgas[:,i] > np.log10(maxZ))&(lzgas[:,i] != c.notnum)] = np.log10(maxZ)
        lzgas[:,i][(lzgas[:,i] < np.log10(minZ))&(lzgas[:,i] != c.notnum)] = np.log10(minZ)
                
    return lms, lssfr, lu, lnH, lzgas


def get_limits(propname, photmod='gutkin16',verbose=True):
    '''
    Given a file with a structure: property + lower limit + upper limit,
    gets the limits of the parameters of the photoionization model.

    In the file we must find the properties well specified i.e U, Z and nH.
    The header lines have to start with '#'

    Parameters
    -------
    propname : string
        name of the property that we want
    photomod : string
        Name of the considered photoionisation model

    Returns
    -------
    lower_limit: float
        lower limit of the requested property
    upper_limit: float
        upper limits of the requested property

    Examples
    -------
    >>> get_limits(propname = 'nH', photmod = 'gutkin16')
        10  10000
    '''

    try:
        infile = c.mod_lim[photmod]
    except KeyError:
        print('STOP (gne_photio.get_limits): the {}'.format(photmod) +
              ' is an unrecognised model in the dictionary mod_lim')
        print('                  Possible photmod= {}'.format(c.mod_lim.keys()))
        sys.exit()

    # Check if the limits file exists:
    io.check_file(infile, verbose=verbose)
    # print(infile)

    prop = np.loadtxt(infile,dtype=str,comments='#',usecols=(0),unpack=True)
    prop = prop.tolist()
    if propname not in prop:
        print('STOP (gne_photio.get_limits): property {} '.format(propname)+
              'not found in the limits file {}'.format(infile))
        sys.exit()
    else:
        ind = prop.index(propname)

        ih = io.get_nheader(infile,firstchar='#')
        lower_limit = np.loadtxt(infile, skiprows=ind+ih, max_rows=1, usecols=(1),unpack=True)
        upper_limit = np.loadtxt(infile,skiprows=ind+ih, max_rows=1,usecols=(2),unpack=True)
        return float(lower_limit),float(upper_limit)


    
def calculate_flux(nebline,filenom,origin='sfr'):
    '''
    Get the fluxes for the emission lines given the luminosity and redshift.

    Params
    -------
    nebline : array of floats
        Luminosities of the lines per component.
        Lsun for L_AGN = 10^45 erg/s
    filenom : string
        Name of file with output
    origin : string
        Emission source (star-forming region or AGN).
      
    Returns
    -------
    fluxes : floats
        Array with the fluxes of the lines per component.
    '''
    
    if nebline.any():
        # Read redshift and cosmo
        f = h5py.File(filenom, 'r')
        header = f['header']
        redshift = header.attrs['redshift']
        h0 = header.attrs['h0']
        omega0 = header.attrs['omega0']
        omegab = header.attrs['omegab']
        lambda0 = header.attrs['lambda0']
        f.close()
        
        set_cosmology(omega0=omega0, omegab=omegab,lambda0=lambda0,h0=h0)
        
        luminosities = np.zeros(nebline.shape)
        luminosities[nebline>0] = np.log10(nebline[nebline>0]*h0**2)
        if (origin=='agn') and (luminosities.shape[0]==2):
            luminosities[1] = 0
            
        fluxes = np.zeros(luminosities.shape)
        for comp in range(luminosities.shape[0]):
            for i in range(luminosities.shape[1]):
                for j in range(luminosities.shape[2]):
                    if luminosities[comp,i,j] == 0:  
                        fluxes[comp,i,j] = 0
                    else:
                        fluxes[comp,i,j] = logL2flux(luminosities[comp,i,j],redshift)
    else:
        fluxes = np.copy(nebline)
            
    return fluxes


def get_Zgrid(zgrid_str):
    '''
    Get the metallicity values from the file names

    Params
    -------
    zgrid_str : array or list of strings
        String with decimal part of the metallicity grid
      
    Returns
    -------
    nz : integer
        Number of elements of zgrid_str
    zgrid, lzgrid : array of floats
        Numerical value of the metallicity grid, Z, and log10(Z)
    '''
    nz = len(zgrid_str)

    zgrid = np.full(nz,c.notnum)
    zgrid = np.array([float('0.' + zz) for zz in zgrid_str])

    lzgrid = np.full(nz, c.notnum)
    ind = np.where(zgrid > 0.)
    if (np.shape(ind)[1] > 0):
        lzgrid[ind] = np.log10(zgrid[ind])

    return nz,zgrid,lzgrid


def interp_u_z(grid,u,ud,iu,zd,iz):
    '''
    Bilinear interpolation on U and Z

    Params
    -------
    grid : array of floats
       Grid for U and Z given values
    u : array of floats
       U is used to select values
    ud, zd : array of floats
       Weights for the bilinear interpolation
    iu, iz : array of integers
       Indeces for the bilinear interpolation
    
    Returns
    -------
    emline : array of floats
        Interpolated emission line values
    '''    
    ndat = u.size
    nlines = grid.shape[2]
    emline = np.zeros((nlines, ndat))

    ind = np.where(u > c.notnum)[0]
    if (ind.size < 1):
        print('WARNING (gne_photio.interp_u_z): no adequate log(Us) found')
        return emline
    
    # Extract relevant indices for vectorized operation
    iu_ind = iu[ind]; ud_ind = ud[ind, np.newaxis]
    iz_ind = iz[ind]; zd_ind = zd[ind, np.newaxis]
    
    # Get the four corner values for all points simultaneously
    q11 = grid[iz_ind, iu_ind, :]
    q12 = grid[iz_ind, iu_ind + 1, :]
    q21 = grid[iz_ind + 1, iu_ind, :]
    q22 = grid[iz_ind + 1, iu_ind + 1, :]
    
    # Compute interpolation in U direction first, then in Z direction
    # Broadcasting handles the operations across all emission lines simultaneously
    u_interp1 = q11 * (1 - ud_ind) + q12 * ud_ind  # Upper Z interpolation
    u_interp2 = q21 * (1 - ud_ind) + q22 * ud_ind  # Lower Z interpolation
    
    # Final interpolation in Z direction
    result = u_interp1 * (1 - zd_ind) + u_interp2 * zd_ind
    
    # Place results in the output array
    emline[:, ind] = result.T
    
    return emline


def get_lines_gutkin16(lu, lnH, lzgas, xid_phot=0.3,
                     co_phot=1,imf_cut_phot=100,verbose=True):
    '''
    Get the interpolations for the emission lines,
    using the tables
    from Gutkin et al. (2016) (https://arxiv.org/pdf/1607.06086.pdf)

    Parameters
    ----------
    lu : floats
     U of the galaxies per component.
    lnH : floats
     ne of the galaxies per component (cm^-3).
    lzgas : floats
     Metallicity of the galaxies per component (log10(Z))
    xid_phot : float
       Dust-to-metal ratio
    co_phot : float
       C/O ratio
    imf_cut_phot : float
       Solar mass high limit for the IMF
    verbose : boolean
       If True print out messages
      
    Returns
    -------
    nebline : array of floats
       Line luminosity per component
       Units: Lbolsun per unit SFR(Msun/yr) for 10^8yr, assuming Chabrier
    '''

    photmod = 'gutkin16'

    # Read line names
    line_names = c.line_names[photmod]
    nemline = len(line_names)

    # Initialize the matrix to store the emission lines
    ndat = lu.shape[0]
    ncomp = lu.shape[1]
    nebline = np.zeros((ncomp,nemline,ndat))

    # Get table limits
    minU, maxU = get_limits(propname='logUs', photmod=photmod)
    minnH, maxnH = get_limits(propname='nH', photmod=photmod)
    minZ, maxZ = get_limits(propname='Z', photmod=photmod)

    minZ = np.log10(minZ); maxZ = np.log10(maxZ)

    # Read grid of Zs
    zmet_str = c.zmet_str[photmod]
    nzmet, zmets, lzmets = get_Zgrid(zmet_str)

    zmet_str_reduced = c.zmet_str_reduced[photmod]
    nzmet_reduced, zmets_reduced, lzmets_reduced = get_Zgrid(zmet_str_reduced)

    # Read grid of Us
    logubins = c.lus_bins[photmod]
    nu = len(logubins)
    
    # Store grids for different nH values (different Z grids)
    nHbins = c.nH_bins[photmod]
    nnH = len(nHbins)

    emline_grid1 = np.zeros((nzmet_reduced,nu,nemline))
    emline_grid2 = np.zeros((nzmet,nu,nemline))
    emline_grid3 = np.zeros((nzmet,nu,nemline))
    emline_grid4 = np.zeros((nzmet_reduced,nu,nemline))

    # Store the photoionisation model tables into matrices
    for k, zname in enumerate(zmets):
        infile = get_zfile(zmet_str[k],photmod=photmod)
        io.check_file(infile,verbose=True)
        ih = io.get_nheader(infile)
        
        with open(infile,'r') as ff:
            iline = -1.
            for line in ff:
                iline += 1

                if iline<ih:continue

                data = np.array((line.split()))
                u = float(data[0])
                xid = float(data[1])
                nH = float(data[2])
                co = float(data[3])
                imf_cut = float(data[4])

                l = 0; kred = 0
                if xid == xid_phot and co == co_phot and imf_cut == imf_cut_phot:
                    l = np.where(logubins==u)[0][0]

                    
                    if nH==10 or nH==100 or nH==1000 or nH==10000:
                        if nH==10 or nH==10000:
                            # Reduced metalliticy grid
                            if k==0:
                                kred = 0
                            if k==4:
                                kred = 1
                            if k==9:
                                kred = 2
                            if k==12:
                                kred = 3
                        for j in range(nemline):
                            if nH == 10:
                                emline_grid1[kred,l,j] = float(data[j+5])
                            if nH == 100:
                                emline_grid2[k,l,j] = float(data[j+5])
                            if nH == 1000:
                                emline_grid3[k,l,j] = float(data[j+5])
                            if nH == 10000:
                                emline_grid4[kred,l,j] = float(data[j+5])
        ff.close()

    # Interpolate in all three grids: logUs, logZ, nH
    for comp in range(ncomp):
        ind = np.where(lu[:,comp] != c.notnum)[0]

        # Calculate the weights for interpolating linearly u and reduced z
        uu = lu[:,comp]
        ud, iu = st.interpl_weights(uu,logubins) 

        zz = lzgas[:,comp]
        zd, iz = st.interpl_weights(zz,lzmets_reduced) 

        # Interpolate for each line over u and reduced z
        emline_int1 = interp_u_z(emline_grid1,uu,ud,iu,zd,iz)
        emline_int4 = interp_u_z(emline_grid4,uu,ud,iu,zd,iz) 
    
        # Calculate the weights for interpolating linearly z
        xx = lzgas[:,comp]
        zd, iz = st.interpl_weights(xx,lzmets) 

        # Interpolate for each line over u and z
        emline_int2 = interp_u_z(emline_grid2,uu,ud,iu,zd,iz)
        emline_int3 = interp_u_z(emline_grid3,uu,ud,iu,zd,iz) 
    
        # Interpolate over nH
        #xx = lnH[:,comp] ###here
        #nebline_c = interp_nH(emline_grid2,uu,ud,iu,zd,iz)
        for n in ind:
            if (lnH[:,comp][n] > 2. and lnH[:,comp][n] <= 3.):
                dn = (lnH[:,comp][n] -2.)/(3. - 2.)
                for k in range(nemline):
                    nebline[comp][k][n] = (1.-dn)*emline_int2[k][n] + (dn)*emline_int3[k][n]
    
            elif (lnH[:,comp][n] > 1. and lnH[:,comp][n] <= 2.):
                dn = (lnH[:,comp][n] -1.)/(2. - 1.)
                for k in range(nemline):
                    nebline[comp][k][n] = (1.-dn)*emline_int1[k][n] + (dn)*emline_int2[k][n]
    
            elif (lnH[:,comp][n] > 3. and lnH[:,comp][n]<=4.):
                dn = (lnH[:,comp][n] - 3.)/(4. - 3.)
                for k in range(nemline):
                    nebline[comp][k][n] = (1. - dn) * emline_int3[k][n] + (dn) * emline_int4[k][n]
                # print('hay mayor que 3')
    
            elif (lnH[:,comp][n] <= 1.):
                for k in range(nemline):
                    nebline[comp][k][n] = emline_int1[k][n]
            elif (lnH[:,comp][n] > 4.):
                for k in range(nemline):
                    nebline[comp][k][n] = emline_int4[k][n]
            else:
                print('log(nH)disk out of limits','log(nH)disk = {}'.format(lnH[:,comp][n]))

    return nebline


def get_lines_feltre16(lu, lnH, lzgas, xid_phot=0.5,
                     alpha_phot=-1.7,verbose=True):
    '''
    Get the interpolations for the emission lines,
    using the tables from Feltre+2016 (https://arxiv.org/pdf/1511.08217)
    
    lnH : floats
     ne of the galaxies per component (cm^-3).
    lzgas : floats
     Metallicity of the galaxies per component (log10(Z))
    xid_phot : float
     Dust-to-metal ratio for the Feltre et. al. photoionisation model.
    alpha_phot : float
     Alpha value for the Feltre et. al. photoionisation model.
    verbose : boolean
      If True print out messages
      
    Returns
    -------
    nebline : array of floats
       Line luminosities per galaxy component.
       Units: Lsun for L_AGN = 10^45 erg/s
    '''

    photmod = 'feltre16'
    
    # Read line names
    line_names = c.line_names[photmod]
    nemline = len(line_names)

    # Initialize the matrix to store the emission lines
    ndat = lu.shape[0]
    ncomp = lu.shape[1]
    nebline = np.zeros((ncomp,nemline,ndat))

    # Get table limits
    minU, maxU = get_limits(propname='logUs', photmod=photmod)
    minnH, maxnH = get_limits(propname='nH', photmod=photmod)
    minZ, maxZ = get_limits(propname='Z', photmod=photmod)

    minZ = np.log10(minZ); maxZ = np.log10(maxZ)

    # Read grid of Zs
    zmet_str = c.zmet_str[photmod]
    nzmet, zmets, lzmets = get_Zgrid(zmet_str)

    # Read grid of Us
    logubins = c.lus_bins[photmod]
    nu = len(logubins)
    
    # Store grids for different nH values (different Z grids)
    nHbins = c.nH_bins[photmod]
    nnH = len(nHbins)

    emline_grid1 = np.zeros((nzmet,nu,nemline))
    emline_grid2 = np.zeros((nzmet,nu,nemline))
    emline_grid3 = np.zeros((nzmet,nu,nemline))

    for k, zname in enumerate(zmets):
        infile = get_zfile(zmet_str[k],photmod=photmod)
        io.check_file(infile,verbose=True)
        ih = io.get_nheader(infile)

        with open(infile,'r') as ff:
            iline = -1.
            for line in ff:
                iline += 1

                if iline<ih:continue
                data = np.array((line.split()))
                u = float(data[0])
                xid = float(data[1])
                nH = float(data[2])
                alpha = float(data[3])

                l = 0
                if xid==xid_phot and alpha==alpha_phot:
                    l = np.where(logubins==u)[0][0]

                    if nH==100 or nH==1000 or nH==10000:
                        for j in range(nemline):
                            if nH == 100:
                                emline_grid1[k,l,j] = float(data[j+4])
                            if nH == 1000:
                                emline_grid2[k,l,j] = float(data[j+4])
                            if nH == 10000:
                                emline_grid3[k,l,j] = float(data[j+4])
        ff.close()

    # Interpolate in all three grids: logUs, logZ, nH
    for comp in range(ncomp):
        ind = np.where(lu[:,comp] != c.notnum)[0]

        emline_int1 = np.zeros((nemline, ndat))
        emline_int2 = np.zeros((nemline, ndat))
        emline_int3 = np.zeros((nemline, ndat))

        # Calculate the weights for interpolating linearly u and reduced z
        uu = lu[:,comp]
        ud, iu = st.interpl_weights(uu,logubins) 

        zz = lzgas[:,comp]
        zd, iz = st.interpl_weights(zz,lzmets) 

        # Interpolate for each line over u and z
        emline_int1 = interp_u_z(emline_grid1,uu,ud,iu,zd,iz)
        emline_int2 = interp_u_z(emline_grid2,uu,ud,iu,zd,iz)
        emline_int3 = interp_u_z(emline_grid3,uu,ud,iu,zd,iz) 
    
        # Interpolate over ne
        # use gas density in disk logned
        for n in ind:
            if (lnH[:,comp][n] > 2. and lnH[:,comp][n] <= 3.):
                dn = (lnH[:,comp][n] -2.)/(3. - 2.)
                for k in range(nemline):
                    nebline[comp][k][n] = (1.-dn)*emline_int1[k][n] + (dn)*emline_int2[k][n]
    
            elif (lnH[:,comp][n] > 3. and lnH[:,comp][n] <= 4.):
                dn = (lnH[:,comp][n] - 3.)/(4. - 3.)
                for k in range(nemline):
                    nebline[comp][k][n] = (1. - dn) * emline_int2[k][n] + (dn) * emline_int3[k][n]
                # print('hay mayor que 3')
    
            elif (lnH[:,comp][n] <= 2.):
                for k in range(nemline):
                    nebline[comp][k][n] = emline_int1[k][n]
            elif (lnH[:,comp][n] > 4.):
                for k in range(nemline):
                    nebline[comp][k][n] = emline_int3[k][n]
            else:
                print('log(ne)disk out of limits','log(ne)disk = {}'.format(lnH[:,comp][n]))
                
    return nebline


def get_lines(lu, lnH, lzgas, photmod='gutkin16',xid_phot=0.3,
              co_phot=1,imf_cut_phot=100,alpha_phot=-1.7, verbose=True):
    '''
    Get the emission lines

    Parameters
    ----------
    lu : floats
       U of the galaxies per component.
    lnH : floats
       ne of the galaxies per component (cm^-3).
    lzgas : floats
       Metallicity of the galaxies per component (log10(Z))
    photomod : string
       Name of the considered photoionisation model.
    xid_phot : float
       Dust-to-metal ratio for the photoionisation model
    co_phot : float
       C/O ratio  for the photoionisation model
    imf_cut_phot : float
       Solar mass high limit for the IMF  for the photoionisation model
    alpha_phot : float
       Alpha value for the AGN photoionisation model.
    verbose : boolean
       If True print out messages

    Returns
    -------
    nebline : array of floats
        Line luminosity per galaxy component, if relevant.
        Units depend on the photoionisation model.
    '''

    if photmod not in c.photmods:
        if verbose:
            print('STOP (gne_photio.get_lines): Unrecognised model to get emission lines.')
            print('                Possible photmod= {}'.format(c.photmods))
        sys.exit()
    elif (photmod == 'gutkin16'):
        nebline = get_lines_gutkin16(lu,lnH,lzgas,xid_phot=xid_phot,co_phot=co_phot,
                                   imf_cut_phot=imf_cut_phot,verbose=verbose)
    elif (photmod == 'feltre16'):
        nebline = get_lines_feltre16(lu,lnH,lzgas,xid_phot=xid_phot,
                                   alpha_phot=alpha_phot,verbose=verbose)

    return nebline

