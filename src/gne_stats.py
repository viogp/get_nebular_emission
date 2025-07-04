#! /usr/bin/env python
"""
Some useful functions
  percentiles(xbins,xarray,yarray,per): obtains percentiles of yarray in xbins
  convert_to_stdev(grid): normalised a grid to cumulative standard deviations.
  n_gt_x(x,array): returns the number of elements in the array larger than each of the values in x.
  chi2(obs,model,err): returns the chi^2 for a model
 ...
"""
import sys
import numpy as np
import src.gne_const as c

def percentiles(val, data, weights=None):
    """
    Examples
    --------
    >>> import numpy as np
    >>> import stats
    >>> data = np.array(np.arange(0.,100.,10.))
    >>> stats.percentiles(0.5,data)
    >>> 45.0
    """

    if (val < 0 or val > 1):
        sys.exit('STOP percentiles: 0<val<1')

    if (weights is None):
        ws = np.zeros(shape=(len(data)));
        ws.fill(1.)
    else:
        ws = weights

    data = np.array(data);
    ws = np.array(ws)
    ind_sorted = np.argsort(data)  # Median calculation from wquantiles
    sorted_data = data[ind_sorted];
    sorted_weights = ws[ind_sorted]

    num = np.cumsum(sorted_weights) - 0.5 * sorted_weights
    den = np.sum(sorted_weights)
    if (den != 0):
        pn = num / den
        percentiles = np.interp(val, pn, sorted_data)
    else:
        sys.exit('STOP percentiles: problem with weights')
    return percentiles


def perc_2arrays(xedges, xarray, yarray, val, weights=None, nmin=None):
    """
    Returns percentiles of yarray over xbins
    Parameters
    ----------
    xedges : array of floats
        Bin edges on the x-axis
    xarray : array of floats
        Values for the x-axis
    yarray : array of floats
        Values for the y-axis
    val : float from 0 to 1
        Value used to determine the percentile to be calculated
    weights : array of floats
        Weights for the yarray values
    nmin : integer
        Minimal number of points to be considered in a bin
    Returns
    -------
    apercentile : string of floats
       Percentiles of the yarray within xbins
    Examples
    --------
    >>> import numpy as np
    >>> import stats
    >>> xedges = np.array([0.,1.,2.])
    >>> xarray = np.array(np.arange(0.,2.,0.1))
    >>> yarray = np.append(np.array(np.arange(1.,11.,1.)),np.array(np.arange(1.,11.,1.)))
    >>> stats.perc_2arrays(xedges,xarray,yarray,0.5)
    >>> array([5.5, 5.5])
    """
    xlen = len(xedges) - 1
    apercentile = np.zeros(shape=(xlen));
    apercentile.fill(-999.)

    if (len(xarray) != len(yarray)):
        sys.exit('ERROR @ perc_2arrays: The lenght of the input arrays should be equal.')

    if (nmin is None):
        nmin = 1

    for i in range(xlen):
        ind = np.where((xarray >= xedges[i]) & (xarray < xedges[i + 1]))
        # We require at least nmin points per bin
        if (np.shape(ind)[1] >= nmin):
            data = yarray[ind]

            if len(data)<5:
                apercentile[i] = -999.
            elif (weights is None):
                apercentile[i] = percentiles(val, data)
                # print(val, apercentile[i])
            else:
                if (len(weights) != len(yarray)):
                    sys.exit(
                        'ERROR @ perc_2arrays: The lenght of the weights array should be equal to the input array.')

                ws = weights[ind]
                apercentile[i] = percentiles(val, data, weights=ws)

    return apercentile


def ensure_2d(arr, axis=0):
    if len(arr.shape) == 1:
        # If 1D array, expand along the specified axis
        return np.expand_dims(arr, axis=axis)
    return arr


def av_2arrays(xbins, xarray, yarray, weights, nmin):
    """ Returns average of yarray over xbins"""
    xlen = len(xbins) - 1
    av_2arrays = np.zeros(shape=(xlen));
    av_2arrays.fill(-999.)

    if len(xarray) != len(yarray):
        sys.exit('ERROR @ perc_2arrays: The lenght of the input arrays should be equal.')

    for i in range(xlen):
        ind = np.where((xarray >= xbins[i]) & (xarray < xbins[i + 1]))
        # We require at least nmin points per bin
        num = np.shape(ind)[1]
        if (num > nmin):
            data = yarray[ind];
            ws = weights[ind]
            dw = ws * data
            av_2arrays[i] = np.sum(dw) / np.sum(ws)

    return av_2arrays


def get_cumulative_2Ddensity(xin, yin, n_grid=100):
    """
    Calculate the normalised cumulative values for a 2D distribution.
    Useful for then plotting contours with percentiles.
    
    Parameters:
    -----------
    xin, yin : 1D array 
        Input coordinates
    n_grid : int
        Number of bins for 2D histogram

    Returns:
    --------
    xx, yy : 1D array
        Centers of the output grid
    zz : 1D array
        Normalised (to 1) cumulative density
    """
    # Create the grid
    xmin, xmax = xin.min(), xin.max()
    ymin, ymax = yin.min(), yin.max()
    xylims = [[xmin, xmax], [ymin, ymax]]
    
    # Calculate the 2D histogram and edges of the grid
    hist2D, xedges, yedges = np.histogram2d(xin,yin,bins=n_grid,range=xylims)

    # Create mesh grid from bin centers
    xx = (xedges[:-1] + xedges[1:])/2.
    yy = (yedges[:-1] + yedges[1:])/2.
    xx, yy = np.meshgrid(xx, yy)

    # Flatten the 2D histogram into a 1D array
    hist = hist2D.ravel()
    
    # Sort in descending order
    i_sort = np.argsort(hist)[::-1]
    hsort = hist[i_sort]

    # Normalised (max=1) the cumulative distribution
    hcumsum = hsort.cumsum()
    hcumsum /= hcumsum[-1]

    # Return density in adequate shape
    shape = hist2D.shape
    i_unsort = np.argsort(i_sort)
    zz = hcumsum[i_unsort].reshape(shape).T
    return xx, yy, zz


def n_gt_x(xedges, array):
    y = np.zeros(len(xedges))

    for i, xedge in enumerate(xedges):
        ind = np.where(array > xedge)
        y[i] = np.shape(ind)[1]

    return y


def locate_interval(val, edges):
    '''
    Get the index, i, of the interval, [), to which val belongs.
    If outside the limits, using values -1 or the number of bins+1.

    Parameters
    ----------
    val : int or float or array of ints or floats
        Value to evaluate
    edges : array of int or floats
        Array of the n edges for the (n-1) intervals
        
    Returns
    -------
    jl : integer
        Index of the interval, [edges(jl),edges(jl+1)), where val is place.
    '''

    n = edges.size
    jl = np.searchsorted(edges, val, side='right') - 1
    jl = np.clip(jl, -1, n - 1)
    return jl


def interpl_weights(xx,edges):
    '''
    Get linear interpolation weights: xd=(x-x1)/(x2-x1)
    Values outside the edges limits are given the weights
    corresponding to the minimum and maximum edge values.
    
    Parameters
    ----------
    xx : float (or int) or array of floats (or int)
        Values to be evaluated
    edges : array of floats (or int)
        Array of the n edges for the (n-1) intervals
        
    Returns
    -------
    xd : float or list of float (or int)
        Weights for linear interpolation
    ix : int or list of ints
        Lower index of the interval the value belongs to
    '''
    # Size of the 1D grid
    n = edges.size

    # If scalar, turn it into array
    scalar = False
    if isinstance(xx, (float, int)): # Floats
        scalar = True
        xx = np.array([xx])
        
    # Locate intervals and handle boundaries
    ix = locate_interval(xx, edges)

    # Initialize interpolation weights
    xd = np.zeros(len(ix))
    
    # Calculate interpolation weights
    ind = np.where((ix>-1) & (ix<n-1))
    if (np.shape(ind)[1]>0):
        ii = ix[ind]
        xd[ind] = (xx[ind] - edges[ii])/(edges[ii + 1] - edges[ii])
    
    # Handle boundaries
    xd[ix > n-2] = 1.0
    ix = np.clip(ix, 0, n-2)

    outxd = np.asarray(xd)
    outix = np.asarray(ix,dtype=int)
    if scalar:
        outxd = outxd[0]
        outix = outix[0]
    return outxd, outix


def bilinear_interpl(xx, yy, xedges, yedges, zedges, verbose=False):
    """
    Bilinear interpolation. If the points to be interpolated are outside
    the boundaries of the coordinate grid, the resulting interpolated values
    are evaluated at the boundary.
    
    Parameters
    ----------
    xx : 1D array or scalar, shape N
        x-coordinates for point(s) to be interpolated.
    yy : 1D array or scalar, shape N
        y-coordinates for point(s) to be interpolated.
    xedges : 1D array, shape Nx
        x-coordinates of data points zp (grid coordinates).
    yedges : 1D array, shape Ny
        y-coordinates of data points zp (grid coordinates).
    zedges : array, shape (Nx, Ny) or (Nx, Ny, M)
        Data points on grid from which to interpolate.
    verbose : bool, optional
        If True, print intermediate calculation steps.
    
    Returns
    -------
    zz : scalar or array, shape N if 2D zedges or (N,M) if 3D zedges
        Interpolated values at given point(s).
    """       
    # If scalar, turn it into array
    scalar = False
    if isinstance(xx, (float, int)): 
        scalar = True
        xx = np.array([xx])
        yy = np.array([yy])
        
    # Initialize input validation
    n = xx.size
    if (yy.size != n):
        sys.exit('STOP bilinear_interpl: input sizes different for x and y')
        
    # Get the intervals and weights
    xd, ix = interpl_weights(xx, xedges)
    yd, iy = interpl_weights(yy, yedges)
    if verbose: print('xd,ix=', xd, ix, '\nyd,iy=', yd, iy)
    
    # Handle both 2D and 3D zedges input
    if verbose: print(zedges.ndim,'D zedges')
    if zedges.ndim == 2:
        zz = np.zeros(n)
        
        # Get the four corner values for all points
        c00 = zedges[ix, iy]
        c01 = zedges[ix, iy+1]
        c10 = zedges[ix+1, iy]
        c11 = zedges[ix+1, iy+1]
        if verbose: print('cij=', c00, c01, c10, c11)

        # Linear interpolation ove x
        c0 = c00*(1-xd) + c10*xd
        c1 = c01*(1-xd) + c11*xd
        if verbose: print('c0=',c0,'\nc1=',c1)

        # Linear interpolation ove y
        zz = c0*(1-yd) + c1*yd

        if scalar: 
            zz = zz[0]

    elif zedges.ndim == 3:
        m = zedges.shape[2]
        zz = np.zeros((n, m))

        ## Get the four corner values for each z-layer
        c00 = zedges[ix, iy, :]
        c01 = zedges[ix, iy+1, :]
        c10 = zedges[ix+1, iy, :]
        c11 = zedges[ix+1, iy+1, :]
        if verbose: print('c00=',c00,'\nc01=',c01,'\nc10=',c10,'\nc11=',c11)
        
        # Linear interpolation over x
        c0 = c00*(1-xd[:, np.newaxis]) + c10*xd[:, np.newaxis]
        c1 = c01*(1-xd[:, np.newaxis]) + c11*xd[:, np.newaxis]
        if verbose: print('c0=',c0,'\nc1=',c1)
        
        # Linear interpolation over y
        zz = c0*(1-yd[:, np.newaxis]) + c1*yd[:, np.newaxis]

        if scalar: 
            zz = zz[0,:]

    else:
        raise ValueError('bilinear_interpl: zedges must be a 2D or 3D array')

    return zz


def chi2(obs, model, err2):
    '''
    Get the chi^2 for a given model

    Parameters:
    obs : array of floats
        The observatioins or target values
    model : array of floats
        The model values (should be the same length as obs)
    err2 : array of floats
        The error**2 of the observations
    Returns:
    val : float
       chi^2 values
    '''
    val = 0.
    for i, iobs in enumerate(obs):
        val = val + (iobs - model[i]) ** 2 / err2[i]
    return val


def get_err2Pk(k, Pk, dk, N, vol):
    '''
    Get the error of the Power Spectrum
    Parameters:
    -----------
    k : numpy array of floats
       Wavenumber of the modes
    Pk : numpy array of floats
       Power spectrum at each k
    dk : float
      Size of the step for k
    N : float
      Number of elements
    vol : float
      Considered volume
    Returns:
    --------
    err2Pk : float
      Square of the power spectrum error
    '''

    err2Pk = None

    if (len(k) != len(Pk)):
        print('STOP (stats.get_err2Pk): k and Pk are different lengths')
        return err2Pk

    norm = (2 * np.pi) ** 2 / (k * k * dk * vol)
    err2Pk = norm * (Pk + vol / N) ** 2

    return err2Pk


def components2tot(comps, log10input=True):
    '''
    Calculate the total property as log10(sum(comps)),
    from provided information on components,
    if log10input, otherwise return sum(comps).

    Parameters
    ----------
    comps : array of floats
        Array with properties
    log10input: boolean
        True if input components as log10(prop)
    
    Returns
    -------
    log_tot : array of floats
        Total log10(sum(comps))
    '''
    ncomp = comps.shape[1]
    if ncomp > 1:
        log_tot = np.zeros(comps.shape[0]); log_tot.fill(c.notnum)
        ptot = np.zeros(log_tot.shape)
        for ii in range(ncomp):
            props = np.copy(comps[:,ii])
            mask = props>c.notnum
            if log10input:
                ptot[mask] = ptot[mask] + 10**props[mask]
            else:
                ptot[mask] = ptot[mask] + props[mask]

        if log10input:
            mask = ptot>0
            log_tot[mask] = np.log10(ptot[mask])
        else:
            log_tot = ptot
    else:
        log_tot = np.copy(comps)
        
    return log_tot


def romberg(f, a, b, max_steps=10, acc=1e-8):
    """
    Calculates the integral of a function using Romberg integration.
    (Adapted from Wikipedia)
    
    Args:
        f: The function to integrate.
        a: Lower limit of integration.
        b: Upper limit of integration.
        max_steps: Maximum number of steps.
        acc: Desired accuracy.

    Returns:
        The approximate value of the integral.
    """
    R1, R2 = [0] * max_steps, [0] * max_steps  # Buffers for storing rows
    Rp, Rc = R1, R2  # Pointers to previous and current rows

    h = b - a  # Step size
    Rp[0] = 0.5 * h * (f(a) + f(b))  # First trapezoidal step

    for i in range(1, max_steps+1):
        h /= 2.0

        # Compute Rc[0]=R(i,0)
        ep = 2 ** (i - 1); sumf = 0.
        for k in range(1, ep + 1):
            sumf += f(a + (2*k - 1)*h)
        Rc[0] = 0.5*Rp[0] + h*sumf  

        # Compute Rc[j]=R(i,j)
        for j in range(1, i + 1):
            m4 = 4**j
            Rc[j] = (m4*Rc[j - 1] - Rp[j - 1])/(m4 - 1)  

        # Stop the calculation if the desired accuracy has been achieved
        if i > 1 and abs(Rp[i - 1] - Rc[i]) < acc:
            return Rc[i]

        # Swap Rn and Rc for next iteration
        Rp, Rc = Rc, Rp
    return Rp[max_steps]  # Return our best guess


def vol_sphere(R):
    '''
    Calculate the volume of a sphere at a given a radius

    Parameters
    ----------
    R : float
      Radius of the sphere.
     
    Returns
    -------
    V : float
    '''
    
    V = (4./3.)*np.pi*R**3
    return V
