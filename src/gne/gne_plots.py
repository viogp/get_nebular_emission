"""
.. moduleauthor:: Violeta Gonzalez-Perez <violetagp@protonmail.com>
.. contributions:: Olivia Vidal <ovive.pro@gmail.com>
.. contributions:: Julen Expósito-Márquez <expox7@gmail.com>
"""

import os.path
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.colors as mcol
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.offsetbox as moffbox

import warnings
warnings.filterwarnings('ignore', message='Input line .* contained no data')

import gne.gne_const as c
import gne.gne_io as io
import gne.gne_stats as st
import gne.gne_plot_obs as obs
from gne.gne_stats import n_gt_x
from gne.gne_photio import get_limits,read_gutkin16_grids,read_feltre16_grids
from gne.gne_cosmology import set_cosmology
from gne.gne_flux import flux2L
import gne.gne_style
plt.style.use(gne.gne_style.style1)

cmap = 'magma'
n4contour = 3000
min_Lbol = 42 # Based on Griffin+2020, fig 14
max_Lbol = 50
min_Ms = 8    # To be obtained from sim. res. ###here
max_Ms = 12   # To be obtained from sim. res. ###here

markers = ['o','^', 's', '*','D', 'p', 'h', 'H', '+', 'x', 'v', '<', '>', '|', '_']

def get_ngrid_nlev(nobj):
    ngrid = 100 if nobj > n4contour*4 else 50
    nlev  = None if nobj > n4contour*4 else 3
    return ngrid, nlev


def contour2Dsigma(n_levels=None,color='darkgrey'):
    '''
    Get levels following the standard deviation numbers expected for
    a 2D-Gaussian distribution. Generate colours varying in intensity.
    '''
    if n_levels is not None:
        levels=c.sigma_2Dprobs[0:n_levels]
    else:
        levels=c.sigma_2Dprobs.copy()
    
    nl = len(levels)
    levels.insert(0,0)
    alphas = np.linspace(0.2, 1, nl)[::-1].tolist()
    colors = [(*mcol.to_rgba(color, alpha=a),)for a in alphas]

    return levels,colors


def plot_comp_quartiles(ax, xx, yy, xmin, xmax, tots, ins, cm=plt.cm.tab20):
    """
    Plot quartiles for components and return legend elements.
    
    Parameters:
    -----------
    ax : matplotlib axis
        Axis to plot on
    xin : ndarray
        X values for each component
    yin : ndarray
        Y values for each component
    n_comp : int
        Number of components
    xmin, xmax: float
        Limits for the x-axis
    tots : ndarray
        Total number of elements for each component
    ins : ndarray
        Number within limits for each component
    cm : matplotlib colormap, optional
        Colormap to use
        
    Returns:
    --------
    proxies : list
        List of proxy artists for legend
    labels : list
        List of labels for legend
    """
    proxies = []; labels = []
    
    dx = 0.2    
    xbins = np.arange(xmin,xmax + dx, dx)
    xhist = xbins + dx * 0.5
    ax.set_xlim(xmin, xmax)

    n_comp = np.shape(xx)[1]
    for i in range(n_comp):
        ind = np.where((xx[:, i] > c.notnum) & (yy[:, i] > c.notnum))[0]
        if len(ind) > 0:
            x = xx[ind, i]
            y = yy[ind, i]
            col = np.array([cm(float(i) / n_comp)])

            med = st.perc_2arrays(xbins, x, y, 0.5)        
            upq = st.perc_2arrays(xbins, x, y, 0.84)
            low = st.perc_2arrays(xbins, x, y, 0.16)
            jnd = np.where(med>c.notnum)
            if (np.shape(jnd)[1]>1):
                el = med[jnd] - low[jnd]
                eh = upq[jnd] - med[jnd]
                quart = ax.errorbar(xhist[jnd],med[jnd],yerr=[el,eh],c=col)
                proxies.append(quart)
                
                leg = "{} component {} ({:.1f}% in)".format(
                    int(tots[i]), i, ins[i]*100./tots[i])
                labels.append(leg)
                
    return proxies, labels


def plot_unh(root, endf, subvols=[0], outpath=None,
             metadata=None,verbose=True):
    '''
    Make contour plots of the properties of the ionised regions:
    U_SF, nH_SF and Zcold_SF against M*, sSFR and among themselves,
    and U_AGN, Zcold_AGN against M* and Lbol, if calculated.

    Figures numbered from the top-left, moving right and then down:
    0) M* vs nH_SF        1) sSFR vs nH_SF       2) nH_SF vs U_SF
    3) M* vs U_SF         4) sSFR vs U_SF         5) Zcold_SF vs U_SF
    6) M* vs U_AGN        7) Lbol vs U_AGN       8) Zcold_AGN vs U_AGN

    Parameters
    ----------
    root : string
       Path to input files.
    endf : string
       Ending of input files.
    subvols: list of integers
        List of subvolumes to be considered
    outpath : string
        Path to output, default is output/
    metadata : dictionary
        Cosmology and other metadata information
    verbose : boolean
       If True print out messages.
    '''
    # Get metadata
    photmod_sfr = metadata['photmod_sfr']
    AGN = metadata['AGN']
    if AGN:
        photmod_agn = metadata['photmod_agn']
    redshift = metadata['redshift']

    # Read limits of the photoionisation models
    minU_sf, maxU_sf = get_limits(propname='logUs', photmod=photmod_sfr)
    minZ_sf, maxZ_sf = np.log10(get_limits(propname='Z', photmod=photmod_sfr))
    if AGN:
        minU_ag, maxU_ag = get_limits(propname='logUs', photmod=photmod_agn)
        minZ_ag, maxZ_ag = np.log10(get_limits(propname='Z', photmod=photmod_agn))

    # Read data from each subvolume
    first_vol = True
    for ivol in subvols:
        filenom = os.path.join(root + str(ivol), endf)
        f = h5py.File(filenom, 'r')

        # SF properties, summed over components (stored as log10)
        dd = {'lm_s': st.components2tot(f['data/lm_s']),
              'lssfr': st.components2tot(f['data/lssfr']),
              'lnH_sfr': st.components2tot(f['sfr_data/lnH_sfr']),
              'lu_sfr': st.components2tot(f['sfr_data/lu_sfr']),
              'lz_sfr': st.components2tot(f['sfr_data/lz_sfr'])}

        if AGN:
            dd['lu_agn'] = f['agn_data/lu_agn'][:]
            dd['lz_agn'] = f['agn_data/lz_agn'][:]
            Lagn = f['agn_data/Lagn'][:]
            dd['lLagn'] = np.where(Lagn>0,
                                   np.log10(np.maximum(Lagn,1)),c.notnum)
        f.close()

        if first_vol:
            data = dd
            first_vol = False
        else:
            for key in dd:
                data[key] = np.append(data[key], dd[key], axis=0)

    # Axis labels
    xtit_ms = 'log$_{10}(M_{*}/M_{\\odot})$'
    xtit_ssfr = 'log$_{10}$(sSFR/yr$^{-1}$)'
    xtit_nh = 'log$_{10}(n_{H,\\rm SF}$/cm$^{-3}$)'
    xtit_zsf = 'log$_{10}(Z_{\\rm cold,SF})$'
    xtit_zag = 'log$_{10}(Z_{\\rm cold,AGN})$'
    xtit_lb = 'log$_{10}(L_{\\rm bol}$/erg s$^{-1}$)'
    ytit_nhsf = 'log$_{10}(n_{H,\\rm SF}/cm^{-3})$'
    ytit_usf = 'log$_{10}(U_{\\rm SF})$'
    ytit_uagn = 'log$_{10}(U_{\\rm AGN})$'

    pad = 0.5
    npanel = 9 if AGN else 6
    nrows = npanel//3

    # Panel definition: x-data, x-title, y-data, y-title,
    #                   x-limits, model limits tag ('sfr'/'agn'/None)
    panels = [
        dict(x=data['lm_s'], xtit=xtit_ms, y=data['lnH_sfr'],
             ytit=ytit_nhsf, xlim=(min_Ms-pad, max_Ms+pad), mod=None),
        dict(x=data['lssfr'], xtit=xtit_ssfr, y=data['lnH_sfr'],
             ytit=ytit_nhsf, xlim=None, mod=None),
        dict(x=data['lnH_sfr'], xtit=xtit_nh, y=data['lu_sfr'],
             ytit=ytit_usf, xlim=None, mod=None),
        dict(x=data['lm_s'], xtit=xtit_ms, y=data['lu_sfr'],
             ytit=ytit_usf, xlim=(min_Ms-pad, max_Ms+pad), mod=None),
        dict(x=data['lssfr'], xtit=xtit_ssfr, y=data['lu_sfr'],
             ytit=ytit_usf, xlim=None, mod=None),
        dict(x=data['lz_sfr'], xtit=xtit_zsf, y=data['lu_sfr'],
             ytit=ytit_usf, xlim=None, mod='sfr')]
    if AGN:
        panels.append(dict(x=data['lm_s'], xtit=xtit_ms, y=data['lu_agn'],
                           ytit=ytit_uagn, xlim=(min_Ms-pad, max_Ms+pad),
                           mod=None))
        panels.append(dict(x=data['lLagn'], xtit=xtit_lb, y=data['lu_agn'],
                           ytit=ytit_uagn, xlim=(min_Lbol-pad, max_Lbol+pad),
                           mod=None))
        panels.append(dict(x=data['lz_agn'], xtit=xtit_zag, y=data['lu_agn'],
                           ytit=ytit_uagn, xlim=None, mod='agn'))

    # Prep plots
    fig, axes = plt.subplots(nrows, 3, figsize=(30, 10*nrows),
                             layout='constrained')
    axes = axes.flatten()
    fig.suptitle(f'z = {redshift:.2f}')

    col = 'black'
    for ipan, pan in enumerate(panels):
        ax = axes[ipan]
        xx, yy = pan['x'], pan['y']
        ax.set_xlabel(pan['xtit']); ax.set_ylabel(pan['ytit'])
        ax.minorticks_on()

        # Contours in grey values as in plot_bpts
        ind = np.where((xx > c.notnum) & (yy > c.notnum))[0]
        nsel = len(ind)
        if nsel == 0:
            print('WARNING plots.unh: no valid data in one panel')
            continue

        if nsel > n4contour:
            ngrid, nlev = get_ngrid_nlev(nsel)
            xc, yc, zc = st.get_cumulative_2Ddensity(xx[ind], yy[ind],
                                                     n_grid=ngrid)
            levels, colors = contour2Dsigma(n_levels=nlev, color=col)
            ax.contour(xc, yc, zc, levels=levels, colors=colors, zorder=1)
        else:
            ax.scatter(xx[ind], yy[ind], c=col, s=40, marker='o', zorder=2)

        if pan['xlim'] is not None:
            ax.set_xlim(pan['xlim'])

        # Limits of the photoionisation models (Figs. 5 and 8)
        if pan['mod'] is not None:
            if pan['mod'] == 'sfr':
                umin, umax, zmin, zmax = minU_sf, maxU_sf, minZ_sf, maxZ_sf
                legm = 'SF'
            else:
                umin, umax, zmin, zmax = minU_ag, maxU_ag, minZ_ag, maxZ_ag
                legm = 'AGN'

            colr = 'limegreen'
            ax.set_xlim(zmin-pad, zmax+pad)
            ax.set_ylim(umin-pad, umax+pad)
            ax.add_patch(plt.Rectangle((zmin, umin), zmax-zmin, umax-umin,
                                       ec=colr, ls='-', lw=6,
                                       fc='none', zorder=3))

            # Percentage of galaxies within the model limits
            inw = np.sum((xx[ind] >= zmin) & (xx[ind] <= zmax) &
                         (yy[ind] >= umin) & (yy[ind] <= umax))
            per = inw*100./nsel
            ax.text(zmin+0.1, umax-0.4,
                    f'{per:.1f}%', color=colr, fontsize='large',zorder=4)
            if verbose:
                print(f'    {per:.1f}% of galaxies '
                      f'({inw} out of {nsel}) within model limits')

    # Output
    nom = io.get_plotfile(root,endf,'unh')
    plt.savefig(nom)
    if verbose:
        print(f'* U and nH plots: {nom}')

    return nom



def plot_model_bpt_grids(photmod='gutkin16',xid=0.3,co=1,imf_cut=100,
                         alpha=-1.7,verbose=True):
    '''
    Plot photoionisation grids on 2 BPT diagrams.
    
    Parameters
    ----------
    photmod : string
       Name of the photoionisation model to be plotted
    verbose : boolean
       If True print out messages.

    Return
    ------
    outpath : string
       Name of output plot within output/photoio_grids
    '''
    # Prep plots
    fig, (axn, axs) = plt.subplots(1, 2, figsize=(32, 17),
                                   layout='constrained')
    col = 'darkgrey'
    #plt.subplots_adjust(right=0.85, top=0.9) 
    ytit = 'log$_{10}$([OIII]$\\lambda$5007/H$\\beta$)'
    xmins = [-1.9,-1.9]
    xmaxs = [0.8,0.9]
    ymins = [-1.5,-2.1]
    ymaxs = [1.5,1.6]
    for ii, bpt in enumerate(['NII','SII']):
        if bpt=='NII':
            xtit = 'log$_{10}$([NII]$\\lambda$6584/H$\\alpha$)'
            axn.set_xlim(xmins[ii], xmaxs[ii])
            axn.set_ylim(ymins[ii], ymaxs[ii])
            axn.set_xlabel(xtit); axn.set_ylabel(ytit)
        elif bpt=='SII':
            xtit = 'log$_{10}$([SII]$\\lambda\\lambda$6717,6731/H$\\alpha$)'
            axs.set_xlim(xmins[ii], xmaxs[ii])
            axs.set_ylim(ymins[ii], ymaxs[ii])
            axs.set_xlabel(xtit); axs.set_ylabel(ytit)
            
        xobs, yobs, obsdata = obs.get_obs_bpt(0.,bpt)
        if obsdata:
            nobs = len(xobs)
            if nobs > n4contour:
                ngrid, nlev = get_ngrid_nlev(nobs)
                x,y,z = st.get_cumulative_2Ddensity(xobs,yobs,n_grid=ngrid)
                levels, colors = contour2Dsigma(n_levels=nlev,color=col)
                if bpt=='NII':
                    contour = axn.contourf(x,y,z,levels=levels,colors=colors)
                elif bpt=='SII':
                    contour = axs.contourf(x,y,z,levels=levels,colors=colors)
            else:
                if bpt=='NII':
                    axn.scatter(xobs,yobs,c=col)
                elif bpt=='SII':
                    axs.scatter(xobs,yobs,c=col)
            
    for ii, bpt in enumerate(['NII','SII']):
        # Lines
        xline = np.arange(xmins[ii],xmaxs[ii]+0.1, 0.03)
        if bpt=='NII':
            yline = obs.lines_BPT(xline,bpt,'Kauffmann2003')
            axn.plot(xline,yline,'k--')

            yline = obs.lines_BPT(xline,bpt,'Kewley2001')
            axn.plot(xline,yline,'k-')
            
        elif bpt=='SII':
            yline = obs.lines_BPT(xline,bpt,'Kewley2001')
            axs.plot(xline,yline,'k-')

            ylinel = obs.lines_BPT(xline,bpt,'Kewley2006')
            axs.plot(xline[ylinel>yline],ylinel[ylinel>yline],'k-.')


    # Read grids of photoionisation models
    if photmod == 'gutkin16':
        grid1,grid2,grid3,grid4 = read_gutkin16_grids(xid, co, imf_cut)
        grids = [grid1, grid2, grid3, grid4]
        nl = 5
        col_ha = 10-nl
        col_hb = 6-nl 
        col_o3 = 8-nl     #[OIII]5007
        col_n2 = 11-nl    #[NII]6584
        col_s2_a = 12-nl  #[SII]6717
        col_s2_b = 13-nl  #[SII]6731        
    elif photmod == 'feltre16':
        grid1,grid2,grid3 = read_feltre16_grids(xid, alpha)
        grids = [grid1, grid2, grid3]
        nl = 4
        col_ha = 10-nl
        col_hb = 5-nl 
        col_o3 = 7-nl     #[OIII]5007
        col_n2 = 11-nl    #[NII]6584  
        col_s2_a = 12-nl  #[SII]6717
        col_s2_b = 13-nl  #[SII]6731

    for i, grid in enumerate(grids):
        nz = grid.shape[0]
        nu = grid.shape[1]

        size = 100 + i * 30
        for iz in range(nz):
            color = cm.tab20(iz % 20)
            
            for iu in range(nu):
                marker = markers[iu % len(markers)]
                
                el = grid[iz,iu,:]
                y = np.log10(el[col_o3]/el[col_hb])
                for ii, bpt in enumerate(['NII','SII']):
                    if bpt=='NII':
                        x = np.log10(el[col_n2]/el[col_ha])
                        axn.scatter(x,y,s=size,c=[color],marker=marker)
                    elif bpt=='SII':
                        s2 = el[col_s2_a] + el[col_s2_b]
                        x = np.log10(s2/el[col_ha])
                        axs.scatter(x,y,s=size,c=[color],marker=marker)

    # Add legend on Z values
    zvals = c.zmet_str[photmod]
    z_handles = []
    for iz, z_val in enumerate(zvals):
        color = cm.tab20(iz % 20)
        z_handle = mpatches.Patch(color=color, label=f'0.{z_val}')
        z_handles.append(z_handle)
    legend1 = fig.legend(handles=z_handles, loc='center left',
                         bbox_to_anchor=(1.02, 0.5), 
                         title="Z$_{gas}$", frameon=False)
    fig.add_artist(legend1)

    # Add legend on U values
    uvals = c.lus_bins[photmod]
    u_handles = []
    for iu, u_val in enumerate(uvals):
        marker = markers[iu % len(markers)]
        u_handle = mlines.Line2D([], [], color='black',
                                 marker=marker, ls='None',
                                 markersize=10, label=f'{u_val}')
        u_handles.append(u_handle)
    legend2 = fig.legend(handles=u_handles, 
                         loc='upper center',
                         bbox_to_anchor=(0.5, 1.1),
                         title='log$_{10}$ U',
                         ncol=min(5, len(u_handles)),
                         frameon=False)
    legend2._legend_box.align = "left"
    legend2._legend_box.sep = 7 
    plt.setp(legend2.get_title(), ha='right')
    legend2._legend_box = moffbox.VPacker(
        pad=0, sep=0, align="left",
        children=[
            moffbox.HPacker(
                pad=0, sep=5, align="center",
                children=[legend2._legend_box.get_children()[0],
                          legend2._legend_box.get_children()[1]])])
    fig.add_artist(legend2)
    
    # Add legend on model information
    if photmod == 'gutkin16':
        legend_model = (f'Gutkin+16\n'
                        f'$\\xi_d$ = {xid}\n'
                        f'C/O = {co} (C/O)$_\\odot$\n'
                        f'M(IMF)$<{imf_cut}$ M$_\\odot$')
    elif photmod == 'feltre16':
        legend_model = (f'Feltre+16\n'
                        f'$\\xi_d$ = {xid}\n'
                        f'$\\alpha$ = {alpha}\n')
    axn.text(0.05, 0.97, legend_model, transform=axn.transAxes,
             verticalalignment='top')

    # Output
    pltpath = 'output/plots/photoio_grids/'
    io.create_dir(pltpath)
    if (photmod == 'gutkin16'):
        bptnom = pltpath+photmod+'_xi'+str(xid)+'_bpts.pdf'
    elif (photmod == 'feltre16'):
        bptnom = pltpath+photmod+'_alpha'+str(abs(alpha))+'_bpts.pdf'

    plt.savefig(bptnom)
    if verbose:
         print(f'* Photoionisation model grids on BPT plots: {bptnom}')
    
    return bptnom


def plot_bpts(root, endf, subvols=[0], outpath=None,
              metadata=None,verbose=True):
    '''
    Make the 2 BPT diagrams without attenuation
    
    Parameters
    ----------
    root : string
       Path to input files. 
    endf : string
       Ending of input files. 
    subvols: List of integers
        List of subvolumes to be considered
    outpath : string
        Path to output, default is output/
    metadata : dictionary
        Cosmology and other metadata information
    verbose : boolean
       If True print out messages.
    '''
    
    # Get metadata
    photmod_sfr = metadata['photmod_sfr']
    AGN = metadata['AGN']
    if AGN:
        photmod_agn = metadata['photmod_agn']
    redshift = metadata['redshift']

    # Read limits for properties and photoionisation models
    minU, maxU = get_limits(propname='logUs', photmod=photmod_sfr)
    minZ, maxZ = get_limits(propname='Z', photmod=photmod_sfr)

    # Prep plots
    fig, (axn, axs) = plt.subplots(1, 2, figsize=(30, 15))
    col = 'darkgrey'
    ytit = 'log$_{10}$([OIII]$\\lambda$5007/H$\\beta$)'
    xmins = [-1.9,-1.9]
    xmaxs = [0.8,0.9]
    ymins = [-1.5,-2.1]
    ymaxs = [1.5,1.6]
    for ii, bpt in enumerate(['NII','SII']):
        if bpt=='NII':
            xtit = 'log$_{10}$([NII]$\\lambda$6584/H$\\alpha$)'
            axn.set_xlim(xmins[ii], xmaxs[ii])
            axn.set_ylim(ymins[ii], ymaxs[ii])
            axn.set_xlabel(xtit); axn.set_ylabel(ytit)
        elif bpt=='SII':
            xtit = 'log$_{10}$([SII]$\\lambda\\lambda$6717,6731/H$\\alpha$)'
            axs.set_xlim(xmins[ii], xmaxs[ii])
            axs.set_ylim(ymins[ii], ymaxs[ii])
            axs.set_xlabel(xtit); axs.set_ylabel(ytit)

        xobs, yobs, obsdata = obs.get_obs_bpt(redshift,bpt)
        if obsdata:
            nobs = len(xobs)
            if nobs > n4contour:
                ngrid, nlev = get_ngrid_nlev(nobs)
                x,y,z = st.get_cumulative_2Ddensity(xobs,yobs,n_grid=ngrid)
                levels, colors = contour2Dsigma(n_levels=nlev,color=col)
                if bpt=='NII':
                    contour = axn.contourf(x,y,z,levels=levels,colors=colors)
                elif bpt=='SII':
                    contour = axs.contourf(x,y,z,levels=levels,colors=colors)
            else:
                if bpt=='NII':
                    axn.scatter(xobs,yobs,c=col)
                elif bpt=='SII':
                    axs.scatter(xobs,yobs,c=col)

    # Read data in each subvolume and add data to plots
    seltot = 0
    chatot=None; O3Hb_tot=None; N2Ha_tot=None; S2Ha_tot=None 
    for ivol in subvols:
        filenom = os.path.join(root+str(ivol),endf)
        f = h5py.File(filenom, 'r')
        
        # Read SF information from file
        lu_sfr = f['sfr_data/lu_sfr'][:,0]
        lz_sfr = f['sfr_data/lz_sfr'][:,0]
        Ha_sfr = st.components2tot(f['sfr_data/Halpha_sfr'],
                                   log10input=False,icomps=0)
        Hb_sfr = st.components2tot(f['sfr_data/Hbeta_sfr'],
                                   log10input=False,icomps=0)
        NII6548_sfr = st.components2tot(f['sfr_data/NII6584_sfr'],
                                        log10input=False,icomps=0)
        OII3727_sfr = st.components2tot(f['sfr_data/OII3727_sfr'],
                                        log10input=False,icomps=0)
        OIII5007_sfr = st.components2tot(f['sfr_data/OIII5007_sfr'],
                                         log10input=False,icomps=0)
        SII6731_sfr = st.components2tot(f['sfr_data/SII6731_sfr'],
                                        log10input=False,icomps=0)
        SII6717_sfr = st.components2tot(f['sfr_data/SII6717_sfr'],
                                        log10input=False,icomps=0)
        
        # Read AGN information if it exists
        if AGN:
            # Read AGN information from file
            lu_agn = f['agn_data/lu_agn'][:]
            lz_agn = f['agn_data/lz_agn'][:]
            Ha_agn = f['agn_data/Halpha_agn'][:]
            Hb_agn = f['agn_data/Hbeta_agn'][:]
            NII6548_agn = f['agn_data/NII6584_agn'][:]
            OII3727_agn = f['agn_data/OII3727_agn'][:]
            OIII5007_agn= f['agn_data/OIII5007_agn'][:]
            SII6731_agn = f['agn_data/SII6731_agn'][:]
            SII6717_agn = f['agn_data/SII6717_agn'][:]
        
        # Magnitudes for cuts
        ismagr = True
        try:
            magr = f['data/magR'][:]
        except:
            ismagr = False
        
        ismagk = True
        try:
            magk = f['data/magK'][:]
        except:
            ismagk = False
        f.close()
        
        # Combine luminosities
        if AGN:
            Ha = Ha_sfr + Ha_agn
            Hb = Hb_sfr + Hb_agn
            NII = NII6548_sfr + NII6548_agn
            OII = OII3727_sfr + OII3727_agn
            OIII = OIII5007_sfr + OIII5007_agn
            SII = SII6731_sfr + SII6731_agn +\
                SII6717_sfr + SII6717_agn
        else:
            Ha = Ha_sfr
            Hb = Hb_sfr
            NII = NII6548_sfr
            OII = OII3727_sfr
            OIII = OIII5007_sfr
            SII = SII6731_sfr + SII6717_sfr

        ind = np.where((Ha>0)   & (Hb>0)  & 
                       (NII>0)  & (OII>0) &
                       (OIII>0) & (SII>0) &
                       (lu_sfr>minU)&(lu_sfr<maxU)&
                       (lz_sfr>np.log10(minZ))&(lz_sfr<np.log10(maxZ)))
        if (np.shape(ind)[1] < 1):
            print('STOP BPT plots: not enough adequate data')
            return None

        if AGN:
            Halpha_ratio = Ha_agn[ind]/Ha[ind]
        else:
            Halpha_ratio = Ha[ind]
        
        Ha = Ha[ind]
        Hb = Hb[ind]
        NII = NII[ind]
        OII = OII[ind]
        OIII = OIII[ind]
        SII = SII[ind]
    
        O3Hb = np.log10(OIII) - np.log10(Hb)
        N2Ha = np.log10(NII) - np.log10(Ha)
        S2Ha = np.log10(SII) - np.log10(Ha)

        if ismagr:
            mag_r = magr[ind]
        if ismagk:
            mag_k = magk[ind]

        sel = (np.arange(len(O3Hb)),)
        # Add further cuts if adequate
        if redshift <= 0.2:
            flux = 2e-16 # erg/s/cm^2 Favole+2024
            Lmin = flux2L(flux,redshift) #erg/s

            if ismagr:
                sel = np.where((Ha> Lmin) & (Hb> Lmin) &
                               (OIII> Lmin) & (NII> Lmin) &
                               (SII> Lmin)&(mag_r<17.77))
            else:
                sel = np.where((Ha> Lmin) & (Hb> Lmin) &
                               (OIII> Lmin) & (NII> Lmin) &
                               (SII> Lmin))
        elif 0.7 <= redshift <= 0.9:
            flux = 1e-16  # erg/s/cm^2 Kashino+2019
            Lmin = flux2L(flux,redshift) #erg/s
            
            if ismagr:
                sel = np.where((Ha> Lmin) & (mag_r<124.1))
            else:
                sel = np.where(Ha> Lmin)
        elif 1.45 <= redshift <= 1.75:
            flux = 5e-17  # erg/s/cm^2 Kashino+2019
            Lmin = flux2L(flux,redshift) #erg/s
            
            if ismagk:
                sel = np.where((Ha> Lmin) & (mag_k<23.5))
            else:
                sel = np.where(Ha > Lmin)
            
        if (np.shape(sel)[1]<1):
            continue
        seltot = seltot + np.shape(sel)[1]

        # Model spectral line ratios
        if chatot is None:
            chatot = Halpha_ratio[sel]
            O3Hb_tot = O3Hb[sel] 
            N2Ha_tot = N2Ha[sel]
            S2Ha_tot = S2Ha[sel] 
        else:
            chatot = np.append(chatot,Halpha_ratio[sel])
            O3Hb_tot = np.append(O3Hb_tot, O3Hb[sel])
            N2Ha_tot = np.append(N2Ha_tot, N2Ha[sel])
            S2Ha_tot = np.append(S2Ha_tot, S2Ha[sel]) 

    # Information on the selection
    if verbose:
        if ismagr and ismagk:
            magmsg = '(R and K mag. used for selection)'
        elif ismagr:
            magmsg = '(R mag. used for selection)'
        elif ismagk:
            magmsg = '(K mag. used for selection)'
        else:
            magmsg = ''
        print(f'    {seltot} gal. for BPT plots at z={redshift:.1f} {magmsg}\n')

    # BPT classification lines
    for ii, bpt in enumerate(['NII','SII']):
        xline = np.arange(xmins[ii],xmaxs[ii]+0.1, 0.03)
        if bpt=='NII':
            yline = obs.lines_BPT(xline,bpt,'Kauffmann2003')
            axn.plot(xline,yline,'k--')

            yline = obs.lines_BPT(xline,bpt,'Kewley2001')
            axn.plot(xline,yline,'k-')
            
        elif bpt=='SII':
            yline = obs.lines_BPT(xline,bpt,'Kewley2001')
            axs.plot(xline,yline,'k-')

            ylinel = obs.lines_BPT(xline,bpt,'Kewley2006')
            axs.plot(xline[ylinel>yline],ylinel[ylinel>yline],'k-.')

    # Contour plots for model galaxies
    if AGN:
        agn_bins = [
            {'sel': lambda x: x < 0.3,'color':'blue',
             'label': r'$L_{\rm H\alpha,AGN}/L_{\rm H\alpha,tot}<0.3$'},
            {'sel': lambda x: (x >= 0.3) & (x <= 0.7),'color':'lime',
             'label': r'$0.3\leq L_{\rm H\alpha,AGN}/L_{\rm H\alpha,tot}\leq0.7$'},
            {'sel': lambda x: x > 0.7,'color':'red',
             'label': r'$L_{\rm H\alpha,AGN}/L_{\rm H\alpha,tot}>0.7$'}]
    else:
        agn_bins = [
            {'sel': lambda x: np.ones_like(x, dtype=bool),'color':'blue',
             'label': r'$L_{\rm H\alpha}$'+f' (z={redshift:.1f})'}]

    ntot = len(chatot)
    proxies = []; labels = []
    for ib, agn_bin in enumerate(agn_bins):
        ind = agn_bin['sel'](chatot)
        nsel = np.sum(ind)
        ngrid, nlev = get_ngrid_nlev(nsel)
        
        leg = agn_bin['label']
        col = agn_bin['color']
        per = nsel*100/ntot
        print(f'{per:.1f}% of {leg} ({nsel} out of {ntot})')
        
        if nsel == 0:
            continue
        
        yy = O3Hb_tot[ind]
        for ii, bpt in enumerate(['NII','SII']):
            if bpt=='NII':
                xx = N2Ha_tot[ind]
                if nsel > n4contour:
                    xc,yc,zc = st.get_cumulative_2Ddensity(xx,yy,n_grid=ngrid)
                    levels, colors = contour2Dsigma(n_levels=nlev,color=col)
                    axn.contour(xc,yc,zc,levels=levels,colors=colors,zorder=1)
                else:
                    axn.scatter(xx,yy,c=col, s=40,marker='o',zorder=2)
                # For legend
                proxy = mlines.Line2D([],[],color=col,alpha=0.7,
                                      marker='o',markersize=8)
                proxies.append(proxy)
                labels.append(leg+f' ({per:.1f}%)')
            elif bpt=='SII':
                xx = S2Ha_tot[ind]
                if nsel > n4contour:
                    xc,yc,zc = st.get_cumulative_2Ddensity(xx,yy,n_grid=ngrid)
                    levels, colors = contour2Dsigma(n_levels=nlev,color=col)
                    axs.contour(xc,yc,zc,levels=levels,colors=colors,zorder=1)
                else:
                    axs.scatter(xx,yy,c=col, s=40,marker='o',zorder=2)

    # Single shared legend on top
    if len(proxies) == 0:
        print('WARNING plot_bpts: no proxies for legend, skipping legend.')
    else:
        fig.legend(proxies, labels,
                   loc='lower center', bbox_to_anchor=(0.5, 1.0),
                   ncol=len(proxies), frameon=True,
                   title=f'z = {redshift:.2f}, {ntot} model tracers',
                   fancybox=True, shadow=False)
        fig.subplots_adjust(top=0.98)
    
    # Output
    bptnom = io.get_plotfile(root,endf,'bpt')
    plt.savefig(bptnom)
    if verbose:
         print(f'* BPT plots: {bptnom}')
    
    return bptnom


def plot_lf(root, endf, subvols=[0], outpath=None,
            outnom = 'masses',vol=None,
            props=['data/mh','data/lm_s','data/lm_gas'],
            prop_labels=[r'M$_{\rm h}(M_{\odot})$',
                         r'M$_{\rm *}(M_{\odot})$',
                         r'M$_{\rm gas}(M_{\odot})$'],
            xmin=9,xmax=14.,dx=0.1,
            metadata=None,verbose=True):
    '''
    Make a (luminosity) function plot
    
    Parameters
    ----------
    root : string
       Path to input files. 
    endf : string
       Ending of input files. 
    subvols: integer or list of integers
       Number of subvolumes to be considered
    outpath : string
       Path to output, default is output/
    outnom : string
       Root name for output plot
    vol : float
       Volume for normalisations (Mpc³)
    props : array of strings
       Dataset names to be plotted
    prop_labels : array of strings
       Dataset names for plot labels
    xmin : float
       Minimum value for the histogram
    xmax : float
       Maximum value for the histogram
    dx : float
       Histogram bin size
    metadata : dictionary
        Cosmology and other metadata information    
    verbose : boolean
       If True print out messages.
    '''
    # Get metadata
    vol_eff = vol
    if vol is None:
        vol_eff = metadata['vol_eff']
    redshift = metadata['redshift']    
    photmod_sfr = metadata['photmod_sfr']
    AGN = metadata['AGN']
    if AGN:
        photmod_agn = metadata['photmod_agn']
    att = metadata['att']
    if att:
        attmod = metadata['attmod']

    # Initialise histogram bins for luminosity functions
    xbins = np.arange(xmin, xmax, dx)
    xhist = xbins + dx*0.5

    # Initialise LF arrays
    nprops = len(props)
    yhist = np.zeros((nprops, len(xhist)))

    # Read data from each subvolume
    for ivol in subvols:
        # Read information from file and get histogram
        filenom = os.path.join(root+str(ivol),endf)
        f = h5py.File(filenom, 'r')
        for iprop, prop_name in enumerate(props):
            yall = io.read_and_add(f,prop_name,verbose=verbose)
            ind = np.where(yall > 0) 
            if np.shape(ind)[1] > 0:
                yy = np.log10(yall[ind])
                H, dum = np.histogram(yy,bins=np.append(xbins,xmax))
                yhist[iprop, :] += H
        f.close()

    # Normalize by bin size and volume
    yhist = yhist/dx/vol_eff

    # Plot settings
    plt.figure(figsize=(18, 18.))
    xtit = r'$\log_{10}$'+'(Property)' 
    plt.xlabel(xtit)
    ytit = r'$\log_{10}(\Phi/\mathrm{Mpc}^{-3}\,\mathrm{dex}^{-1})$'
    plt.ylabel(ytit)

    # Plot each property
    for iprop, prop_name in enumerate(props):
        leg = prop_labels[iprop]
        ilf = yhist[iprop, :]
        ind = np.where(ilf > 0)
        if len(ind[0]) > 0:
            x = xhist[ind]
            y = ilf[ind]
            indy = np.where(y > 0)
            if len(indy[0]) > 0:
                logy = np.log10(y[indy])
                plt.plot(x[indy], logy,label=leg)
    plt.legend(loc='best',frameon=False)
    
    # Output
    outf = outnom+'f'
    nom = io.get_plotfile(root,endf,outf)
    plt.savefig(nom)
    if verbose:
         print(f'* Function plot: {nom}')    
    return nom



def plot_line_lfs(root, endf, subvols=[0],
                  outpath=None,vol=None,
                  metadata=None,verbose=True):
    '''
    Make line luminosity function plots
    
    Parameters
    ----------
    root : string
       Path to input files. 
    endf : string
       Ending of input files. 
    subvols: integer or list of integers
        Number of subvolumes to be considered
    outpath : string
        Path to output, default is output/
    vol : float
       Volume for normalisations (Mpc³)
    metadata : dictionary
        Cosmology and other metadata information
    verbose : boolean
       If True print out messages.
    '''

    # Get metadata
    vol_eff = vol
    if vol is None:
        vol_eff = metadata['vol_eff']
    redshift = metadata['redshift']    
    photmod_sfr = metadata['photmod_sfr']
    AGN = metadata['AGN']
    if AGN:
        photmod_agn = metadata['photmod_agn']
    att = metadata['att']
    if att:
        attmod = metadata['attmod']

    # Read limits for properties and photoionisation models
    minU, maxU = get_limits(propname='logUs', photmod=photmod_sfr)
    minZ, maxZ = get_limits(propname='Z', photmod=photmod_sfr)

    # Input lines
    line_names = ['Halpha', 'Hbeta', 'NII6584', 'OII3727',
                  'OIII5007', 'SII6731', 'SII6717']

    # Define emission lines to plot and initialise LF arrays
    line_labels = [r'H$_{\alpha}$', r'H$_{\beta}$',
                   r'[OII]$\lambda\lambda 3727$', 
                   r'[OIII]$\lambda 5007$', r'[NII]$\lambda 6584$', 
                   r'[SII]$\lambda\lambda 6724$']

    # Initialise histogram bins for luminosity functions
    lmin = 38.0
    lmax = 46.0
    dl = 0.1
    lbins = np.arange(lmin, lmax, dl)
    lhist = lbins + dl * 0.5

    # Initialise LF arrays
    nlines = len(line_labels)
    lf = np.zeros((nlines, len(lhist)))
    lf_att = np.zeros((nlines, len(lhist)))

    # Read data from each subvolume
    for ivol in subvols:
        filenom = os.path.join(root+str(ivol),endf)
        f = h5py.File(filenom, 'r')

        # Read SF information from file
        lu_sfr = f['sfr_data/lu_sfr'][:,0]
        lz_sfr = f['sfr_data/lz_sfr'][:,0]

        # Set the dimensions of the array
        key = 'sfr_data/'+line_names[0]+'_sfr'
        if key in f:
            ngal = f[key][0].shape[0]
            print(key,ngal)

        # Read intrinsic luminosities
        sfr_data = {line: np.full(ngal, c.notnum) for line in line_names}
        for line in line_names:
            key = f'sfr_data/{line}_sfr'
            if key in f:
                ldims = f[key].ndim
                if ldims > 1:
                    sfr_data[line] = st.components2tot(f[key],
                                                       log10input=False,icomps=0)
                else:
                    sfr_data[line] = f[key][:]

        if att:
            sfr_data_att = {line: np.full(ngal, c.notnum) for line in line_names}

            for line in line_names: # Fill in available data
                key = f'sfr_data/{line}_sfr_att'
                if key in f:
                    ldims = f[key].ndim
                    if ldims > 1:
                        sfr_data_att[line] = st.components2tot(f[key],
                                                               log10input=False,icomps=0)
                    else:
                        sfr_data_att[line] = f[key][:]
        if AGN:
            # Read AGN information if it exists
            agn_data = {line: f[f'agn_data/{line}_agn'][:]
                        for line in line_names}
            if att:
                ngal = agn_data[line_names[0]].shape[0]
                agn_data_att = {line: np.full(ngal, c.notnum) for line in line_names}
                for line in line_names: # Fill in available data
                    key = f'agn_data/{line}_agn_att'
                    if key in f:
                        agn_data_att[line] = f[key][:]
        f.close()

        # Combine luminosities if adequate
        combined = {}
        combined_att = {}
        line_mapping = {
            'Ha': ['Halpha'],
            'Hb': ['Hbeta'],
            'NII': ['NII6584'],
            'OII': ['OII3727'],
            'OIII': ['OIII5007'],
            'SII': ['SII6731', 'SII6717']
        }

        for out_name, in_lines in line_mapping.items():
            arrays = [sfr_data[line] for line in in_lines]
            if AGN:
                arrays.extend([agn_data[line] for line in in_lines])
            combined[out_name] = st.safe_sum_arrays(arrays)
    
            if att:
                arrays_att = [sfr_data_att[line] for line in in_lines]
                if AGN:
                    arrays_att.extend([agn_data_att[line] for line in in_lines])
                combined_att[out_name] = st.safe_sum_arrays(arrays_att)

        # Calculate histograms for each line
        for iline, line in enumerate(line_mapping.keys()):
            # Intrinsic luminosity function
            lums = combined[line]
            ind = np.where(lums > 0) ###here more cuts like in bpt?
            if np.shape(ind)[1] > 0:
                ll = np.log10(lums[ind])
                H, dum = np.histogram(ll,bins=np.append(lbins,lmax))
                lf[iline, :] += H

            if att:
                # Dust attenuated luminosity function
                lums = combined_att[line]
                ind = np.where(lums > 0) ###here more cuts like in bpt?
                if np.shape(ind)[1] > 0:
                    ll = np.log10(lums[ind])
                    H, dum = np.histogram(ll,bins=np.append(lbins,lmax))
                    lf_att[iline, :] += H

    # Normalize by bin size and volume
    lf = lf/dl/vol_eff
    if att:
        lf_att = lf_att/dl/vol_eff

    # Plot settings
    fig, axes = plt.subplots(2, 3, figsize=(30,21))
    axes = axes.flatten()
    ytit = r'$\log_{10}(\Phi/\mathrm{Mpc}^{-3}\,\mathrm{dex}^{-1})$'
    xmin = 39.0
    xmax = 44.0
    ymin = -5.5
    ymax = -1.0

    # Plot each line
    for iline in range(nlines):
        ax = axes[iline]
        xtit = r'$\log_{10}$(L' + line_labels[iline] +\
            r'$/\mathrm{erg\,s^{-1}})$' 
        # Plot intrinsic LF (dotted line)
        ilf = lf[iline, :]
        ind = np.where(ilf > 0)
        if len(ind[0]) > 0:
            x = lhist[ind]
            y = ilf[ind]
            indy = np.where(y > 0)
            if len(indy[0]) > 0:
                logy = np.log10(y[indy])
                ax.plot(x[indy], logy, 'b:',
                        label=f'Intrinsic (z={redshift:.1f})')

        if att:
            # Plot dust-attenuated LF (solid line)
            ilf = lf_att[iline, :]
            ind = np.where(ilf > 0)
            if len(ind[0]) > 0:
                x = lhist[ind]
                y = ilf[ind]
                indy = np.where(y > 0)
                if len(indy[0]) > 0:
                    logy = np.log10(y[indy])
                    ax.plot(x[indy], logy, 'r-',
                            label='Dust-attenuated')
            
        # Set axis properties
        ax.set_xlim(left=xmin)
        #ax.set_ylim(bottom=ymin)
        ax.minorticks_on()
        ax.set_xlabel(xtit); ax.set_ylabel(ytit)
        if (iline==0) and len(ind[0]) > 0:
            ax.legend(loc='best',frameon=False)
    
    plt.tight_layout()
    
    # Output
    nom = io.get_plotfile(root,endf,'lf')
    plt.savefig(nom)
    if verbose:
         print(f'* LFs plots: {nom}')
    
    return nom


def plot_ncumu_flux(root, endf, subvols=[0],
                    outpath=None,vol=None,
                    metadata=None,verbose=True):
    '''
    Make plots with the cumulative numbers as a function of flux
    
    Parameters
    ----------
    root : string
       Path to input files. 
    endf : string
       Ending of input files. 
    subvols: integer or list of integers
        Number of subvolumes to be considered
    outpath : string
        Path to output, default is output/
    vol : float
       Volume for normalisations (Mpc³)
    metadata : dictionary
        Cosmology and other metadata information
    verbose : boolean
       If True print out messages.
    '''
    # Get metadata
    vol_eff = vol
    if vol is None:
        vol_eff = metadata['vol_eff']    
    redshift = metadata['redshift']
    photmod_sfr = metadata['photmod_sfr']
    AGN = metadata['AGN']
    if AGN:
        photmod_agn = metadata['photmod_agn']
    att = metadata['att']
    if att:
        attmod = metadata['attmod']

    # Read limits for properties and photoionisation models
    minU, maxU = get_limits(propname='logUs', photmod=photmod_sfr)
    minZ, maxZ = get_limits(propname='Z', photmod=photmod_sfr)

    # Define emission lines to plot and initialise arrays
    line_labels = [r'H$_{\alpha}$',r'H$_{\alpha}$+N[II]','O[III]',r'O[III]+H$_{\beta}$']

    # Initialise histogram bins for luminosity functions
    fmin = -18
    fmax = -14
    df = 0.2
    fbins = np.arange(fmin, fmax, df)

    # Initialise arrays
    nlines = len(line_labels)
    ncum = np.zeros((nlines, len(fbins)))
    ncum_att = np.zeros((nlines, len(fbins)))

    # Read data from each subvolume
    for ivol in subvols:
        filenom = os.path.join(root+str(ivol),endf)
        f = h5py.File(filenom, 'r')

        # Read SF information from file
        lu_sfr = f['sfr_data/lu_sfr'][:,0]
        lz_sfr = f['sfr_data/lz_sfr'][:,0]
        Ha_sfr = st.components2tot(f['sfr_data/Halpha_sfr_flux'],
                                   log10input=False,icomps=0)
        Hb_sfr = st.components2tot(f['sfr_data/Hbeta_sfr_flux'],
                                   log10input=False,icomps=0)
        NII_sfr = st.components2tot(f['sfr_data/NII6584_sfr_flux'],
                                    log10input=False,icomps=0)
        OIII_sfr = st.components2tot(f['sfr_data/OIII5007_sfr_flux'],
                                     log10input=False,icomps=0)

        if att:
            Ha_sfr_att = st.components2tot(f['sfr_data/Halpha_sfr_att_flux'],
                                           log10input=False,icomps=0)
            Hb_sfr_att = st.components2tot(f['sfr_data/Hbeta_sfr_att_flux'],
                                           log10input=False,icomps=0)
            NII_sfr_att = st.components2tot(f['sfr_data/NII6584_sfr_att_flux'],
                                            log10input=False,icomps=0)
            OIII_sfr_att = st.components2tot(f['sfr_data/OIII5007_sfr_att_flux'],
                                             log10input=False,icomps=0)

        if AGN:
            # Read AGN information if it exists
            Ha_agn = f['agn_data/Halpha_agn_flux'][:]
            Hb_agn = f['agn_data/Hbeta_agn_flux'][:]
            OIII_agn = f['agn_data/OIII5007_agn_flux'][:]
            NII_agn = f['agn_data/NII6584_agn_flux'][:]

            if att:
                Ha_agn_att = f['agn_data/Halpha_agn_att_flux'][:]
                Hb_agn_att = f['agn_data/Hbeta_agn_att_flux'][:]
                OIII_agn_att = f['agn_data/OIII5007_agn_att_flux'][:]
                NII_agn_att = f['agn_data/NII6584_agn_att_flux'][:]
        f.close()

        if AGN:
            # Combine without attenuation using safe_sum_arrays
            Ha = st.safe_sum_arrays([Ha_sfr,Ha_agn])
            HaN2 = st.safe_sum_arrays([Ha,NII_sfr,NII_agn])
            O3 = st.safe_sum_arrays([OIII_sfr,OIII_agn])
            O3Hb = st.safe_sum_arrays([O3,Hb_sfr,Hb_agn])
            
            if att:
                Ha_att = st.safe_sum_arrays([Ha_sfr_att,Ha_agn_att])
                HaN2_att = st.safe_sum_arrays([Ha_att,NII_sfr_att,NII_agn_att])
                O3_att = st.safe_sum_arrays([OIII_sfr_att,OIII_agn_att])
                O3Hb_att = st.safe_sum_arrays([O3_att,Hb_sfr_att,Hb_agn_att])
        else:
            Ha =  Ha_sfr
            HaN2 = st.safe_sum_arrays([Ha,NII_sfr])
            O3 = OIII_sfr
            O3Hb = st.safe_sum_arrays([O3,Hb_sfr])
            
            if att:
                Ha_att = Ha_sfr_att
                HaN2_att = st.safe_sum_arrays([Ha_att, NII_sfr_att])
                O3_att = OIII_sfr_att
                O3Hb_att = st.safe_sum_arrays([O3_att, Hb_sfr_att])
                
        flux_int = [Ha, HaN2, O3, O3Hb]
        if att:
            flux_att = [Ha_att, HaN2_att, O3_att, O3Hb_att]

        # Calculate the cumulative numbers for each line
        for iline in range(nlines):
            # Intrinsic flux
            flux = flux_int[iline]
            ind = np.where(flux > 0) ###here more cuts like in bpt?
            if np.shape(ind)[1] > 0:
                ff = np.log10(flux[ind])
                H = n_gt_x(fbins,ff)
                ncum[iline,:] = ncum[iline,:] + H

            if att: # Attenuated flux
                flux = flux_att[iline]
                ind = np.where(flux > 0) ###here more cuts like in bpt?
                if np.shape(ind)[1] > 0:
                    ff = np.log10(flux[ind])
                    H = n_gt_x(fbins,ff)
                    ncum_att[iline,:] = ncum_att[iline,:] + H
                    
    # Get number per volume
    ncum = ncum/vol_eff
    if att:
        ncum_att = ncum_att/vol_eff

    # Plot settings
    nfigs = 2
    fig, axes = plt.subplots(1, 2, figsize=(30,21))
    axes = axes.flatten()
    ytit = r'$\log_{10}(n_{\rm gal}(>F_{\rm lim})/\mathrm{Mpc}^{-3})$'
    xmin = fmin
    xmax = fmax
    
    line = -2
    for ifig in range(nfigs):
        ax = axes[ifig]
        xtit = r'$\log_{10}(F_{\rm lim}/\mathrm{erg\,s^{-1}\,cm^{-2}})$'
        ax.set_xlim([xmin, xmax])
        ax.minorticks_on()
        ax.set_xlabel(xtit); ax.set_ylabel(ytit)
                    
        line += 2
        for iline in [line,line+1]:
            color = plt.cm.tab10(iline % 10)
            # Plot intrinsic n
            yy = ncum[iline, :]
            ind = np.where(yy > 0)
            if len(ind[0]) > 0:
                x = fbins[ind]
                y = np.log10(yy[ind])
                ll = line_labels[iline]+'(int.)'
                ax.plot(x, y, '-',color=color,label=ll)

            if att: # Dust-attenuated 
                yy = ncum_att[iline, :]
                ind = np.where(yy > 0)
                if len(ind[0]) > 0:
                    x = fbins[ind]
                    y = np.log10(yy[ind])
                    ll = line_labels[iline]+'(att.)'
                    ax.plot(x, y,'--',color=color,label=ll)

        if ifig == 0: #Add Pozzetti's model no3 if in z range
            xobs,yobs,obsdata = obs.get_pozzetti(metadata=metadata,
                                                 outpath=None,
                                                 verbose=verbose)
            if obsdata:
                ll = 'Model3 Pozzetti+2018 (att.)'
                ax.plot(xobs, yobs, '-',color='gray',label=ll)
                #if verbose:
                #    print('Model3 Pozzetti+2018: ',xobs,yobs)
            
        # Legend
        if len(ind[0]) > 0:
            ax.legend(loc='best',frameon=False)
    fig.suptitle(f'z = {redshift:.1f}')
    plt.tight_layout()
    
    # Output
    nom = io.get_plotfile(root,endf,'flux_ncumu')
    plt.savefig(nom)
    if verbose:
         print(f'* Cumulative numbers vs flux plots: {nom}')
    
    return nom


def make_gridplots(xid_sfr=0.3,co_sfr=1,imf_cut_sfr=100,
                   xid_NLR=0.5,alpha_NLR=-1.7,verbose=True):
    '''
    Make plots for photoionisation tables
    
    Parameters
    ----------
    verbose : boolean
       If True print out messages.
    '''

    # Plot photoionisation grids on BPT diagrams
    grids_sfr = plot_model_bpt_grids(photmod='gutkin16',xid=xid_sfr,
                                     co=co_sfr,imf_cut=imf_cut_sfr,
                                     verbose=verbose)
    grids_agn = plot_model_bpt_grids(photmod='feltre16',
                                     xid=xid_NLR,alpha=alpha_NLR,
                                     verbose=verbose)
    
    return


def make_testplots(snap,ending,outpath=None,
                   subvols=[0],vol=None,
                   gridplots=False,verbose=True):
    '''
    Make test plots
    
    Parameters
    ----------
    snap: integer
        Simulation snapshot number
    out_ending : string
       End name of input files
    outpath : string
       Path to input files
    subvols: list of integers
       List of subvolumes to be considered
    vol : float
       Volume for normalisations (Mpc³)
    gridplots : boolean
       True for plotting input tables 
    verbose : boolean
       If True print out messages.
    '''
    root, endf = io.get_outroot(snap,ending,outpath=outpath,
                                verbose=verbose)
    plots_dir = os.path.join(os.path.dirname(root), 'plots')

    # Get metadata from line files
    ivol0 = str(subvols[0])
    filenom = os.path.join(root+ivol0,endf)
    metadata = io.get_metadata(filenom, verbose=verbose)

    # Set cosmology only once
    set_cosmology(omega0 = metadata['omega0'],
                  omegab = metadata['omegab'],
                  lambda0 = metadata['lambda0'],
                  h0 = metadata['h0'])  

    ### Photoionisation plots
    #if gridplots:
    #    make_gridplots() ###here work in progress

    #### Characterisation of global properties
    #if (metadata['AGN']):
    #    lbol_lf = plot_lf(root,endf,subvols=subvols,outpath=outpath,
    #                      metadata=metadata,verbose=verbose)
        
    # Characterisation of properties of ionising regions
    unh = plot_unh(root,endf,subvols=subvols,outpath=outpath,
                   metadata=metadata,verbose=verbose) 
        
    # Line plots
    # Make NII and SII bpt plots
    bpt = plot_bpts(root,endf,subvols=subvols,outpath=outpath,
                    metadata=metadata,verbose=verbose)
    
    # Make line LFs
    lfs = plot_line_lfs(root,endf,subvols=subvols,
                        outpath=outpath,vol=vol,
                        metadata=metadata,verbose=verbose)
    
    # Cumulative numbers with flux limits (if possible)
    if (metadata['flux'] and metadata['redshift']>0):
        ncumu_flux = plot_ncumu_flux(root,endf,subvols=subvols,
                                     outpath=outpath,vol=vol,
                                     metadata=metadata,
                                     verbose=verbose)
    else:
        if verbose:
            if (metadata['flux']):
                print(f'WARNING: Skipping cumulative flux plot at z=0.')
            else:
                print(f'WARNING: No flux data found in {filenom}.')

    print(f'SUCCESS: plots in {plots_dir}')
    return
