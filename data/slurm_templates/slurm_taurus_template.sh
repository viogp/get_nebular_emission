#!/bin/sh
# Template for SLURM job submission - HDF5 input processing
# Placeholders: JOB_NAME, SIM_NAME, SNAP_NUM, SUBVOLS_LIST, VERBOSE
# Placeholders: GET_EMISSION, GET_ATTENUATION, GET_FLUX, PLOT_TESTS

#SBATCH -A 16cores
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=JOB_NAME
#SBATCH --error=output/JOB_NAME.err
#SBATCH --output=output/JOB_NAME.out
##SBATCH --mem=600000
#SBATCH --partition=all
#SBATCH --time=30-00:00:00
#
export OMP_NUM_THREADS=16
srun python -c "
from src.gne import gne
from src.gne_att import gne_att
from src.gne_flux import gne_flux
from src.gne_plots import make_testplots
import h5py

# Job parameters
sim = 'SIM_NAME'
snapshot = SNAP_NUM
subvols_list = SUBVOLS_LIST
verbose = VERBOSE

# Processing flags
get_emission_lines = GET_EMISSION
get_attenuation = GET_ATTENUATION
get_flux = GET_FLUX
plot_tests = PLOT_TESTS

# Configuration (modify as needed for your setup)
root = f'data/{sim}/iz{snapshot}/ivol'
endf = '/gne_input.hdf5'
outpath = None

# SF emission model parameters
model_nH_sfr = 'kashino20'
model_U_sfr = 'kashino20'
photmod_sfr = 'gutkin16'

# Input parameters
m_sfr_z = [['data/mstars_disk','data/mstardot','data/Zgas_disc'],
           ['data/mstars_bulge','data/mstardot_burst','data/Zgas_bst']]
mtot2mdisk = False
inoh = False
IMF = ['Kennicut','Kennicut']

# AGN parameters
AGN = True
photmod_agn = 'feltre16'
Zgas_NLR = ['data/Zgas_bst','data/Zgas_disc']
Z_correct_grad = True
model_U_agn = 'panuzzo03'
model_spec_agn = 'feltre16'
Lagn_inputs = 'Lagn'
Lagn_params = ['data/Lbol_AGN','data/mstars_bulge']

# Gas and radius parameters
mgas_r = [['data/mcold','data/rdisk'],
          ['data/mcold_burst','data/rbulge']]
mgasr_type = ['disc','bulge']
r_type = [2,2]

# Attenuation parameters
attmod = 'ratios'
att_config = ['Halpha', 'Hbeta', 'NII6583', 'OII3727', 'OIII5007', 'SII6716']
line_att = False

# Extra parameters
extra_params_names = ['type','mh','xgal','ygal','zgal',
                      'vxgal','vygal','vzgal','magK','magR','M_SMBH']
extra_params = ['data/type','data/mhhalo',
                'data/xgal','data/ygal','data/zgal',
                'data/vxgal','data/vygal','data/vzgal',
                'data/mag_UKIRT-K_o_tot_ext',
                'data/mag_SDSSz0.1-r_o_tot_ext',
                'data/M_SMBH']
if attmod == 'ratios':
    for line in att_config:
        extra_params_names.append('ratio_'+line)
        extra_params.append('data/ratio_'+line)
extra_params_labels = extra_params_names

# Selection criteria
cutcols = ['data/mhhalo']
mincuts = [21*9.35e8]
maxcuts = [None]

# High-z correction
root_z0 = None

# Run processing
for ivol in subvols_list:
    infile = root + str(ivol) + endf
    
    infile_z0 = root_z0
    if root_z0 is not None:
        infile_z0 = root_z0 + str(ivol) + endf

    # Read header info
    f = h5py.File(infile)
    header = f['header']
    redshift = header.attrs['redshift']
    boxside = header.attrs['bside_Mpch']
    h0 = header.attrs['h0']
    omega0 = header.attrs['omega0']
    omegab = header.attrs['omegab']
    lambda0 = header.attrs['lambda0']
    mp = header.attrs['mp_Msunh']
    try:
        p = header.attrs['percentage']/100.
    except:
        p = 1
    f.close()
    vol = p * boxside**3

    if get_emission_lines:
        gne(infile, redshift, snapshot, h0, omega0, omegab, lambda0, vol, mp,
            inputformat='hdf5', outpath=outpath,
            units_h0=True, units_Gyr=True, units_L40h2=True,
            model_nH_sfr=model_nH_sfr, model_U_sfr=model_U_sfr,
            photmod_sfr=photmod_sfr,
            m_sfr_z=m_sfr_z, mtot2mdisk=mtot2mdisk,
            inoh=inoh, IMF=IMF,
            AGN=AGN, photmod_agn=photmod_agn,
            Zgas_NLR=Zgas_NLR, Z_correct_grad=Z_correct_grad,
            model_U_agn=model_U_agn,
            mgas_r=mgas_r, mgasr_type=mgasr_type, r_type=r_type,
            model_spec_agn=model_spec_agn,
            Lagn_inputs=Lagn_inputs, Lagn_params=Lagn_params,
            infile_z0=infile_z0,
            extra_params=extra_params,
            extra_params_names=extra_params_names,
            extra_params_labels=extra_params_labels,
            cutcols=cutcols, mincuts=mincuts, maxcuts=maxcuts,
            testing=False, verbose=verbose)

    if get_attenuation:
        gne_att(infile, outpath=outpath, attmod=attmod, line_att=line_att,
                att_config=att_config, verbose=verbose)

    if get_flux:
        gne_flux(infile, outpath=outpath, verbose=verbose,
                 line_names=['Halpha','Hbeta','NII6584','OIII5007'])

if plot_tests:
    make_testplots(root, endf, snapshot, subvols=len(subvols_list),
                   gridplots=False, outpath=outpath, verbose=verbose)
"
