"""
This is a modular python code to provide model spectral emission 
lines, both from star-forming regions and narrow-line regions of AGNs.
The input of the code are global galactic properties. 

The intrinsic luminosities can be passed through an attenuation model 
to also get the predicted attenuated luminosities.

@authors: viogp
"""

import src.gne_const as const
from src.gne import gne
from src.gne_plots import make_testplots
import h5py

### RUN the code with the given parameters and/or make plots
testing = False    # If True: use only the first 50 elements
run_code = True
plot_tests = True

# Calculate emission from AGNs: AGN = True
AGN = True

###############################################################
### OUTPUT FILES: Default output path is output/
outpath = None

###############################################################
### INPUT FILES: given as a root, ending and number of subvolumes
# Input files are expected to have, AT LEAST:
# Stellar mass (M*) of the galaxy (or disc, SF burst, buldge, etc).
# Star formation rate (SFR) OR magnitude of Lyman Continuum photons (m_LC).
# Mean metallicity of the cold gas (Z).
root = 'data/example_data/iz61/GP20_31p25kpc_z0_example_vol'
endf   = '.hdf5'
subvols = 2

### INPUT FORMAT ('txt' for text files; 'hdf5' for HDF5 files)
inputformat = 'hdf5'

### OUTPUT PATH (Default: output/)
outpath = None  

### UNITS: 
# units_h0=False if input units [Mass]=Msun, [Radius]=Mpc (default)
# units_h0=True  if input units [Mass]=Msun/h, [Radius]=Mpc/h
units_h0=True
# units_Gyr=False if input units [SFR,Mdot]=[Mass]/yr (default)
# units_Gyr=True  if input units [SFR,Mdot]=[Mass]/Gyr 
units_Gyr=True 
# units_L40h2=False if input units [L]=erg/s  (default)
# units_L40h2=True  if input units [L]=1e40 h^-2 erg/s
units_L40h2=False 

####################################################
############  Emission from SF regions #############
####################################################

# All available models can be seen in gne_const module.
# NEBULAR model connecting global properties to ionising properties:
# nH: number density of Hydrogen (or electrons); U: ionising parameter
model_nH_sfr='kashino20'
model_U_sfr='kashino20'
# PHOTOIONIZATION model for SF regions to get line luminosities
photmod_sfr='gutkin16'

### INPUT PARAMETERS
# m_sfr_z has the location in the input files of the three mandatory parameters:
# M*(units), SFR or m_LC and Zgas. 
# m_sfr_z is a list of lists with either the column number
# for each parameters or the name of the HDF5 variable.
m_sfr_z = [['data/mstar_disk','data/SFR_disk','data/Zgas_disk'],
           ['data/mstar_bulge','data/SFR_bulge','data/Zgas_bulge']]

# mtot2mdisk is True if the stellar mass of discs is calculated 
# from the total and buldge values (False by default)
# mtot2mdisk = True; cols = [[M,SFR,Z],[M_bulge,SFR_bulge,Z_bulge]]
# mtot2mdisk = False; cols = [[M_disk,SFR_disk,Z_disk],[M_bulge,SFR_bulge,Z_bulge]]        
mtot2mdisk = False

# LC2sfr is True when Lyman Continuum photons are given instead of the SFR
# LC2sfr = True; cols = [[M,m_LC,Z]]
# LC2sfr = False; cols = [[M,SFR,Z]] (Default option)      
LC2sfr = False

# inoh True if the gas metallicity input as log(O/H)+12
#      False if Zgas = MZcold/Mcold (Default option)
inoh = False

### INITIAL MASS FUNCTIONs
# Specify the assumed IMFs for each galaxy component in the input data.
# Example for two components: IMF = ['Kennicut','Kennicut']
IMF = ['Kennicut','Kennicut']

####################################################
#####  Emission from AGN narrow line regions #######
####################################################
# PHOTOIONIZATION model for AGN NLR to get line luminosities
photmod_agn = 'feltre16'

# Columns to read either the central or global metallicity 
Zgas_NLR = ['data/Zgas_bulge','data/Zgas_disk']
# Z_correct_grad 
#    False (default) if the central gas metallicity has been provided
#    True to correct a global metallicity with the gradients from Belfiore+2017
Z_correct_grad = True

# Connecting global properties to AGN NLR characteristics:
# Model to calculate the ionising parameter, U
model_U_agn    = 'panuzzo03'

# Panuzzo's model requires the calculation of the filling factor
# epsilon(Mgas, Scalelength, n_NLR, T_NLR, r_NLR)
# n_NLR, T_NLR and r_NLR are taken as constants.
# If mgas_r = None, a fixed volume-filling factor is assumed, otherwise
# mgas_r is a list of lists with either the column number
# for each parameters or the name of the HDF5 variable.
# Each list can correspond to a different component:
# mgas_r = [[mgas_comp1,R_comp1],...]
# If mgas_r given, specify also mgasr_type = 'disc', 'sphere' or None
mgas_r = [['data/mgas_disk','data/rhm_disk'],
          ['data/mgas_bulge','data/rhm_bulge']]
mgasr_type = ['disc','sphere']

# Type of radius input, per component:
# 0: scalelength;
# 1: effective radius, Re
# 2: half-mass/light radius, R50 (Re=r502re*R50 with a default r502re=1) 
# 3: radius of the galaxy or host halo
r_type = [1,1]

# spec: model for the spectral distribution of the AGN
model_spec_agn = 'feltre16'
    
# The AGNs bolometric luminosity, Lagn, is needed.
# This value can be either firectly input or calculated.
# The way of obtaining Lagn is indicated in Lagn_inputs.
# The calcultions require different black hole (BH) parameters.
# Lagn_inputs='Lagn' if Lagn in input
#            in erg/s,h^-2erg/s,1e40erg/s,1e40(h^-2)erg/s
#            Lagn_params=[Lagn, Mbh] 
# Lagn_inputs='Mdot_hh' for a calculation from
#            the mass accretion rate of the BH, Mdot,
#            the BH mass, Mbh,
#            and, as an optional input, the BH spin, Mspin. 
#            Lagn_params=[Mdot,Mbh] or [Mdot,Mbh,Mspin]
# Lagn_inputs='Mdot_stb_hh' for a calculation from
#            the mass accretion rate during the last starburst, Mdot_stb,
#            the hot halo or radio mass accretion, Mdot_hh,
#            the BH mass, Mbh,
#            and, as an optional input, the BH spin, Mspin. 
#            Lagn_params=[Mdot_stb,Mdot_hh,Mbh] or [Mdot_stb,Mdot_hh,Mbh,Mspin]
# Lagn_inputs='radio_mode' for a calculation from
#            the mass of the hot gas, Mhot,
#            the BH mass, Mbh,
#            and, as an optional input, the BH spin, Mspin. 
#            Lagn_params=[Mhot,Mbh] or [Mhot,Mbh,Mspin]
# Lagn_inputs='quasar_mode' for a calculation from
#            the mass of the bulge, Mbulge,
#            the half-mass radius of the bulge, rbulge,
#            the circular velocity of the bulge, vbulge,
#            the BH mass, Mbh,
#            and, as an optional input, the BH spin, Mspin. 
#            Lagn_params=[Mbulge,rbulge,vbulge,Mbh,(Mspin)]
# Lagn_inputs='complete' for a calculation from
#            the mass of the bulge, Mbulge,
#            the half-mass radius of the bulge, rbulge,
#            the circular velocity of the bulge, vbulge,
#            the mass of the hot gas, Mg,
#            the BH mass, Mbh,
#            and, as an optional input, the BH spin, Mspin. 
#            Lagn_params=[Mbulge,rbulge,vbulge,Mhot,Mbh,(Mspin)]
Lagn_inputs = 'Lagn'; Lagn_params=['data/lagn','data/mstar_bulge']

####################################################
########  Redshift evolution parameters  ###########
####################################################

### HIGH REDSHIFT CORRECTION ###
# Empirical relationships to connect global galaxy properties and nebular
    # properties are often derived from local galaxies. get_emission_lines has
    # a way of evolving the filling factor with redshift. If this correction is to be used,
    # a fixed number of files is needed equal to that at z=0.
    # If local relations are to be used: infiles_z0 = [None]
root_z0 = None

####################################################
##########       Dust attenuation      #############
####################################################

# Continuum and line attenuation calculation. If this option is selected 
    # the output file will have intrinsic AND attenuated values of
    # luminosity for the emission lines.
# att=True to calculate the dust attenuation; False, otherwise
att = False
    
# To use Cardelli's law (following Favole et. al. 2020):
    # attmod = 'cardelli89' (default)
    # att_params = [half-mass radius, cold gas mass, cold gas metallicity]
# To use already available attenuation coefficients: attmod = 'ratios'
    # att_params in this case has the location of the attenuation coefficients
    # for each line for which attenuation is going to be calculated.
    # This mode requieres an extra variable, att_ratio_lines, with the names
    # of the lines corresponding to the coefficients listed in att_params.
# Example:
    # attmod = 'ratios'
    # att_params = [31,32,33,34,35,36,36]
    # att_ratio_lines = ['Halpha','Hbeta','NII6584','OII3727','OIII5007','SII6717','SII6731'] 

attmod='cardelli89'
att_params=['data/rhm_disk','data/mgas_disk','data/Zgas_disk'] 

####################################################
##########      Other calculations     #############
####################################################

# Include other parameters in the output files
extra_params_names = ['mh','magK','magR','type','MBH']
extra_params_labels = extra_params_names
extra_params = ['data/mh','data/magK','data/magR','data/type','data/MBH']

# Make the calculation on a subsample based on selection cuts
# Paramter to impose cuts
cutcols = ['data/mh']
# List of minimum values. None for no inferior limit.
mincuts = [21*9.35e8]
# List of maximum values. None for no superior limit.
maxcuts = [None]

##################################################################
#############    Run the code and/or make plots   ################
##################################################################

verbose = True
for ivol in range(subvols):
    infile = root+str(ivol)+endf

    infile_z0 = root_z0
    if root_z0 is not None:
        infile_z0 = root_z0+str(ivol)+endf
    
    # Get the redshift, cosmology and volume of the model galaxies
    f = h5py.File(infile)
    header = f['header']
    redshift = header.attrs['redshift']
    snapshot = header.attrs['snapnum']
    vol = header.attrs['bside_Mpch']**3
    h0 = header.attrs['h0']
    omega0 = header.attrs['omega0']
    omegab = header.attrs['omegab']
    lambda0 = header.attrs['lambda0']
    mp = header.attrs['mp_Msunh']        
    
    if run_code:  # Run the code
        gne(infile,redshift,snapshot,h0,omega0,omegab,lambda0,vol,mp,
            inputformat=inputformat,outpath=outpath,
            units_h0=units_h0,units_Gyr=units_Gyr,units_L40h2=units_L40h2,
            model_nH_sfr=model_nH_sfr, model_U_sfr=model_U_sfr,
            photmod_sfr=photmod_sfr,
            m_sfr_z=m_sfr_z,mtot2mdisk=mtot2mdisk, LC2sfr=LC2sfr,
            inoh=inoh,IMF = IMF,
            AGN=AGN,photmod_agn=photmod_agn,
            Zgas_NLR=Zgas_NLR,Z_correct_grad=Z_correct_grad,
            model_U_agn=model_U_agn,           
            mgas_r_agn=mgas_r,mgasr_type_agn=mgasr_type,r_type_agn=r_type,
            model_spec_agn=model_spec_agn,
            Lagn_inputs=Lagn_inputs, Lagn_params=Lagn_params,
            infile_z0=infile_z0, 
            att=att, attmod=attmod, att_params=att_params,
            extra_params=extra_params,extra_params_names=extra_params_names,
            extra_params_labels=extra_params_labels,
            cutcols=cutcols, mincuts=mincuts, maxcuts=maxcuts,
            testing=testing,verbose=verbose)
        
if plot_tests:  # Make test plots
    make_testplots(root,snapshot,subvols=subvols,gridplots=False,
                   outpath=outpath,verbose=verbose)
