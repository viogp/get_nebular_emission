''' Generate and submit SLURM jobs for running GNE (get_nebular_emission) '''
from gne.slurm_utils import create_slurm_script, submit_slurm_job

#verbose = True
#nvol = 2
#submit_jobs = True  # False to only generate scripts
#
## Processing flags - what to run
#get_emission = True
#get_attenuation = True
#get_flux = True
#plot_tests = True
#
## Test simulations
#test_taurus_sims_GP20 = [
#    ('GP20SU_1', [87, 128], list(range(nvol))),
#]
#
## Galform in taurus
#taurus_sims_GP20 = [
#    ('GP20SU_1', [109, 104, 98, 90, 87, 128, 96, 78], list(range(nvol))),
#    ('GP20SU_2', [109, 104, 98, 90, 87], list(range(nvol))),
#    ('GP20UNIT1Gpc_fnl0', [98, 109, 87, 90, 104], [0] + list(range(3, nvol))),
#    ('GP20UNIT1Gpc_fnl0', [128,109,105,104,103,101,98,92,90,87,84,81,79,77], [1,2]),
#    ('GP20UNIT1Gpc_fnl100', [127, 108, 103, 97, 95, 89, 86, 77], [0]),
#    ('GP20UNIT1Gpc_fnl100', [108, 103, 97, 89, 86], list(range(1, nvol))),
#]
#
## Galform in cosma
#cosma_sims_GP20 = [
#    ('GP20cosma', [39, 61], list(range(64)))
#]
#
## Select which simulations to process
##simulations = taurus_sims_GP20
#simulations = test_taurus_sims_GP20
#hpc = 'taurus'
#
#job_count = 0
#for sim, snaps, subvols in simulations:
#    for snap in snaps:
#        # Generate SLURM script
#        script_path, job_name = create_slurm_script(
#            hpc, sim, snap, subvols, 
#            verbose=verbose,
#            get_emission=get_emission,
#            get_attenuation=get_attenuation,
#            get_flux=get_flux,
#            plot_tests=plot_tests
#        )
#        if verbose: 
#            print(f'  Created script: {script_path}')
#            
#        # Submit the job
#        if submit_jobs:
#            submit_slurm_job(script_path, job_name)
#            job_count += 1
#
#if submit_jobs and verbose:
#    print(f'Total jobs submitted: {job_count}')
