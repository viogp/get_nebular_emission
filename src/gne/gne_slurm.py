''' Auxiliary functions for SLURM submission - HDF5 input processing '''
import os
import sys
import re
import subprocess
import gne.gne_const as c

def extract_job_suffix_from_params(param_file):
    """
    Extract job suffix from parameter file based on cutcols and limits.
    
    Parameters
    ----------
    param_file : string
        Path to the parameter file
        
    Returns
    -------
    suffix : string
        A suffix string derived from cutcols and mincuts/maxcuts
    """
    cutcols = None
    mincuts = None
    maxcuts = None
    
    with open(param_file, 'r') as f:
        content = f.read()
    
    # Extract cutcols
    match = re.search(r"cutcols\s*=\s*\[([^\]]+)\]", content)
    if match:
        # Extract the variable names, clean them up
        cutcols_str = match.group(1)
        # Get just the last part of paths like 'data/Lbol_AGN' -> 'Lbol_AGN'
        parts = re.findall(r"['\"]([^'\"]+)['\"]", cutcols_str)
        cutcols = [p.split('/')[-1] for p in parts]
    
    # Extract mincuts
    match = re.search(r"mincuts\s*=\s*\[([^\]]+)\]", content)
    if match:
        mincuts_str = match.group(1)
        # Handle None and numeric values
        mincuts = []
        for val in mincuts_str.split(','):
            val = val.strip()
            if val == 'None':
                mincuts.append(None)
            else:
                # Just note that there's a min cut
                mincuts.append('min')
    
    # Extract maxcuts
    match = re.search(r"maxcuts\s*=\s*\[([^\]]+)\]", content)
    if match:
        maxcuts_str = match.group(1)
        maxcuts = []
        for val in maxcuts_str.split(','):
            val = val.strip()
            if val == 'None':
                maxcuts.append(None)
            else:
                maxcuts.append('max')
    
    # Build suffix from cutcols and cuts
    suffix_parts = []
    if cutcols:
        for i, col in enumerate(cutcols):
            part = col
            if mincuts and i < len(mincuts) and mincuts[i] is not None:
                part += '_min'
            if maxcuts and i < len(maxcuts) and maxcuts[i] is not None:
                part += '_max'
            suffix_parts.append(part)
    
    if suffix_parts:
        return '_'.join(suffix_parts)
    else:
        return 'nocut'


def generate_job_name(param_file, simpath, snap, subvols,
                      job_suffix=None):
    """
    Generate a unique job name based on parameter file, snap, and subvols.
    
    Parameters
    ----------
    param_file : string
        Path to file
    simpath : string
        Path to the catalogues
    snap : int
        Snapshot number
    subvols : list of integers
        List of subvolumes
    job_suffix : string or None
        User-defined suffix. If None, derived from cutcols/limits in param_file
        
    Returns
    -------
    job_name : string
        Unique job name
    """
    # Extract base name from simpath
    base_name = os.path.basename(simpath)
    
    # Create subvols representation
    subvols_str = str(subvols)
    
    # Get suffix from cutcols if not provided
    if job_suffix is None:
        job_suffix = extract_job_suffix_from_params(param_file)
    
    return f'gne_{base_name}_iz{snap}_iv{subvols_str}_{job_suffix}'


def modify_param_file(param_file, simpath, snap, subvols):
    """
    Read parameter file and modify the subvols and root lines.
    
    Parameters
    ----------
    param_file : string
        Path to the parameter file
    simpath : string
        Path to model catalogues
    snap : int
        Snapshot number to set in root path
    subvols : int
        Number of subvolumes
        
    Returns
    -------
    modified_content : string
        Modified content of the parameter file
    """
    with open(param_file, 'r') as f:
        content = f.read()

    # Modify outpat line: 
    content = re.sub(
        r"^(outpath\s*=\s*).*$",
        rf"\g<1>'{simpath}'",
        content,
        flags=re.MULTILINE
    )
        
    # Modify subvols line:
    content = re.sub(
        r'^(\s*subvols\s*=\s*)\d+',
        rf'\g<1>{subvols}',
        content,
        flags=re.MULTILINE
    )

    # Modify root line: replace iz<number> with iz<snap>
    # This handles patterns like: root = ...'iz87/ivol'
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if re.match(r'\s*root\s*=', line):
            lines[i] = re.sub(r'(iz)\d+', rf'\g<1>{snap}', line)
    content = '\n'.join(lines)
    return content


def get_slurm_template(hpc):
    """Read the SLURM template file for the specified HPC."""
    fnom = f'slurm_{hpc}_template.sh'
    template_file = os.path.join(c.slurm_temp_dir, fnom)

    # Check if template file exists
    if not os.path.exists(template_file):
        print(f'ERROR: Template file {template_file} not found')
        sys.exit()

    # Read template content
    with open(template_file, 'r') as f:
        slurm_template = f.read()
    return slurm_template


def create_slurm_script(hpc, param_file, simpath, snap, subvols,
                        logdir=None, job_suffix=None, verbose=True):
    """
    Create a SLURM script that runs the modified parameter file.

    Parameters
    ----------
    hpc : string
        HPC machine to submit jobs
    param_file : string
        Path to the parameter file (e.g., 'run_gne_SU1.py')
    snap : int
        Simulation snapshot number 
    subvols : list of integers
        List of subvolumes
    outdir : string
        Name of output directory, if different from output/
    verbose : bool
        Verbose output flag
    job_suffix : string or None
        User-defined suffix for job name. If None, derived from cutcols/limits
    
    Returns
    -------
    script_path : string
        Path to the generated SLURM script
    job_name : string
        Name of the job
    """
    # Check parameter file exists
    if not os.path.exists(param_file):
        print(f'ERROR: Parameter file {param_file} not found')
        sys.exit()
    
    job_name = generate_job_name(param_file, simpath, snap,
                                 subvols, job_suffix)

    # Read the SLURM template
    slurm_template = get_slurm_template(hpc)

    # Get modified parameter file content
    modified_params = modify_param_file(param_file, simpath, snap, subvols)

    # Replace placeholders in template
    script_content = slurm_template
    script_content = script_content.replace('__GNE_LOG_DIR__', logdir)
    script_content = script_content.replace('__GNE_JOB_NAME__', job_name)
    script_content = script_content.replace('__GNE_PARAM_CONTENT__',
                                            modified_params)

    # Write script to file
    if logdir is None:
        output_dir = 'logs'
    else:
        output_dir = logdir
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    script_path = os.path.join(output_dir, f'submit_{job_name}.sh')

    with open(script_path, 'w') as f:
        f.write(script_content)
    
    return script_path, job_name


def submit_slurm_job(script_path, job_name):
    """Submit a SLURM job and return the job ID."""
    try:
        process = subprocess.Popen(
            ['sbatch', script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate()
        
        if process.returncode == 0:
            # Extract job ID from output (format: "Submitted batch job XXXXX")
            output = stdout.decode('utf-8').strip()
            job_id = output.split()[-1] if output else 'unknown'
            print(f'  Submitted {job_name}: Job ID {job_id}')
            return job_id
        else:
            print(f'  ERROR submitting {job_name}: {stderr.decode("utf-8")}')
            return None
    except FileNotFoundError:
        print(f'  WARNING: sbatch not found. Script saved to {script_path}')
        return None


def check_job_status(job_name, logdir=None, success_string='SUCCESS', verbose=True):
    """
    Check the status of a completed SLURM job by examining its output files.

    Parameters
    ----------
    job_name : string
        Name of the job (used to find .out and .err files)
    logdir : string
        Directory containing output files, default is 'output/'
    success_string : string
        String to search for in .out file to confirm success, default is 'SUCCESS'
    verbose : bool
        If True, print detailed status messages

    Returns
    -------
    status : string
        'success' - job completed successfully (.err empty, .out contains success_string)
        'error' - job has errors (.err file is not empty)
        'incomplete' - job may not have finished (.out missing success_string)
        'not_found' - output files not found
    error_content : string or None
        Content of .err file if there are errors, None otherwise
    """
    if logdir is None:
        output_dir = 'logs'
    else:
        output_dir = logdir
    
    out_file = os.path.join(output_dir, f'{job_name}.out')
    err_file = os.path.join(output_dir, f'{job_name}.err')
    
    # Check if files exist
    out_exists = os.path.exists(out_file)
    err_exists = os.path.exists(err_file)
    
    if not out_exists and not err_exists:
        if verbose:
            print(f'  {job_name}: NOT FOUND - output files do not exist')
        return 'not_found', None
    
    # Check .err file (should be empty)
    has_errors = False
    error_content = None
    if err_exists:
        with open(err_file, 'r') as f:
            error_content = f.read().strip()
        if error_content:
            has_errors = True
            if verbose:
                print(f'  {job_name}: ERROR - .err file is not empty')
                print(f'    Error content: {error_content[:200]}...' if len(error_content) > 200 else f'    Error content: {error_content}')
    
    # Check .out file for success string
    has_success = False
    if out_exists:
        with open(out_file, 'r') as f:
            out_content = f.read()
        if success_string in out_content:
            has_success = True
    
    # Determine overall status
    if has_errors:
        return 'error', error_content
    elif has_success:
        if verbose:
            print(f'  {job_name}: SUCCESS')
        return 'success', None
    else:
        if verbose:
            print(f'  {job_name}: INCOMPLETE - "{success_string}" not found in .out file')
        return 'incomplete', None


def check_all_jobs(runs, root, sam, param_file, subvols,
                   logdir=None, success_string='SUCCESS',
                   job_suffix=None, verbose=True):
    """
    Check the status of all jobs for a list of simulations.

    Parameters
    ----------
    runs : list of tuples
        List of (sim, snaps) tuples, where snaps is a list of snapshot numbers
    root : string
        Root path to the simulation data
    sam : string
        SAM name (e.g., 'Galform')
    param_file : string
        Path to the parameter file
    subvols : int
        Number of subvolumes
    logdir : string
        Directory containing output files, default is 'output/'
    success_string : string
        String to search for in .out file to confirm success
    job_suffix : string or None
        User-defined suffix. If None, derived from cutcols/limits in param_file
    verbose : bool
        If True, print detailed status messages

    Returns
    -------
    results : dict
        Dictionary with keys 'success', 'error', 'incomplete', 'not_found',
        each containing a list of job names in that status
    """
    results = {
        'success': [],
        'error': [],
        'incomplete': [],
        'not_found': []
    }
    
    for sim, snaps in runs:
        simpath = os.path.join(root, sam, sim)
        for snap in snaps:
            job_name = generate_job_name(param_file, simpath, snap,
                                         subvols, job_suffix)
            status, _ = check_job_status(job_name, logdir=logdir,
                                         success_string=success_string,
                                         verbose=verbose)
            results[status].append(job_name)
    
    # Print summary
    if verbose:
        print('\n--- Summary ---')
        print(f'  Success:    {len(results["success"])}')
        print(f'  Error:      {len(results["error"])}')
        print(f'  Incomplete: {len(results["incomplete"])}')
        print(f'  Not found:  {len(results["not_found"])}')
    
    return results


def clean_job_files(job_name=None, logdir=None, only_show=True, verbose=True):
    """
    Remove .out, .err, and .sh files for a specific job or all jobs.

    Parameters
    ----------
    job_name : string or None
        Name of the job to clean. If None, clean all job files in logdir.
    logdir : string
        Directory containing output files, default is 'logs/'
    only_show : bool
        If True, only list files that would be deleted without removing them.
        Set to False to actually delete files.
    verbose : bool
        If True, print information about deleted files

    Returns
    -------
    deleted_files : list
        List of files that were deleted (or would be deleted if only_show=True)
    """
    if logdir is None:
        output_dir = 'logdir'
    else:
        output_dir = logdir
    
    if not os.path.exists(output_dir):
        if verbose:
            print(f'Directory {output_dir} does not exist')
        return []
    
    deleted_files = []
    
    if job_name is not None:
        # Clean files for a specific job
        extensions = ['.out', '.err']
        for ext in extensions:
            filepath = os.path.join(output_dir, f'{job_name}{ext}')
            if os.path.exists(filepath):
                deleted_files.append(filepath)
                if not only_show:
                    os.remove(filepath)
        
        # Also remove the submit script
        script_path = os.path.join(output_dir, f'submit_{job_name}.sh')
        if os.path.exists(script_path):
            deleted_files.append(script_path)
            if not only_show:
                os.remove(script_path)
    else:
        # Clean all .out, .err, and .sh files in the directory
        for filename in os.listdir(output_dir):
            if filename.endswith('.out') or filename.endswith('.err') or filename.endswith('.sh'):
                filepath = os.path.join(output_dir, filename)
                deleted_files.append(filepath)
                if not only_show:
                    os.remove(filepath)
    
    # Print results
    if verbose:
        action = 'Would delete' if only_show else 'Deleted'
        if deleted_files:
            print(f'{action} {len(deleted_files)} file(s):')
            for f in deleted_files:
                print(f'  {f}')
        else:
            print('No files to delete')
        
        if only_show and deleted_files:
            print('\n(Set only_show=False to delete.)')
    
    return deleted_files


def clean_all_jobs(runs, root, sam, param_file, subvols,
                   logdir=None, only_show=True, job_suffix=None, verbose=True):
    """
    Remove .out, .err, and .sh files for all jobs in a simulation list.

    Parameters
    ----------
    runs : list of tuples
        List of (sim, snaps) tuples, where snaps is a list of snapshot numbers
    root : string
        Root path to the simulation data
    sam : string
        SAM name (e.g., 'Galform')
    param_file : string
        Path to the parameter file
    subvols : int
        Number of subvolumes
    logdir : string
        Directory containing output files, default is 'output/'
    only_show : bool
        If True, only list files that would be deleted without removing them.
        Set to False to actually delete files.
    job_suffix : string or None
        User-defined suffix. If None, derived from cutcols/limits in param_file
    verbose : bool
        If True, print information about deleted files

    Returns
    -------
    deleted_files : list
        List of all files that were deleted (or would be deleted if only_show=True)
    """
    all_deleted = []
    
    for sim, snaps in runs:
        simpath = os.path.join(root, sam, sim)
        for snap in snaps:
            job_name = generate_job_name(param_file, simpath, snap,
                                         subvols, job_suffix)
            deleted = clean_job_files(job_name, logdir=logdir,
                                      only_show=only_show, verbose=False)
            all_deleted.extend(deleted)
    
    # Print summary
    if verbose:
        action = 'Would delete' if only_show else 'Deleted'
        if all_deleted:
            print(f'{action} {len(all_deleted)} file(s):')
            for f in all_deleted:
                print(f'  {f}')
        else:
            print('No files to delete')
        
        if only_show and all_deleted:
            print('\n(Set only_show=False to delete.)')
    
    return all_deleted
