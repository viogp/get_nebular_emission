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


def generate_job_name(param_file, snap, subvols, job_suffix=None):
    """
    Generate a unique job name based on parameter file, snap, and subvols.
    
    Parameters
    ----------
    param_file : string
        Path to the parameter file (e.g., 'run_gne_SU1.py')
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
    # Extract base name from parameter file (e.g., 'run_gne_SU1.py' -> 'SU1')
    base_name = os.path.basename(param_file)
    base_name = base_name.split('_')[-1]
    base_name = re.sub(r'\.py$', '', base_name)
    
    # Create subvols representation
    if len(subvols) <= 2:
        subvols_str = '_'.join(map(str, subvols))
    else:
        subvols_str = f'{subvols[0]}-{subvols[-1]}'
    
    # Get suffix from cutcols if not provided
    if job_suffix is None:
        job_suffix = extract_job_suffix_from_params(param_file)
    
    return f'gne_{base_name}_iz{snap}_iv{subvols_str}_{job_suffix}'


def modify_param_file(param_file, snap, subvols):
    """
    Read parameter file and modify the subvols and root lines.
    
    Parameters
    ----------
    param_file : string
        Path to the parameter file
    snap : int
        Snapshot number to set in root path
    subvols : list of integers
        List of subvolumes
        
    Returns
    -------
    modified_content : string
        Modified content of the parameter file
    """
    with open(param_file, 'r') as f:
        content = f.read()

    # Modify subvols line: match 'subvols = <number>' or 'subvols=<number>'
    content = re.sub(
        r'^(\s*subvols\s*=\s*)\d+',
        rf'\g<1>{len(subvols)}',
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


def create_slurm_script(hpc, param_file, snap, subvols,
                        outdir=None, verbose=True, job_suffix=None):
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
    
    job_name = generate_job_name(param_file, snap, subvols, job_suffix)

    # Read the SLURM template
    slurm_template = get_slurm_template(hpc)

    # Get modified parameter file content
    modified_params = modify_param_file(param_file, snap, subvols)
    
    # Escape the content for embedding in the shell script
    # Replace single quotes with escaped version for shell
    escaped_params = modified_params.replace("'", "'\"'\"'")

    # Replace placeholders in template
    script_content = slurm_template
    script_content = script_content.replace('JOB_NAME', job_name)
    script_content = script_content.replace('PARAM_CONTENT', escaped_params)
    
    # Write script to file
    if outdir is None:
        output_dir = 'output'
    else:
        output_dir = outdir
    
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
