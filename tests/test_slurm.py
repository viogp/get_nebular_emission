# python -m unittest tests/test_slurm.py 

import unittest
import os
import tempfile
import shutil

import gne.gne_const as c
import gne.gne_slurm as su


class TestSlurmUtilsHdf5(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before running tests in the class"""
        # Create a temporary directory in the system temp location
        cls.test_dir = tempfile.mkdtemp(prefix='gne_test_')
        
        # Create a mock SLURM template file for testing
        cls.template_content = '''#!/bin/sh
#SBATCH --job-name=JOB_NAME
#SBATCH --output=output/JOB_NAME.out
#SBATCH --error=output/JOB_NAME.err

srun python -c '
PARAM_CONTENT
'
'''
        # Create a mock parameter file for testing
        cls.param_content = '''import gne.gne_const as const
from gne.gne import gne

verbose = True
testing = False
get_emission_lines = True

outpath = '/home/user/Data/Galform/SU1/'
subvols = 2
root = outpath+'iz87/ivol'
endf = 'gne_input.hdf5'

cutcols = ['data/Lbol_AGN']
mincuts = [1e5]
maxcuts = [None]

for ivol in range(subvols):
    infile = root+str(ivol)+'/'+endf
    print(f"Processing {infile}")
'''
        # Get the directory where slurm templates are expected
        cls.template_dir = c.slurm_temp_dir
        
        # Ensure template directory exists
        os.makedirs(cls.template_dir, exist_ok=True)
        
        # Store original template if it exists (to restore later)
        cls.taurus_template_path = os.path.join(cls.template_dir,
                                                'slurm_taurus_template.sh')
        cls.taurus_backup = None
        if os.path.exists(cls.taurus_template_path):
            with open(cls.taurus_template_path, 'r') as f:
                cls.taurus_backup = f.read()
        
        # Create test template
        with open(cls.taurus_template_path, 'w') as f:
            f.write(cls.template_content)
        
        # Create test parameter file
        cls.param_file_path = os.path.join(cls.test_dir, 'run_gne_TestSim.py')
        with open(cls.param_file_path, 'w') as f:
            f.write(cls.param_content)
   
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests in the class are finished"""
        # Remove temporary test directory
        if os.path.exists(cls.test_dir):
            try:
                shutil.rmtree(cls.test_dir)
            except (OSError, PermissionError) as e:
                print(f"Warning: Could not remove test directory {cls.test_dir}: {e}")
        
        # Restore original template or remove test template
        if cls.taurus_backup is not None:
            with open(cls.taurus_template_path, 'w') as f:
                f.write(cls.taurus_backup)
        elif os.path.exists(cls.taurus_template_path):
            os.remove(cls.taurus_template_path)

    def test_extract_job_suffix_from_params(self):
        """Test extraction of job suffix from parameter file"""
        suffix = su.extract_job_suffix_from_params(self.param_file_path)
        # Should extract 'Lbol_AGN' with '_min' since mincuts is not None
        self.assertEqual(suffix, 'Lbol_AGN_min')

    def test_extract_job_suffix_no_cuts(self):
        """Test extraction when no cuts are defined"""
        # Create a param file without cuts
        no_cuts_file = os.path.join(self.test_dir, 'run_gne_NoCuts.py')
        with open(no_cuts_file, 'w') as f:
            f.write('''
subvols = 2
root = '/path/iz87/ivol'
''')
        suffix = su.extract_job_suffix_from_params(no_cuts_file)
        self.assertEqual(suffix, 'nocut')

    def test_generate_job_name(self):
        """Test job name generation with various configurations"""
        # Single subvolume with auto suffix
        job_name = su.generate_job_name(self.param_file_path, 87, [0])
        self.assertEqual(job_name, 'gne_TestSim_iz87_iv0_Lbol_AGN_min')
        
        # Two subvolumes with auto suffix
        job_name = su.generate_job_name(self.param_file_path, 128, [0, 1])
        self.assertEqual(job_name, 'gne_TestSim_iz128_iv0_1_Lbol_AGN_min')

        # Three subvolumes with auto suffix
        job_name = su.generate_job_name(self.param_file_path, 104, [0, 1, 2])
        self.assertEqual(job_name, 'gne_TestSim_iz104_iv0-2_Lbol_AGN_min')
        
        # With user-defined suffix
        job_name = su.generate_job_name(self.param_file_path, 87, [0], job_suffix='custom')
        self.assertEqual(job_name, 'gne_TestSim_iz87_iv0_custom')

    def test_get_slurm_template_valid(self):
        """Test reading a valid template file"""
        template = su.get_slurm_template('taurus')
        self.assertIsNotNone(template)
        self.assertIn('JOB_NAME', template)
        self.assertIn('PARAM_CONTENT', template)

    def test_get_slurm_template_invalid(self):
        """Test that invalid HPC name causes sys.exit()"""
        with self.assertRaises(SystemExit):
            su.get_slurm_template('nonexistent_hpc')

    def test_modify_param_file(self):
        """Test modification of parameter file content"""
        modified = su.modify_param_file(self.param_file_path, 128, [0, 1, 2, 3])

        # Check subvols was modified
        self.assertIn('subvols = 4', modified)
        self.assertNotIn('subvols = 2', modified)
        
        # Check root path was modified with new snapshot
        self.assertIn('iz128', modified)
        self.assertNotIn('iz87', modified)

    def test_create_slurm_script(self):
        """Test creating a SLURM script with parameter file embedding"""
        script_path, job_name = su.create_slurm_script(
            'taurus', self.param_file_path, 87, [0, 1],
            outdir=self.test_dir, verbose=True
        )
        
        # Check returned values
        self.assertEqual(job_name, 'gne_TestSim_iz87_iv0_1_Lbol_AGN_min')
        self.assertTrue(script_path.endswith(f'submit_{job_name}.sh'))
        
        # Check file was created
        self.assertTrue(os.path.exists(script_path))
        
        # Check content
        with open(script_path, 'r') as f:
            content = f.read()
        
        # Job name should be replaced
        self.assertIn('gne_TestSim_iz87_iv0_1_Lbol_AGN_min', content)
        self.assertNotIn('JOB_NAME', content)
        
        # Parameter content should be embedded
        self.assertIn('get_emission_lines = True', content)
        self.assertIn('subvols = 2', content)  # Modified for 2 subvols
        self.assertNotIn('PARAM_CONTENT', content)

    def test_create_slurm_script_with_custom_suffix(self):
        """Test creating a SLURM script with custom job suffix"""
        script_path, job_name = su.create_slurm_script(
            'taurus', self.param_file_path, 90, [0],
            outdir=self.test_dir, verbose=True,
            job_suffix='lbol45'
        )
        
        self.assertEqual(job_name, 'gne_TestSim_iz90_iv0_lbol45')

    def test_create_slurm_script_invalid_param_file(self):
        """Test creating a SLURM script with non-existent parameter file"""
        with self.assertRaises(SystemExit):
            su.create_slurm_script(
                'taurus', 'nonexistent_file.py', 87, [0],
                outdir=self.test_dir, verbose=True
            )

    def test_create_slurm_script_invalid_hpc(self):
        """Test creating a SLURM script with invalid HPC -> sys.exit()"""
        with self.assertRaises(SystemExit):
            su.create_slurm_script(
                'invalid_hpc', self.param_file_path, 87, [0],
                outdir=self.test_dir, verbose=True
            )

    def test_submit_slurm_job_no_sbatch(self):
        """Test submit_slurm_job when sbatch is not available"""
        # Check if sbatch is available
        import shutil as sh
        if sh.which('sbatch') is not None:
            self.skipTest("sbatch is available; skipping test to avoid submitting jobs")
        
        # Create a dummy script file
        dummy_script = os.path.join(self.test_dir, 'dummy_script.sh')
        with open(dummy_script, 'w') as f:
            f.write('#!/bin/sh\necho "test"')
        
        # This should handle the FileNotFoundError gracefully
        job_id = su.submit_slurm_job(dummy_script, 'test_job')
        
        # Should return None when sbatch is not found
        self.assertIsNone(job_id)

        
if __name__ == '__main__':
    unittest.main()
