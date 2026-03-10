# python -m unittest tests/test_slurm.py 

import unittest
import os
import tempfile
import shutil
import sys
from unittest.mock import patch

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
#SBATCH --job-name=__GNE_JOB_NAME__
#SBATCH --error=__GNE_LOG_DIR__/__GNE_JOB_NAME__.err
#SBATCH --output=__GNE_LOG_DIR__/__GNE_JOB_NAME__.out
#
srun python << 'EOF_PYTHON_SCRIPT'
__GNE_PARAM_CONTENT__
EOF_PYTHON_SCRIPT
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
        
        # Mock get_slurm_template
        cls.patcher = patch('gne.gne_slurm.get_slurm_template')
        cls.mock_get_slurm_template = cls.patcher.start()
        
        def side_effect(hpc):
            if hpc == 'taurus':
                return cls.template_content
            else:
                sys.exit()
                
        cls.mock_get_slurm_template.side_effect = side_effect
        
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
                print("Warning: Could not remove test directory {}: {}".format(cls.test_dir, e))
        
        # Stop patcher
        cls.patcher.stop()

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
        job_name = su.generate_job_name(self.param_file_path, simpath='TestSim', snap=87, subvols=[0])
        self.assertEqual(job_name, 'gne_TestSim_iz87_iv0_Lbol_AGN_min')
        
        # Two subvolumes with auto suffix
        job_name = su.generate_job_name(self.param_file_path, simpath='TestSim', snap=128, subvols=[0, 1])
        self.assertEqual(job_name, 'gne_TestSim_iz128_iv0-1_Lbol_AGN_min')

        # Three subvolumes with auto suffix
        job_name = su.generate_job_name(self.param_file_path, simpath='TestSim', snap=104, subvols=[0, 1, 2])
        self.assertEqual(job_name, 'gne_TestSim_iz104_iv0-2_Lbol_AGN_min')

        # Four subvolumes with auto suffix
        job_name = su.generate_job_name(self.param_file_path, simpath='TestSim', snap=104, subvols=[4, 5, 6, 7])
        self.assertEqual(job_name, 'gne_TestSim_iz104_iv4-7_Lbol_AGN_min')

        # Four subvolumes with auto suffix
        job_name = su.generate_job_name(self.param_file_path, simpath='TestSim', snap=104, subvols=[5, 4, 6, 7])
        self.assertEqual(job_name, 'gne_TestSim_iz104_iv4-7_Lbol_AGN_min')

        # Four subvolumes with auto suffix
        job_name = su.generate_job_name(self.param_file_path, simpath='TestSim', snap=104, subvols=[4, 5, 7])
        self.assertEqual(job_name, 'gne_TestSim_iz104_iv4_5_7_Lbol_AGN_min')
        
        # With user-defined suffix
        job_name = su.generate_job_name(self.param_file_path, simpath='TestSim', snap=87, subvols=[0], job_suffix='custom')
        self.assertEqual(job_name, 'gne_TestSim_iz87_iv0_custom')

    def test_get_slurm_template_valid(self):
        """Test reading a valid template file"""
        template = su.get_slurm_template('taurus')
        self.assertIsNotNone(template)
        self.assertIn('__GNE_JOB_NAME__', template)
        self.assertIn('__GNE_PARAM_CONTENT__', template)

    def test_get_slurm_template_invalid(self):
        """Test that invalid HPC name causes sys.exit()"""
        with self.assertRaises(SystemExit):
            su.get_slurm_template('nonexistent_hpc')

    def test_modify_param_file(self):
        """Test modification of parameter file content"""
        modified = su.modify_param_file(self.param_file_path, simpath='TestSim', snap=128, subvols=4)

        # Check subvols was modified
        self.assertIn('subvols = 4', modified)
        self.assertNotIn('subvols = 2', modified)
        
        # Check root path was modified with new snapshot
        self.assertIn('iz128', modified)
        self.assertNotIn('iz87', modified)

    def test_modify_param_file_list(self):
        """Test modification of parameter file content with list of subvolumes"""
        modified = su.modify_param_file(self.param_file_path, simpath='TestSim', snap=128, subvols=[42, 63])

        # Check subvols was modified
        self.assertIn('subvols = [42, 63]', modified)
        self.assertNotIn('subvols = 2', modified)
        
        # Check root path was modified with new snapshot
        self.assertIn('iz128', modified)
        self.assertNotIn('iz87', modified)


    def test_create_slurm_script(self):
        """Test creating a SLURM script with parameter file embedding"""
        script_path, job_name = su.create_slurm_script(
            'taurus', self.param_file_path, simpath='TestSim', snap=87, subvols=[0, 1],
            logdir=self.test_dir, verbose=True
        )
        
        # Check returned values
        self.assertEqual(job_name, 'gne_TestSim_iz87_iv0-1_Lbol_AGN_min')
        self.assertTrue(script_path.endswith(f'submit_{job_name}.sh'))
        
        # Check file was created
        self.assertTrue(os.path.exists(script_path))
        
        # Check content
        with open(script_path, 'r') as f:
            content = f.read()
        
        # Job name should be replaced
        self.assertIn('gne_TestSim_iz87_iv0-1_Lbol_AGN_min', content)
        self.assertNotIn('__GNE_JOB_NAME__', content)
        
        # Parameter content should be embedded
        self.assertIn('get_emission_lines = True', content)
        self.assertIn('subvols = [0, 1]', content)
        self.assertNotIn('__GNE_PARAM_CONTENT__', content)

    def test_create_slurm_script_with_custom_suffix(self):
        """Test creating a SLURM script with custom job suffix"""
        script_path, job_name = su.create_slurm_script(
            'taurus', self.param_file_path, simpath='TestSim', snap=90, subvols=[0],
            logdir=self.test_dir, verbose=True,
            job_suffix='lbol45'
        )
        
        self.assertEqual(job_name, 'gne_TestSim_iz90_iv0_lbol45')

    def test_create_slurm_script_invalid_param_file(self):
        """Test creating a SLURM script with non-existent parameter file"""
        with self.assertRaises(SystemExit):
            su.create_slurm_script(
                'taurus', 'nonexistent_file.py', simpath='TestSim', snap=87, subvols=[0],
                logdir=self.test_dir, verbose=True
            )

    def test_create_slurm_script_invalid_hpc(self):
        """Test creating a SLURM script with invalid HPC -> sys.exit()"""
        with self.assertRaises(SystemExit):
            su.create_slurm_script(
                'invalid_hpc', self.param_file_path, simpath='TestSim', snap=87, subvols=[0],
                logdir=self.test_dir, verbose=True
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
