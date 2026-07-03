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
#SBATCH --error=__GNE_LOG_DIR__/%x_ivol%a_%A.err 
#SBATCH --output=__GNE_LOG_DIR__/%x_ivol%a_%A.out
#SBATCH --array=__GNE_VOLS__%30 
#
export SLURM_ID=$SLURM_ARRAY_JOB_ID 
export GNE_SUBVOL_INDEX=$SLURM_ARRAY_TASK_ID
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
'''
        # Create a REAL template file in the temp directory
        template_file = os.path.join(cls.test_dir, 'slurm_taurus_template.sh')
        with open(template_file, 'w') as f:
            f.write(cls.template_content)

        # Redirect slurm_temp_dir to temp directory
        cls.dir_patcher = patch.object(c, 'slurm_temp_dir', cls.test_dir)
        cls.dir_patcher.start()

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
        cls.dir_patcher.stop()

    def test_generate_job_name(self):
        job_name = su.generate_job_name(model='sam',snap=87)
        self.assertEqual(job_name, 'gne_sam_iz87')

        job_name = su.generate_job_name(model='sam',snap=87,
                                        job_suffix='custom')
        self.assertEqual(job_name, 'gne_sam_iz87_custom')

    def test_get_slurm_template(self):
        template = su.get_slurm_template('taurus')
        self.assertIsNotNone(template)
        self.assertIn('__GNE_JOB_NAME__', template)
        self.assertIn('__GNE_PARAM_CONTENT__', template)
        self.assertIn('__GNE_VOLS__', template)

        with self.assertRaises(SystemExit):
            su.get_slurm_template('nonexistent_hpc')

    def test_modify_param_file(self):
        """Test modification of parameter file content"""
        modified = su.modify_param_file(self.param_file_path,'TestSim','128')

        self.assertIn('TestSim', modified)
        self.assertIn('iz128', modified)
        self.assertNotIn('iz87', modified)


    def test_create_slurm_script(self):
        verbose = False
        script_path = su.create_slurm_script('taurus',
                                             self.param_file_path,
                                             'TestSim','sam','96','5,46',
                                             logdir=self.test_dir, verbose=verbose)
        # Check returned values
        self.assertIn('gne_sam_iz96',script_path)
        
        # Check file was created
        self.assertTrue(os.path.exists(script_path))
        
        # Variables should have been replaced
        with open(script_path, 'r') as f:
            content = f.read()
        self.assertNotIn('__GNE_LOG_DIR__', content)
        self.assertNotIn('__GNE_JOB_NAME__', content)
        self.assertNotIn('__GNE_PARAM_CONTENT__', content)
        self.assertNotIn('__GNE_VOLS__', content)

            
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
        job_id = su.submit_slurm_job(dummy_script)
        
        # Should return None when sbatch is not found
        self.assertIsNone(job_id)

        
if __name__ == '__main__':
    unittest.main()
