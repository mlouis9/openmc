import openmc
from openmc.stats import delta_function
import numpy as np
import pytest
from openmc.examples import slab_mg
import os
import glob
import shutil # Added for handling multiple gold files
import openmc.cmfd as cmfd

from tests.testing_harness import PyAPITestHarness

RADIUS = 6.5
N = 5

class SubcriticalTestHarness(PyAPITestHarness):
    def __init__(self, sp_name, model=None, use_cmfd=False, cmfd_feedback=False):
        super().__init__(sp_name, model=model)
        self.use_cmfd = use_cmfd
        self.cmfd_feedback = cmfd_feedback
    
    def _cleanup(self):
        super()._cleanup()
        f = 'mgxs.h5'
        if os.path.exists(f):
            os.remove(f)

    def _run_openmc(self):
        if self.use_cmfd:
            # Construct CMFD mesh to cover the slab model
            cmfd_mesh = cmfd.CMFDMesh()
            cmfd_mesh.lower_left = [-RADIUS, -RADIUS, -RADIUS]
            cmfd_mesh.upper_right = [RADIUS, RADIUS, RADIUS]
            cmfd_mesh.dimension = (N, N, N)
            # cmfd_mesh.energy = [0.0, 5.0e5, 20.0e6]

            # Setup and execute CMFD run
            cmfd_run = cmfd.CMFDRun()
            cmfd_run.mesh = cmfd_mesh
            cmfd_run.tally_begin = 1
            cmfd_run.solver_begin = 1
            cmfd_run.spectral = 0.0
            cmfd_run.feedback = self.cmfd_feedback
            cmfd_run.window_type = 'rolling'
            cmfd_run.display = {'entropy': False, 'balance': False}
            
            # Execute OpenMC via the CMFD loop
            cmfd_run.run()
        else:
            # Fall back to standard execution for non-CMFD runs
            super()._run_openmc()

    def _get_results(self, hash_output=False):
        outstr = super()._get_results(hash_output=hash_output)
        statepoint = glob.glob(self._sp_name)[0]
        print(f"Trying to get results, reading statepoint: {self._sp_name}")
        
        with openmc.StatePoint(statepoint) as sp:
            outstr += 'multiplication:\n'
            form = '{0:12.6E} {1:12.6E}\n'
            M = sp.multiplication
            outstr += form.format(M.n, M.s)

            outstr += 'k\n'
            k = sp.keff
            outstr += form.format(k.n, k.s)

            outstr += 'k_generation:\n'
            k_gen = sp.k_generation
            for kg in k_gen:
                outstr += form.format(kg.n, kg.s)

            outstr += 'ks:\n'
            ks = sp.ks
            outstr += form.format(ks.n, ks.s)   

            outstr += 'ks_generation:\n'
            ks_gen = sp.ks_generation
            for ksg in ks_gen:
                outstr += form.format(ksg.n, ksg.s)

            outstr += 'kq:\n'
            kq = sp.kq
            outstr += form.format(kq.n, kq.s) 

            outstr += 'kq_generation:\n'
            kq_gen = sp.kq_generation
            for kqg in kq_gen:
                outstr += form.format(kqg.n, kqg.s)

            outstr += 'keff_fixed_src:\n'
            keff = sp.keff_fixed_src
            outstr += form.format(keff.n, keff.s)
            outstr += 'keff_fixed_src_generation:\n'
            keff_gen = sp.keff_fixed_src_generation
            for keffg in keff_gen:
                outstr += form.format(keffg.n, keffg.s)

            outstr += 'entropy:\n'
            entropy = sp.entropy
            if entropy is not None:
                for e in entropy:
                    outstr += f' {e}\n'
            else:
                outstr += ' None'

            outstr += 'tallies:\n'
            tally = sp.get_tally(name='absorption')
            for score in tally.scores:
                outstr += f' {score}: {tally.get_values(scores=[score])}'

        return outstr   
                        

@pytest.fixture()
def cube_model():
    """Parametrized model builder for the eigenvalue search."""

    # Materials
    fissile_mat = openmc.Material(name='fissile_cube_mat')
    fissile_mat.add_nuclide('U235', 1.0)
    fissile_mat.set_density('g/cm3', 19.1)
    materials = openmc.Materials([fissile_mat])

    tallies = openmc.Tallies()
    tally = openmc.Tally(name='absorption')
    tally.scores = ['absorption']
    tallies.append(tally)
    
    # Geometry
    cube = openmc.model.RectangularParallelepiped(
        -RADIUS, RADIUS, -RADIUS, RADIUS, -RADIUS, RADIUS,
        boundary_type='vacuum'
    )
    cell = openmc.Cell(fill=fissile_mat, region=-cube)
    geometry = openmc.Geometry([cell])
    
    # Settings (Keep batches relatively low for a fast search)
    settings = openmc.Settings()

    settings.particles = 200000
    settings.batches = 125
    settings.inactive = 75

    entropy_mesh = openmc.RegularMesh()
    entropy_mesh.lower_left = [-RADIUS, -RADIUS, -RADIUS]
    entropy_mesh.upper_right = [RADIUS, RADIUS, RADIUS]
    entropy_mesh.dimension = (N, N, N)
    entropy_mesh.energy = [0.0, 5.0e5, 20.0e6]
    settings.entropy_mesh = entropy_mesh
    
    # Simple point source at the center
    settings.source = openmc.IndependentSource(
        space=openmc.stats.Point((0.0, 0.0, 0.0))
    )
    
    return openmc.model.Model(geometry, materials, settings, tallies)


# Parametrize the test to run Accelerated Subcritical, Unaccelerated Subcritical, and Fixed Source
@pytest.mark.parametrize("run_mode, use_cmfd, cmfd_feedback, prefix", [
    ("subcritical multiplication", True, True, "subcritical_accel"),
    ("subcritical multiplication", False, False, "subcritical_unaccel"),
    ("fixed source", False, False, "fixed_source")
])
def test_cube_modes(cube_model, run_mode, use_cmfd, cmfd_feedback, prefix):
    # 1. Dynamically update the model settings for this iteration
    cube_model.settings.run_mode = run_mode
    if run_mode == "fixed source":
        cube_model.settings.particles = int(cube_model.settings.particles/10)
        cube_model.settings.calculate_subcritical_k = True  # Ensure we still calculate k metrics in fixed source mode
    else:
        cube_model.settings.embedded_tally_scaling = True  # Enable embedded tally scaling for subcritical multiplication mode 
    
    cube_model.settings.print_all_k_factors = True
    
    sp_filename = f"statepoint.{cube_model.settings.batches}.h5"

    harness = SubcriticalTestHarness(
        sp_filename, 
        model=cube_model, 
        use_cmfd=use_cmfd, 
        cmfd_feedback=cmfd_feedback
    )
    
    # Define our unique mode-specific gold filenames
    specific_inputs = f"inputs_true_{prefix}.dat"
    specific_results = f"results_true_{prefix}.dat"
    
    # 2. PRE-TEST: Stage the mode-specific files as the default names OpenMC expects
    if os.path.exists(specific_inputs):
        shutil.copy(specific_inputs, 'inputs_true.dat')
    elif os.path.exists('inputs_true.dat'):
        os.remove('inputs_true.dat')  # Clean out leftover stale files
        
    if os.path.exists(specific_results):
        shutil.copy(specific_results, 'results_true.dat')
    elif os.path.exists('results_true.dat'):
        os.remove('results_true.dat')

    try:
        # 3. RUN: Execute the configured harness loop
        harness.main()
        
    finally:
        # 4. POST-TEST: Clean up and support the `--update` workflow
        # Move files back to their specific names (captures new runs or updates)
        if os.path.exists('inputs_true.dat'):
            shutil.move('inputs_true.dat', specific_inputs)
            
        if os.path.exists('results_true.dat'):
            shutil.move('results_true.dat', specific_results)

            