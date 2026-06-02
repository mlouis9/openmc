"""This test is based on a simple 4-group slab model from 
"MCNP Calculations of Subcritical Fixed and Fission Multiplication Factors",
LA-UR-10-00141 which can be found at https://mcnp.lanl.gov/pdf_files/TechReport_2010_LANL_LA-UR-10-00141_KiedrowskiBrown.pdf 
"""
import openmc
from openmc.stats import delta_function
import numpy as np
import pytest
from openmc.examples import slab_mg
import os
import glob
import shutil

from tests.testing_harness import PyAPITestHarness


class MGXSTestHarness(PyAPITestHarness):
    def _cleanup(self):
        super()._cleanup()
        f = 'mgxs.h5'
        if os.path.exists(f):
            os.remove(f)
        # Clean up mode-specific tallies.out if needed
        if os.path.exists('tallies.out'):
            os.remove('tallies.out')

    def _get_results(self, hash_output=False):
        # 1. Capture standard PyAPITestHarness results (like statepoint values)
        outstr = super()._get_results(hash_output=hash_output)
        statepoint = glob.glob(self._sp_name)[0]
        
        form = '{0:12.6E} {1:12.6E}\n'
        cov_form = '{0:12.6E}\n'
        
        with openmc.StatePoint(statepoint) as sp:
            # Subcritical multiplication specific variables
            outstr += 'multiplication:\n'
            M = sp.multiplication
            outstr += form.format(M.n, M.s)

            outstr += 'k\n'
            k = sp.keff
            outstr += form.format(k.n, k.s)

            outstr += 'k_generation:\n'
            for kg in sp.k_generation:
                outstr += form.format(kg.n, kg.s)

            outstr += 'ks:\n'
            ks = sp.ks
            outstr += form.format(ks.n, ks.s)   

            outstr += 'ks_generation:\n'
            for ksg in sp.ks_generation:
                outstr += form.format(ksg.n, ksg.s)

            outstr += 'kq:\n'
            kq = sp.kq
            outstr += form.format(kq.n, kq.s) 

            outstr += 'kq_generation:\n'
            for kqg in sp.kq_generation:
                outstr += form.format(kqg.n, kqg.s)

            outstr += 'keff_fixed_src:\n'
            keff = sp.keff_fixed_src
            outstr += form.format(keff.n, keff.s)
            
            outstr += 'keff_fixed_src_generation:\n'
            for keffg in sp.keff_fixed_src_generation:
                outstr += form.format(keffg.n, keffg.s)

            # 2. Extract new metrics if tally covariances were requested/calculated
            try:
                import warnings
                
                outstr += 'tally_covariance_and_scaling:\n'
                
                # Check for individual estimator covariances
                for estimator in ['combined', 'collision', 'absorption', 'tracklength']:
                    cov = sp.get_covariance_with_k(tally_id=1, k_estimator=estimator)
                    
                    # FIX: Use .flat to ensure we are always formatting scalar floats, not sub-arrays
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        for val in np.atleast_1d(cov).flat:
                            outstr += f"cov_{estimator}: " + cov_form.format(val)
                
                # Check for scaled tallies
                for estimator in ['combined', 'collision', 'tracklength']:
                    scaled = sp.get_scaled_tally(tally_id=1, k_estimator=estimator)

                    for val in scaled.flat:
                        outstr += f"scaled_{estimator}: {val:12.6E}\n"
                            
            except (AttributeError, KeyError, ValueError):
                # Gracefully skip if covariance options weren't enabled or active in the run
                outstr += 'covariance data not available or skipped\n'

        # 3. Append the contents of tallies.out to outstr so it is audited by results_true.dat
        if os.path.exists('tallies.out'):
            outstr += '\n=== tallies.out ===\n'
            with open('tallies.out', 'r') as f_tallies:
                outstr += f_tallies.read()
                
        return outstr   


@pytest.fixture()
def slab_model():
    openmc.reset_auto_ids()
    model = slab_mg(mgxslib_name='mgxs.h5')
    right_boundary = model.geometry.get_all_surfaces()[2]
    right_boundary.coefficients['x0'] = 10.0
    
    cell = model.geometry.get_all_cells()[1]
    mat = model.geometry.get_all_materials()[1]
    mat.set_density('macro', 0.01) 

    # Instantiate energy group and multigroup cross-section data
    ebins = np.geomspace(1e-5, 20.0e6, 5)
    groups = openmc.mgxs.EnergyGroups(group_edges=ebins)
    nusigma_f = np.array([9.6,5.4,5.2,2.5])
    sigma_s = np.array([[0.5,0.5,0.5,0.5],
                        [0.0,1.0,0.5,0.5],
                        [0.0,0.0,1.5,0.5],
                        [0.0,0.0,0.0,2.0]])
    sigma_t = np.array([5.0,5.0,5.0,5.0])
    sigma_a = sigma_t - sigma_s.sum(axis=1)
    chi = np.array([0.0,0.2,0.8,0.0])
    mat_data = openmc.XSdata('mat_1', groups)
    mat_data.order = 0
    mat_data.set_total(sigma_t)
    mat_data.set_nu_fission(nusigma_f)
    mat_data.set_chi(chi)
    mat_data.set_absorption(sigma_a)
    mat_data.set_scatter_matrix(sigma_s[...,np.newaxis])

    mg_cross_sections_file = openmc.MGXSLibrary(groups)
    mg_cross_sections_file.add_xsdata(mat_data)
    mg_cross_sections_file.export_to_hdf5()

    tally = openmc.Tally(name='absorption_tally')
    tally.scores = ['absorption']
    tallies = openmc.Tallies([tally])
    model.tallies = tallies

    # Default settings
    model.settings.particles = 10000
    model.settings.inactive = 10
    model.settings.batches = 30

    space = openmc.stats.Box([0,-1000,-1000],[10,1000,1000])
    model.settings.source = openmc.IndependentSource(
        space=space, energy=delta_function(10e6))
        
    return model


# Parametrize both modes with distinct options
@pytest.mark.parametrize("run_mode, prefix, cov_with_k", [
    ("subcritical multiplication", "multiplication", True),
    ("fixed source", "fixed_source", False)
])
def test_slab_modes(slab_model, run_mode, prefix, cov_with_k):
    # Dynamically update the model settings for this iteration
    slab_model.settings.run_mode = run_mode
    
    # Configure covariance settings based on parameterized parameters
    if cov_with_k:
        slab_model.settings.tally_covariance_with_k = cov_with_k


    if run_mode == "fixed source":
        slab_model.settings.calculate_subcritical_k = True  

    slab_model.settings.print_all_k_factors = True

    # Use the appropriate batch statepoint expected
    harness = MGXSTestHarness("statepoint.30.h5", model=slab_model)
    
    # Define our unique mode-specific gold filenames
    specific_inputs = f"inputs_true_{prefix}.dat"
    specific_results = f"results_true_{prefix}.dat"
    specific_tallies = f"tallies_true_{prefix}.out"
    
    # PRE-TEST: Stage the mode-specific files
    if os.path.exists(specific_inputs):
        shutil.copy(specific_inputs, 'inputs_true.dat')
    elif os.path.exists('inputs_true.dat'):
        os.remove('inputs_true.dat')
        
    if os.path.exists(specific_results):
        shutil.copy(specific_results, 'results_true.dat')
    elif os.path.exists('results_true.dat'):
        os.remove('results_true.dat')

    try:
        # RUN: Execute the standard OpenMC harness loop
        harness.main()
        
    finally:
        # POST-TEST: Clean up and support the `--update` workflow
        if os.path.exists('inputs_true.dat'):
            shutil.move('inputs_true.dat', specific_inputs)
            
        if os.path.exists('results_true.dat'):
            shutil.move('results_true.dat', specific_results)

        # Move the produced tallies.out into a mode-specific gold file backup if updating
        if os.path.exists('tallies.out'):
            shutil.copy('tallies.out', specific_tallies)