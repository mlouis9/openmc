"""Random ray inside the continuous-energy (CE) process (gate G2.5).

Multilevel RMC runs OpenMC's random ray solver in the CE process, as the
multigroup sweep, with the multigroup tables of the field file instead of an
MGXS library (data::random_ray_mg, filled by rmc::load_mg_tables()). These
tests run the graphite pincell's random ray solve twice, once in a
multigroup process and once in a CE process from the first run's field
file, and compare the two field files the runs export.

- The tables the CE process uses equal the multigroup run's exactly.
- On one thread, k, every region flux, and every emission are bitwise
  equal: nothing in the CE process changes the solve.
- On two threads they agree to round-off (OpenMP reductions sum in a
  different order from run to run).
- After random ray, a CE Monte Carlo run in the same process works: the
  solver type and source-writing settings are restored.
"""
import h5py
import numpy as np
import pytest

import openmc
import openmc.lib

import field as fld
import mg
import pipeline

SMALL_RR = dict(rays=200, batches=60, inactive=30)
TABLES = ('sigma_t', 'sigma_s', 'nu_sigma_f', 'chi', 'sigma_f',
          'kappa_fission', 'in_model', 'material_id')


@pytest.fixture(scope='module')
def mg_run(pincell, field_files, tmp_path_factory):
    """Single-threaded multigroup run (reusing the session's MGXS library)."""
    model, mesh = pincell
    work = tmp_path_factory.mktemp('rr_mg')
    f_mg, _, _, mg_model = mg.run_random_ray(
        model, mesh, work, mgxs_path=field_files['mgxs'], threads=1, **SMALL_RR)
    return f_mg, mg_model


def _keff_after_mc():
    """A short CE Monte Carlo run in the same session."""
    openmc.lib.settings.inactive = 2
    openmc.lib.settings.set_batches(6)
    openmc.lib.settings.particles = 500
    openmc.lib.run(output=False)
    return openmc.lib.keff()


@pytest.fixture(scope='module')
def ce_runs(pincell, mg_run, tmp_path_factory):
    model, _ = pincell
    f_mg, mg_model = mg_run
    one = pipeline.run_random_ray_in_ce(
        model, mg_model, f_mg, tmp_path_factory.mktemp('rr_ce1'), threads=1)
    two, k_mc = pipeline.run_random_ray_in_ce(
        model, mg_model, f_mg, tmp_path_factory.mktemp('rr_ce2'), threads=2,
        after=_keff_after_mc)
    return dict(one=one, two=two, k_mc=k_mc)


def _tables(path):
    with h5py.File(path, 'r') as f:
        return {k: f['materials'][k][()] for k in TABLES}


@pytest.mark.slow
def test_ce_tables_equal_mg_tables(mg_run, ce_runs):
    """The multigroup tables random ray used in the CE process equal those of
    the multigroup run, entry by entry (they pass through the field file and
    rmc::load_mg_tables())."""
    a, b = _tables(mg_run[0]), _tables(ce_runs['one'])
    for k in TABLES:
        np.testing.assert_array_equal(a[k], b[k], err_msg=k)


@pytest.mark.slow
def test_ce_process_reproduces_mg_bitwise(mg_run, ce_runs):
    A = fld.FieldFile.load(mg_run[0])
    B = fld.FieldFile.load(ce_runs['one'])
    assert A.k_mg == B.k_mg
    assert A.keys() == B.keys()
    for name in ('phi', 'Q_s', 'Q_f', 'volume_rr', 'density_mult', 'edges'):
        np.testing.assert_array_equal(getattr(A, name), getattr(B, name),
                                      err_msg=name)


@pytest.mark.slow
def test_ce_process_two_threads_round_off(mg_run, ce_runs):
    A = fld.FieldFile.load(mg_run[0])
    B = fld.FieldFile.load(ce_runs['two'])
    assert B.k_mg == pytest.approx(A.k_mg, rel=1e-12)
    assert A.keys() == B.keys()
    for name in ('phi', 'Q_s', 'Q_f'):
        a, b = getattr(A, name), getattr(B, name)
        np.testing.assert_allclose(b, a, rtol=1e-10, atol=1e-12 * np.max(a),
                                   err_msg=name)


@pytest.mark.slow
def test_monte_carlo_runs_after_random_ray(ce_runs):
    """The CE process can still run Monte Carlo after random ray, which later
    stages need. A broken solver-type restore would make this fail or give a
    nonsense k."""
    k, std = ce_runs['k_mc']
    assert np.isfinite(k) and 0.2 < k < 2.0
    assert np.isfinite(std) and std > 0
