"""Random ray one batch at a time in the CE process, and segment records
(gate G2.6).

Residual Monte Carlo drives random ray batch by batch
(openmc.lib.rmc.sweep_*) and records, for chosen source regions, every
active segment crossing them: direction, chord length, and the chord-averaged
multigroup angular flux psi_bar = q + delta_psi / (Sigma_t * length)
(include/openmc/random_ray/segment_records.h).

- Driven batch by batch, random ray reproduces the whole solve's k in every
  batch, bitwise on one thread.
- With the naive volume estimator, a region's scalar flux iterate is exactly
  the length-weighted mean of psi_bar over the segments crossing it. This
  ties the records to the solver's own flux update.
- The recorded lengths add up to the total active ray length.
- Only flagged regions are recorded, and recording can be turned off.
"""
import os

import numpy as np
import openmc
import pytest

import openmc.lib

import mg
import pipeline

SMALL_RR = dict(rays=150, batches=24, inactive=10)


@pytest.fixture(scope='module')
def mg_run(pincell, field_files, tmp_path_factory):
    model, mesh = pincell
    work = tmp_path_factory.mktemp('sweep_mg')
    f_mg, sp, _, mg_model = mg.run_random_ray(
        model, mesh, work, mgxs_path=field_files['mgxs'], threads=1, **SMALL_RR)
    return dict(field=f_mg, statepoint=sp, mg_model=mg_model)


def _session(model, mg_run, workdir, threads=1, **rr):
    """Export the CE model (with random ray options) and initialize it."""
    ce = pipeline.random_ray_ce_model(model, mg_run['mg_model'], mg_run['field'])
    for k, v in rr.items():
        ce.settings.random_ray[k] = v
    workdir.mkdir(parents=True, exist_ok=True)
    ce.export_to_model_xml(workdir / 'model.xml')
    os.chdir(workdir)
    openmc.lib.init(['-s', str(threads)], output=False)
    return ce


@pytest.fixture(scope='module')
def batch_k(pincell, mg_run, tmp_path_factory):
    """k after every batch, driven one batch at a time."""
    import openmc.lib.rmc as rmc
    model, _ = pincell
    cwd = os.getcwd()
    try:
        _session(model, mg_run, tmp_path_factory.mktemp('sweep_k'))
        rmc.sweep_init()
        k = []
        for _ in range(SMALL_RR['batches']):
            rmc.sweep_iterate(1)
            k.append(rmc.sweep_regions()['k'])
        rmc.sweep_finalize()
    finally:
        openmc.lib.finalize()
        os.chdir(cwd)
    return np.array(k)


@pytest.fixture(scope='module')
def records(pincell, mg_run, tmp_path_factory):
    """Records of every region in one batch (naive volumes), of two regions
    in the next, and of none in the one after."""
    import openmc.lib.rmc as rmc
    model, _ = pincell
    cwd = os.getcwd()
    out = {}
    try:
        ce = _session(model, mg_run, tmp_path_factory.mktemp('sweep_rec'),
                      volume_estimator='naive')
        out['active_length'] = (ce.settings.particles
                                * ce.settings.random_ray['distance_active'])
        rmc.sweep_init()
        rmc.sweep_iterate(5)                   # discover the regions
        rmc.sweep_set_recording(all_regions=True)
        rmc.sweep_iterate(1)
        out['all'] = rmc.sweep_records()
        out['regions'] = rmc.sweep_regions()
        rmc.sweep_set_recording([0, 7])
        rmc.sweep_iterate(1)
        out['two'] = rmc.sweep_records()
        rmc.sweep_set_recording([])
        rmc.sweep_iterate(1)
        out['none'] = rmc.sweep_records()
        rmc.sweep_finalize()
    finally:
        openmc.lib.finalize()
        os.chdir(cwd)
    return out


@pytest.mark.slow
def test_batch_by_batch_matches_whole_solve(mg_run, batch_k):
    """Driving random ray one batch at a time changes nothing: k in every
    batch equals the multigroup run's, bitwise (both on one thread)."""
    with openmc.StatePoint(mg_run['statepoint']) as sp:
        k_mg = np.asarray(sp.k_generation)
    np.testing.assert_array_equal(batch_k, k_mg)


@pytest.mark.slow
def test_flux_is_length_weighted_mean_of_records(records):
    """phi_R,g = sum_j l_j psi_bar_j,g / sum_j l_j for every region hit, with
    the naive volume estimator (whose volume is the batch's track length)."""
    rec, reg = records['all'], records['regions']
    R, G = reg['phi'].shape
    L = np.bincount(rec['region'], weights=rec['length'], minlength=R)
    num = np.zeros((R, G))
    np.add.at(num, rec['region'], rec['length'][:, None] * rec['psi_bar'])
    hit = L > 0
    assert hit.sum() == R
    np.testing.assert_allclose(num[hit] / L[hit, None], reg['phi'][hit], rtol=1e-6)


@pytest.mark.slow
def test_recorded_length_is_total_active_length(records):
    """Every active segment is in some region, so the recorded lengths add up
    to rays x active distance (less the TINY_BIT trimmed at mesh crossings)."""
    total = records['all']['length'].sum()
    assert total == pytest.approx(records['active_length'], rel=1e-6)


@pytest.mark.slow
def test_records_are_unit_directions_and_positive(records):
    rec = records['all']
    np.testing.assert_allclose(np.linalg.norm(rec['u'], axis=1), 1.0, rtol=1e-12)
    assert np.all(rec['length'] > 0)
    assert np.all(np.isfinite(rec['psi_bar']))


@pytest.mark.slow
def test_only_flagged_regions_are_recorded(records):
    assert set(records['two']['region'].tolist()) == {0, 7}
    assert len(records['none']['region']) == 0
