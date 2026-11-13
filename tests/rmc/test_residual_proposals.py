"""Destination and outgoing-energy proposals of the residual sampler (gate
G4.3 for every mode).

Each record's density must be the density its destination and outgoing
state were drawn from. On the graphite pincell, with the region volumes:

- destinations: 'volume' (pi_R proportional to V_R, density 1 / sum V),
  'fissionable' (only fuel regions, density 1 / V_fuel), and 'tabulated'
  (given weights); region frequencies follow pi_R (chi^2);
- outgoing energies: the mixture (1 - beta) times the region's emission
  spectrum plus beta times log-uniform. The density of every record equals
  the mixture written out in Python from the field's emission and shapes, and
  the energies follow it (chi^2 on bins integrated on the lifts' fine grid).
"""
import math

import h5py
import numpy as np
import pytest

import openmc.lib

import pipeline
import problems

E_RANGE = (1e-5, 2e7)


@pytest.fixture(scope='module')
def lib(pincell, field_files):
    import openmc.lib.rmc as rmc
    model, mesh = pincell
    with pipeline.ce_session(model, mesh, field_files['work'] / 'proposals'):
        rmc.field_load(field_files['field'], field_files['lifts'], field_files['volumes'])
        with h5py.File(field_files['volumes']) as f:
            V = f['V_A'][()]
        with h5py.File(field_files['field']) as f:
            mat = f['regions/material_id'][()]
        with h5py.File(field_files['lifts']) as f:
            fine = f['fine_edges'][()]
        yield dict(rmc=rmc, V=V, mat=mat, fine=fine)
        rmc.field_free()


def _chi2_ok(observed, expected):
    ok = expected > 20
    chi2 = float(np.sum((observed[ok] - expected[ok])**2 / expected[ok]))
    dof = int(ok.sum())
    return chi2 < dof + 5 * math.sqrt(2 * dof), chi2, dof


@pytest.mark.slow
@pytest.mark.parametrize('mode', ['volume', 'fissionable', 'tabulated'])
def test_voxel_proposals(lib, mode):
    rmc, V, mat = lib['rmc'], lib['V'], lib['mat']
    n = 20_000
    weights = None
    if mode == 'volume':
        pi = V / V.sum()
    elif mode == 'fissionable':
        pi = np.where(mat == problems.MAT_FUEL, V, 0.0)
        pi /= pi.sum()
    else:
        weights = 1.0 + (np.arange(len(V)) % 3)        # 1, 2, 3, ...
        pi = weights / weights.sum()
    rec = rmc.residual_sample('s', n, seed=3, n_directions=1, sampler_route=False,
                              voxel=mode, voxel_weights=weights, E_range=E_RANGE)
    region = rmc.field_locate(rec['r'])
    assert np.all(pi[region] > 0)
    # density = pi_R / V_R times the log-uniform energy and isotropic direction
    E = rec['E']
    want = pi[region] / V[region] / (E * math.log(E_RANGE[1] / E_RANGE[0])) / (4 * math.pi)
    np.testing.assert_allclose(rec['density'], want, rtol=1e-12)
    counts = np.bincount(region, minlength=len(V))
    ok, chi2, dof = _chi2_ok(counts, n * pi)
    assert ok, f'chi2 = {chi2:.1f} for {dof} regions'


def _mixture_density(rmc, region, E, tag, beta):
    """(1 - beta) Q_{t,R,g(E)} h_t(E) / sum_g Q + beta / (E ln) per unit E."""
    groups = rmc.field_groups()
    out = beta / (E * math.log(E_RANGE[1] / E_RANGE[0]))
    for r in np.unique(region):
        m = region == r
        Q = rmc.field_emission(r, tag)
        if Q.sum() <= 0:
            out[m] = 1 / (E[m] * math.log(E_RANGE[1] / E_RANGE[0]))
            continue
        g = np.searchsorted(groups, E[m], side='right') - 1
        h = rmc.field_shape(np.full(m.sum(), r), E[m], tag)
        out[m] += (1 - beta) * Q[np.clip(g, 0, len(Q) - 1)] / Q.sum() * h
    return out


@pytest.mark.slow
@pytest.mark.parametrize('tag', ['s', 'f'])
def test_outgoing_mixture(lib, tag):
    rmc, V = lib['rmc'], lib['V']
    n, beta = 40_000, 0.3
    rec = rmc.residual_sample(tag, n, seed=5, n_directions=1, sampler_route=False,
                              voxel='volume', beta=beta, E_range=E_RANGE)
    d = rec['sampler'] == 0
    region = rmc.field_locate(rec['r'][d])
    E = rec['E'][d]
    want = (1 / V.sum()) * _mixture_density(rmc, region, E, tag, beta) / (4 * math.pi)
    np.testing.assert_allclose(rec['density'][d], want, rtol=1e-9)

    # Energies follow the mixture: expected counts per bin, integrated on the
    # fine grid where the shapes are constant
    fine = lib['fine']
    edges = np.geomspace(1e-5, 2e7, 41)
    mid = np.sqrt(fine[:-1] * fine[1:])
    lo = np.maximum(fine[:-1, None], edges[None, :-1])
    hi = np.minimum(fine[1:, None], edges[None, 1:])
    # Log-uniform measure of the overlap of each fine bin with each output
    # bin, and the fraction of the fine bin it is
    overlap = np.log(np.maximum(hi, lo) / lo)
    frac = overlap / np.log(fine[1:] / fine[:-1])[:, None]
    expected = np.zeros(len(edges) - 1)
    regs, counts = np.unique(region, return_counts=True)
    log_part = beta / math.log(E_RANGE[1] / E_RANGE[0])
    for r, c in zip(regs, counts):
        # The log-uniform part integrates exactly over each overlap; the
        # emission part is constant per fine bin
        f = _mixture_density(rmc, np.full(mid.size, r), mid, tag, beta)
        shape_part = (f - log_part / mid) * np.diff(fine)
        if rmc.field_emission(r, tag).sum() <= 0:
            shape_part[:] = 0.0
            expected += c * (overlap.sum(axis=0) / math.log(E_RANGE[1] / E_RANGE[0]))
        else:
            expected += c * (log_part * overlap.sum(axis=0) + shape_part @ frac)
    observed, _ = np.histogram(E, edges)
    ok, chi2, dof = _chi2_ok(observed, expected * n / expected.sum())
    assert ok, f'chi2 = {chi2:.1f} for {dof} bins'
