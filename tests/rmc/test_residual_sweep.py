"""Sweep directions: the tandem and sweep-control residual estimators.

One random ray transport sweep from Q_s and one from Q_f of the frozen field,
with the same rays (openmc_rmc_sweep_field), give the segments crossing each
region with their chord-averaged MG angular fluxes. With sweep directions,
the residual's incident directions and quadrature weights
omega_j = 4 pi l_j / (rho V_R) come from those segments, the MG control on
direction j is the production of segment j's flux, and the restored mean is
the sweep's region emission Q-hat (docs/source/methods/rmc.rst,
methods_rmc_residual_family).

- The records cover every region; their lengths add up to the sweep's active
  track length.
- Q-hat, computed in Python from the records and the field file's tables,
  matches the field (a converged field is a fixed point of one sweep, up to
  the sweep's noise).
- The R term of every record is Q-hat at the record's region and energy.
- Tandem identity: with every segment of the region as a direction, the MG
  control summed over directions equals Q-hat exactly, record by record; with
  omega-proportional subsampling, its mean does.
- The CE production (density branch plus sampler-route particles) has the
  same mean with sweep and with isotropic directions.
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
def sw(pincell, field_files):
    import openmc.lib.rmc as rmc
    model, mesh = pincell
    with pipeline.sweep_session(model, mesh, field_files['mgxs'], field_files['field'],
                                field_files['work'] / 'sweep_field', rays=500):
        rmc.field_load(field_files['field'], field_files['lifts'], field_files['volumes'])
        rmc.sweep_init()
        rmc.sweep_iterate(3)          # every region exists
        n = rmc.sweep_field(batch=4)
        with h5py.File(field_files['field']) as h:
            G = int(h.attrs['n_groups'])
            ids = list(h['materials/material_id'][()])
            M = len(ids)
            tables = dict(
                ids=ids,
                nu_sigma_s=h['materials/sigma_s'][()].reshape(M, G, G),  # [m][in][out]
                nu_sigma_f=h['materials/nu_sigma_f'][()].reshape(M, G),
                chi=h['materials/chi'][()].reshape(M, G),
                k=float(h.attrs['k_mg']))
        with h5py.File(field_files['volumes']) as h:
            V = h['V_A'][()]
        yield dict(rmc=rmc, n=n, tables=tables, V=V)
        rmc.sweep_finalize()
        rmc.field_free()




def _sweep_emission(sw, region, material_id):
    """Q-hat_{t,R,g} per source tag from the region's segments (Python)."""
    rmc, t = sw['rmc'], sw['tables']
    seg = rmc.sweep_field_region(region)
    w = 4 * math.pi * seg['length'] / (seg['rho'] * sw['V'][region])
    m = t['ids'].index(material_id)
    out = {}
    for y in 'sf':
        phi = (w[:, None] * seg['psi_' + y]).sum(axis=0)       # [G]
        out[('s', y)] = phi @ t['nu_sigma_s'][m]
        out[('f', y)] = t['chi'][m] * (t['nu_sigma_f'][m] @ phi)
    return out


def _regions(rmc):
    """Field region and material ID at a grid of points over the pincell."""
    x = np.linspace(-problems.PITCH / 2 + 1e-3, problems.PITCH / 2 - 1e-3, 60)
    X, Y = np.meshgrid(x, x)
    pts = np.c_[X.ravel(), Y.ravel(), np.zeros(X.size)]
    reg = rmc.field_locate(pts)
    out = {}
    for p, r in zip(pts, reg):
        if r >= 0 and r not in out:
            mat = problems.MAT_FUEL if np.hypot(p[0], p[1]) < problems.R_FUEL else problems.MAT_MOD
            out[int(r)] = mat
    return out


@pytest.mark.slow
def test_records_cover_regions(sw):
    rmc = sw['rmc']
    n_regions = rmc.field_n_regions()
    counts = np.array([len(rmc.sweep_field_region(r)['length']) for r in range(n_regions)])
    assert np.all(counts > 0)
    assert counts.sum() == sw['n']
    total = sum(rmc.sweep_field_region(r)['length'].sum() for r in range(n_regions))
    rho = rmc.sweep_field_region(0)['rho']
    V_box = problems.PITCH**2 * problems.HEIGHT
    # Random ray accumulates segment lengths in float
    assert total == pytest.approx(rho * V_box, rel=1e-6)


@pytest.mark.slow
def test_sweep_emission_matches_field(sw):
    rmc = sw['rmc']
    regs = _regions(rmc)
    ratios = {('s', 's'): [], ('f', 'f'): []}
    tot = {'s': [0.0, 0.0], 'f': [0.0, 0.0]}
    for r, mat in regs.items():
        q = _sweep_emission(sw, r, mat)
        for t in 'sf':
            Q = rmc.field_emission(r, t)
            Qhat = q[(t, 's')] + q[(t, 'f')]
            if t == 'f':
                Qhat = Qhat / sw['tables']['k']
            tot[t][0] += sw['V'][r] * Qhat.sum()
            tot[t][1] += sw['V'][r] * Q.sum()
            if Q.sum() > 0:
                ratios[(t, t)].append(Qhat.sum() / Q.sum())
    for t in 'sf':
        assert tot[t][0] == pytest.approx(tot[t][1], rel=0.01), t
        assert np.median(np.abs(np.array(ratios[(t, t)]) - 1)) < 0.05


@pytest.mark.slow
@pytest.mark.parametrize('tag', ['s', 'f'])
def test_R_term_is_sweep_emission(sw, tag):
    rmc = sw['rmc']
    regs = _regions(rmc)
    rec = rmc.residual_sample(tag, 200, seed=31, directions='sweep', n_directions=4,
                              control=True, sampler_route=False, E_range=E_RANGE,
                              tail='fixed_length', length=0.5)
    region = rmc.field_locate(rec['r'])
    h = rmc.field_shape(region, rec['E'], tag)
    checked = 0
    for i, r in enumerate(region):
        if r not in regs:
            continue
        q = _sweep_emission(sw, int(r), regs[int(r)])
        g = np.searchsorted(rmc.field_groups(), rec['E'][i], side='right') - 1
        for y in 'sf':
            want = h[i] / (4 * math.pi) * q[(tag, y)][g]
            assert rec['R_' + y][i] == pytest.approx(want, rel=1e-9, abs=1e-300)
        checked += 1
    assert checked > 100


@pytest.mark.slow
@pytest.mark.parametrize('tag', ['s', 'f'])
def test_tandem_identity(sw, tag):
    """Every segment as a direction: sum_j omega_j q^MG_j = Q-hat exactly, so
    the tandem estimator's bracket c Q-hat - F vanishes and M = R."""
    rmc = sw['rmc']
    rec = rmc.residual_sample(tag, 30, seed=41, directions='sweep', n_directions=0,
                              control=True, sampler_route=False, E_range=E_RANGE,
                              tail='fixed_length', length=0.3)
    for y in 'sf':
        np.testing.assert_allclose(rec['M_' + y], rec['R_' + y], rtol=1e-10,
                                   atol=1e-14 * np.abs(rec['R_' + y]).max())


@pytest.mark.slow
@pytest.mark.parametrize('tag,control_energy', [('s', 'exact'), ('f', 'exact'),
                                                ('f', 'same_energy')])
def test_subsampled_control_mean(sw, tag, control_energy):
    rmc = sw['rmc']
    rec = rmc.residual_sample(tag, 3000, seed=43, directions='sweep', n_directions=4,
                              control=True, control_energy=control_energy,
                              incident='flux', sampler_route=False, E_range=E_RANGE,
                              tail='fixed_length', length=0.3)
    d = rec['sampler'] == 0
    for y in 'sf':
        x = (rec['M_' + y][d] - rec['R_' + y][d]) / rec['density'][d]
        scale = np.abs(rec['R_' + y][d] / rec['density'][d]).mean()
        if scale == 0:
            continue
        mean, sem = x.mean(), x.std(ddof=1) / math.sqrt(x.size)
        assert abs(mean) <= 5 * sem + 1e-12 * scale, (y, mean / scale, sem / scale)


def _high_mean(rmc, rec, N, k):
    w = rmc.residual_weights(rec, N, (1.0, 1.0), (0.0, 0.0), k, 0.0, 0.0)
    # Per sample: density record plus its sampler-route particles
    dens = rec['sampler'] == 0
    sample_of = np.cumsum(np.r_[0, dens[:-1]])
    per = np.zeros(N)
    np.add.at(per, sample_of, w * N)
    return per.mean(), per.std(ddof=1) / math.sqrt(N)


@pytest.mark.slow
@pytest.mark.parametrize('tag', ['s', 'f'])
def test_direction_sources_agree(sw, tag):
    """G4.2 for the direction source: the CE production (with sampler-route
    particles) has the same mean with sweep and with isotropic directions."""
    rmc = sw['rmc']
    k = sw['tables']['k']
    N = 4000
    kw = dict(E_range=E_RANGE, tail='roulette', cutoff=math.exp(-4), beta=0.5,
              incident='flux')
    iso = rmc.residual_sample(tag, N, seed=51, directions='isotropic', n_directions=4,
                              **kw)
    swp = rmc.residual_sample(tag, N, seed=52, directions='sweep', n_directions=4, **kw)
    m1, s1 = _high_mean(rmc, iso, N, k)
    m2, s2 = _high_mean(rmc, swp, N, k)
    assert abs(m1 - m2) <= 5 * math.hypot(s1, s2), (m1, s1, m2, s2)
