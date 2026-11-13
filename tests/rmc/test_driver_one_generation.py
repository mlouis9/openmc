"""One generation of the RMC iteration against OpenMC (gate G6.2).

From a fixed state (a, b, k, bank) with a non-empty signed bank, the
unnormalized candidate of a generation, as the driver builds it with the
OpenMC backend (residual records weighted with a, b, k, plus the bank's
flight children), binned by material and outgoing energy, equals the sum of
two references that share no code with it:

- the first-collision production of the field a_s Q_s + a_f Q_f from
  OpenMC's transport, minus the field candidate b Q integrated exactly (as
  in G4.1, here with amplitudes and k different from 1);
- the first-collision production of the bank from OpenMC's transport (as in
  G5.2).

This is Step 1 of the iteration, checked in one generation. Then the
driver's normalization is checked against its own formulas: f, k_next, the
per-tag scaling, and the new amplitudes.
"""
import math

import h5py
import numpy as np
import pytest

import openmc.lib

import driver
import pipeline
import problems
import residual_gates as gates

E_EDGES = np.array([1e-5, 0.1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6, 3e6, 2e7])
MATERIALS = [problems.MAT_FUEL, problems.MAT_MOD]
A, B, K = (0.97, 1.02), (1.01, 0.98), 0.65
N, N_DELTA = 6000, 200_000


def _material_of(r):
    return np.where(np.hypot(r[:, 0], r[:, 1]) < problems.R_FUEL, 0, 1)


def _test_bank(rng, m=1500):
    lo = -np.array([problems.PITCH / 2, problems.PITCH / 2, problems.HEIGHT / 2])
    r = rng.uniform(lo, -lo, size=(m, 3))
    u = rng.normal(size=(m, 3))
    u /= np.linalg.norm(u, axis=1)[:, None]
    E = 10**rng.uniform(-3, 7, m)
    w = rng.uniform(0.002, 0.01, m) * np.where(rng.random(m) < 0.6, 1.0, -1.0)
    tag = (rng.random(m) < 0.3).astype(np.int32)
    return dict(r=r, u=u, E=E, w=w, tag=tag)


@pytest.fixture(scope='module')
def gen(pincell, field_files):
    problem = gates.plugin_problem()
    if problem:
        pytest.fail(problem) if gates.LIB.exists() else pytest.skip(problem)
    model, mesh = pincell
    files = field_files
    bank = _test_bank(np.random.default_rng(21))
    box = (mesh.lower_left, mesh.upper_right)
    with h5py.File(files['field']) as f:
        mat_id = f['regions/material_id'][()]
    regions = [np.flatnonzero(mat_id == m) for m in MATERIALS]
    # References first: no openmc.lib session may be open while they run
    fc = gates.first_collision_reference(model, mesh, files, box,
                                         files['work'] / 'g62_field', E_EDGES, MATERIALS,
                                         particles=300_000, runs=6, a=A, k=K)
    br = gates.bank_reference(model, bank, files['work'] / 'g62_bank', E_EDGES, MATERIALS,
                              particles=200_000, runs=5, k=K)
    import openmc.lib.rmc as rmc
    with pipeline.ce_session(model, mesh, files['work'] / 'g62_ce'):
        rmc.field_load(files['field'], files['lifts'], files['volumes'])
        F = {t: gates.field_integral(rmc, files, regions, E_EDGES, t) for t in 'sf'}
        backend = driver.OpenMCBackend(dict(n_directions=4, control=True,
                                            n_control_directions=2,
                                            control_energy='same_energy',
                                            incident='flux', beta=0.5,
                                            tail='roulette', cutoff=math.exp(-4)), c=1.0)
        # The driver's Step 1, as generation() does it, and the same records
        # drawn directly (same seeds), for errors per sample
        parts = {t: backend.residual(t, N, A, B, K, seed=31 + i)
                 for i, t in enumerate('sf')}
        direct = {}
        for i, t in enumerate('sf'):
            rec = rmc.residual_sample(t, N, seed=31 + i, **backend.opts)
            w = rmc.residual_weights(rec, N, A, B, K, 1.0, 0.0)
            direct[t] = gates.residual_histogram(rec, w, _material_of, E_EDGES, 2)
        children = backend.propagate(bank, N_DELTA, K, seed=41)
        state = driver.State(k=K, a=A, b=B, bank=bank)
        cfg = driver.Config(N={'s': 500, 'f': 500}, N_delta=20_000, N_B=10**9,
                            amplitudes=driver.FixedAmplitudes(B))
        new_state, diag = driver.generation(state, backend, cfg)
        masses = {t: backend.field_mass(t) for t in 'sf'}
        rmc.field_free()
    return dict(fc=fc, br=br, F=F, parts=parts, direct=direct, children=children,
                bank=bank,
                new_state=new_state, diag=diag, masses=masses)


@pytest.mark.slow
@pytest.mark.parametrize('tag', ['s', 'f'])
def test_candidate_matches_references(gen, tag):
    it = {'s': 0, 'f': 1}[tag]
    # Residual part: the driver's bank, binned; it must equal the same
    # records weighted directly, whose per-sample errors are used
    part = gen['parts'][tag]
    nb = len(E_EDGES) - 1
    m = _material_of(part['r'])
    b = np.searchsorted(E_EDGES, part['E'], side='right') - 1
    ok = (b >= 0) & (b < nb) & (part['tag'] == it)
    est_r = np.bincount(m[ok] * nb + b[ok], weights=part['w'][ok],
                        minlength=2 * nb).reshape(2, nb)
    direct, sem_r = gen['direct'][tag]
    np.testing.assert_allclose(est_r, direct, rtol=1e-10, atol=1e-14)
    # Children part
    h = gates.children_histogram(gen['children'], N_DELTA, _material_of, E_EDGES, 2)
    est = est_r + h[tag][0]
    sem = np.hypot(sem_r, h[tag][1])
    want = gen['fc'][tag][0] - B[it] * gen['F'][tag] + gen['br'][tag][0]
    want_std = np.hypot(gen['fc'][tag][1], gen['br'][tag][1])
    live = want_std > 0
    z = (est[live] - want[live]) / np.hypot(sem, want_std)[live]
    dof = int(live.sum())
    chi2 = float(np.sum(z**2))
    assert chi2 < dof + 5 * math.sqrt(2 * dof) and np.all(np.abs(z) < 5), \
        f'chi2 = {chi2:.1f} for {dof} bins, max |z| = {np.abs(z).max():.1f}'


@pytest.mark.slow
def test_normalization(gen):
    d, s = gen['diag'], gen['new_state']
    m = gen['masses']
    # f from its definition, with the unnormalized candidate's f-tag sum
    cand_f = d['sum_f'] / (K / (d['k_next'] * d['f']))
    assert d['f'] == pytest.approx(K * (B[1] * m['f'] + cand_f), rel=1e-12)
    assert d['k_next'] == pytest.approx(K * d['f'], rel=1e-14)
    assert s.a[0] == pytest.approx(B[0] / d['f'], rel=1e-14)
    assert s.a[1] == pytest.approx(B[1] * K / (d['k_next'] * d['f']), rel=1e-14)
    assert s.k == d['k_next'] and s.b == B
