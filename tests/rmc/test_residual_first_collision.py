"""The residual against first-collision tallies (gates G4.1, G4.2, G4.4).

The reference shares no code with the residual sampler beyond the field: the
field a_s Q_s + a_f Q_f is the source of an analog CE fixed-source run, and
the neutron production of each source particle's first collision is tallied
by material and outgoing energy (examples/rmc/residual_gates.py). The low
term is the field integrated over the same bins exactly.

On the graphite pincell, with a = b = (1, 1) and k = k_MG:

- G4.1 / G4.4 (high term alone): with b = 0, c = 0, the binned residual
  records (density branch and sampler-route particles) equal the
  first-collision production, per tag and bin;
- G4.4 (low term alone): with a = 0, they equal minus the field;
- G4.1 / G4.2 (the whole residual, every estimator): frozen uncontrolled,
  frozen with the same-path control, frozen with the sweep control, tandem,
  partial, and a run with region-based destinations and mixture proposals
  all equal the reference minus the field candidate F = b [(1 - lambda) Q +
  lambda Q-hat] (Q-hat from the sweep, integrated the same way).

Pass: chi^2 over the bins within 5 sigma of its degrees of freedom, and no
bin beyond 5 standard errors.
"""
import math

import h5py
import numpy as np
import pytest

import openmc.lib

import pipeline
import problems
import residual_gates as gates

E_EDGES = np.array([1e-5, 0.1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6, 3e6, 2e7])
MATERIALS = [problems.MAT_FUEL, problems.MAT_MOD]
N = 6000
TAIL = dict(tail='roulette', cutoff=math.exp(-4))

VARIANTS = {
    'frozen_uncontrolled': (dict(directions='isotropic', n_directions=8), 0.0, 0.0),
    'frozen_same_path': (dict(directions='isotropic', n_directions=4, control=True,
                              n_control_directions=4), 1.0, 0.0),
    'frozen_sweep': (dict(directions='sweep', n_directions=8, control=True), 1.0, 0.0),
    'tandem': (dict(directions='sweep', n_directions=8, control=True), 1.0, 1.0),
    'partial': (dict(directions='sweep', n_directions=8, control=True), 0.5, 0.5),
    'mixture_proposals': (dict(directions='isotropic', n_directions=4, control=True,
                               n_control_directions=2, control_energy='same_energy',
                               voxel='volume', beta=0.5, incident='flux'), 1.0, 0.0),
}


def _material_of(r):
    return np.where(np.hypot(r[:, 0], r[:, 1]) < problems.R_FUEL, 0, 1)


@pytest.fixture(scope='module')
def ref(pincell, field_files):
    problem = gates.plugin_problem()
    if problem:
        pytest.fail(problem) if gates.LIB.exists() else pytest.skip(problem)
    model, mesh = pincell
    with h5py.File(field_files['field']) as f:
        k = float(f.attrs['k_mg'])
        mat_id = f['regions/material_id'][()]
    box = (mesh.lower_left, mesh.upper_right)
    out = gates.first_collision_reference(
        model, mesh, field_files, box, field_files['work'] / 'first_collision', E_EDGES,
        MATERIALS, particles=300_000, runs=6, k=k)
    regions = [np.flatnonzero(mat_id == m) for m in MATERIALS]
    return dict(fc=out, k=k, regions=regions)


@pytest.fixture(scope='module')
def lib(pincell, field_files, ref):
    """Records of every variant, the field integrals, and the sweep's."""
    import openmc.lib.rmc as rmc
    model, mesh = pincell
    files = field_files
    out = {}
    with pipeline.sweep_session(model, mesh, files['mgxs'], files['field'],
                                files['work'] / 'first_collision_ce', rays=500):
        rmc.field_load(files['field'], files['lifts'], files['volumes'])
        rmc.sweep_init()
        rmc.sweep_iterate(3)
        rmc.sweep_field(batch=4)
        for name, (opts, c, lam) in VARIANTS.items():
            for t in 'sf':
                out[(name, t)] = rmc.residual_sample(
                    t, N, seed=101 + 7 * len(out), E_range=(1e-5, 2e7), **TAIL, **opts)
        out['F'] = {t: gates.field_integral(rmc, files, ref['regions'], E_EDGES, t)
                    for t in 'sf'}
        out['Qhat'] = {t: gates.field_integral(rmc, files, ref['regions'], E_EDGES, t,
                                               emission=_sweep_emission(rmc, files, t,
                                                                        ref['k']))
                       for t in 'sf'}
        rmc.sweep_finalize()
        rmc.field_free()
    out['rmc'] = rmc
    return out


def _sweep_emission(rmc, files, t, k):
    """Q-hat_{t,R,g} of the field sweep with a = (1, 1), 1/k for fission."""
    with h5py.File(files['field']) as h:
        G = int(h.attrs['n_groups'])
        ids = list(h['materials/material_id'][()])
        M = len(ids)
        nss = h['materials/sigma_s'][()].reshape(M, G, G)
        nsf = h['materials/nu_sigma_f'][()].reshape(M, G)
        chi = h['materials/chi'][()].reshape(M, G)
        mat_id = h['regions/material_id'][()]
    with h5py.File(files['volumes']) as h:
        V = h['V_A'][()]
    cache = {}

    def emission(r):
        if r in cache:
            return cache[r]
        seg = rmc.sweep_field_region(r)
        w = 4 * math.pi * seg['length'] / (seg['rho'] * V[r])
        phi = (w[:, None] * (seg['psi_s'] + seg['psi_f'])).sum(axis=0)
        m = ids.index(int(mat_id[r]))
        q = phi @ nss[m] if t == 's' else chi[m] * (nsf[m] @ phi) / k
        cache[r] = q
        return q
    return emission


def _compare(est, sem, want, want_std, statistical=True):
    """chi^2 and z over bins. With a statistical reference, bins where it has
    no events (zero standard error) carry no information and are skipped;
    they are the tails of the fission spectrum below 10 eV, where the
    expected production per bin is below 1e-6 of the total."""
    err = np.hypot(sem, want_std)
    live = (want_std > 0) if statistical else (err > 0)
    z = (est[live] - want[live]) / err[live]
    dof = int(live.sum())
    chi2 = float(np.sum(z**2))
    assert chi2 < dof + 5 * math.sqrt(2 * dof) and np.all(np.abs(z) < 5), \
        f'chi2 = {chi2:.1f} for {dof} bins, max |z| = {np.abs(z).max():.1f}'


@pytest.mark.slow
@pytest.mark.parametrize('tag', ['s', 'f'])
def test_high_term_matches_first_collision(lib, ref, tag):
    rmc = lib['rmc']
    rec = lib[('frozen_uncontrolled', tag)]
    w = rmc.residual_weights(rec, N, (1, 1), (0, 0), ref['k'])
    est, sem = gates.residual_histogram(rec, w, _material_of, E_EDGES, 2)
    want, want_std = ref['fc'][tag]
    _compare(est, sem, want, want_std)


@pytest.mark.slow
@pytest.mark.parametrize('tag', ['s', 'f'])
def test_low_term_alone(lib, ref, tag):
    rmc = lib['rmc']
    rec = lib[('frozen_uncontrolled', tag)]
    w = rmc.residual_weights(rec, N, (0, 0), (1, 1), ref['k'])
    est, sem = gates.residual_histogram(rec, w, _material_of, E_EDGES, 2)
    _compare(est, sem, -lib['F'][tag], np.zeros_like(est), statistical=False)


@pytest.mark.slow
@pytest.mark.parametrize('variant', list(VARIANTS))
@pytest.mark.parametrize('tag', ['s', 'f'])
def test_residual_matches_reference(lib, ref, variant, tag):
    rmc = lib['rmc']
    _, c, lam = VARIANTS[variant]
    rec = lib[(variant, tag)]
    w = rmc.residual_weights(rec, N, (1, 1), (1, 1), ref['k'], c, lam)
    est, sem = gates.residual_histogram(rec, w, _material_of, E_EDGES, 2)
    fc, fc_std = ref['fc'][tag]
    F = (1 - lam) * lib['F'][tag] + lam * lib['Qhat'][tag]
    _compare(est, sem, fc - F, fc_std)
