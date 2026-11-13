"""Residual samples in the infinite medium (gates G4.2 and G4.3).

In a homogeneous, reflective medium with a flat field the means of the
residual record terms have closed forms in one-dimensional energy
integrals, which the test evaluates by quadrature on a fine grid:

- proposal (G4.3): destinations are uniform in the field's box, energies
  log-uniform, directions isotropic, and each record's density is their
  product;
- high term, tag s (elastic production): E[C_s / density] =
  4 pi V int Sigma_el(E') q_s(E') / Sigma_t(E') dE';
- high term, tag f (fission production): E[C_s / density] =
  4 pi V int nu Sigma_f(E') q_s(E') / Sigma_t(E') dE';
- low term: E[U / density] = 4 pi V int q_t(E) dE;
- where a term vanishes (fission production in the resonance window has no
  fission emission there) every record's term is exactly zero;
- weights: the C++ weight formula against NumPy, and linear in a and b.

The backward flux uses the roulette tail, which is unbiased, so the means
are exact apart from the quadrature.
"""
import math

import numpy as np
import pytest

import openmc.lib

import mg
import pipeline
import problems

E_RANGE = (1e-5, 2e7)
EIN_RANGE = (1e-5, 2e7)


@pytest.fixture(scope='module')
def inf(field_files, tmp_path_factory):
    import openmc.lib.rmc as rmc
    model, mesh = problems.infinite_medium()
    work = tmp_path_factory.mktemp('infinite_residual')
    field_path, _, _, _ = mg.run_random_ray(
        model, mesh, work / 'rr', mgxs_particles=2000, rays=100, batches=10,
        inactive=5)
    with pipeline.ce_session(model, mesh, work / 'ce'):
        rmc.field_load(field_path, field_files['lifts'])
        # Quadrature grid, fine enough for the U-238 resonances
        g = np.geomspace(*E_RANGE, 400_001)
        region = np.full(g.size, int(rmc.field_locate(np.zeros(3))[0]))
        q_s = rmc.field_evaluate(region, g, 's')
        q_f = rmc.field_evaluate(region, g, 'f')
        sigma_t, sigma_el, nu_sigma_f = rmc.xs(problems.MAT_FUEL, g)
        V = problems.PITCH**2 * problems.HEIGHT
        yield dict(rmc=rmc, g=g, q_s=q_s, q_f=q_f, t=sigma_t, el=sigma_el,
                   nf=nu_sigma_f, V=V)
        rmc.field_free()


def _mean_and_sem(x):
    return x.mean(), x.std(ddof=1) / math.sqrt(x.size)


def _check(x, exact, rel=3e-3):
    """Mean within 5 standard errors, or within the quadrature's accuracy."""
    mean, sem = _mean_and_sem(x)
    if exact != 0.0:
        print(f'\nmean/exact - 1 = {mean / exact - 1:+.2e}, '
              f'sem/exact = {sem / abs(exact):.2e}')
    assert abs(mean - exact) <= max(5 * sem, rel * abs(exact)), \
        f'mean {mean:.6g} +- {sem:.2g}, exact {exact:.6g}'


@pytest.mark.slow
def test_proposal_density(inf):
    rmc = inf['rmc']
    rec = rmc.residual_sample('s', 20_000, seed=3, n_directions=1, E_range=E_RANGE)
    E = rec['E']
    want = 1 / (E * math.log(E_RANGE[1] / E_RANGE[0])) / (4 * math.pi) / inf['V']
    np.testing.assert_allclose(rec['density'], want, rtol=1e-12)
    # Log-uniform energies, isotropic directions, uniform positions
    x = np.log(E / E_RANGE[0]) / math.log(E_RANGE[1] / E_RANGE[0])
    for y in (x, 0.5 * (rec['u'][:, 2] + 1), rec['r'][:, 0] / problems.PITCH + 0.5):
        counts, _ = np.histogram(y, np.linspace(0, 1, 21))
        chi2 = np.sum((counts - 1000)**2 / 1000)
        assert chi2 < 19 + 5 * math.sqrt(38), chi2
    np.testing.assert_allclose(np.linalg.norm(rec['u'], axis=1), 1.0, rtol=1e-12)


@pytest.mark.slow
def test_elastic_high_term(inf):
    rmc = inf['rmc']
    rec = rmc.residual_sample('s', 40_000, seed=5, E_range=E_RANGE)
    exact = 4 * math.pi * inf['V'] * np.trapezoid(
        inf['el'] * inf['q_s'] / inf['t'], inf['g'])
    _check(rec['C_s'] / rec['density'], exact)


@pytest.mark.slow
@pytest.mark.parametrize('Ein_range', [EIN_RANGE, (1e-3, 1.0), (1.0, 1e3)])
def test_fission_high_term(inf, Ein_range):
    """Over all incident energies, and over windows (thermal; the U-238
    resolved resonances) that the proposal samples more densely."""
    rmc = inf['rmc']
    rec = rmc.residual_sample('f', 200_000, seed=7, E_range=E_RANGE, Ein_range=Ein_range)
    g = inf['g']
    inside = (g >= Ein_range[0]) & (g <= Ein_range[1])
    for key, q in (('C_s', inf['q_s']), ('C_f', inf['q_f'])):
        f = inf['nf'] * q / inf['t']
        exact = 4 * math.pi * inf['V'] * np.trapezoid(f[inside], g[inside])
        _check(rec[key] / rec['density'], exact)


@pytest.mark.slow
@pytest.mark.parametrize('tag', ['s', 'f'])
def test_low_term(inf, tag):
    rmc = inf['rmc']
    rec = rmc.residual_sample(tag, 100_000, seed=9, n_directions=1, E_range=E_RANGE)
    q = inf['q_s'] if tag == 's' else inf['q_f']
    exact = 4 * math.pi * inf['V'] * np.trapezoid(q, inf['g'])
    _check(rec['U'] / rec['density'], exact)


@pytest.mark.slow
def test_weights(inf):
    rmc = inf['rmc']
    recs = [rmc.residual_sample(t, 50, seed=11, n_directions=2) for t in 'sf']
    rec = {k: np.concatenate([r[k] for r in recs]) for k in recs[0]}
    a, b, k, N = (0.7, 1.3), (0.9, 1.1), 1.25, 50
    w = rmc.residual_weights(rec, N, a, b, k)
    f = rec['tag'] == 1
    high = (a[0] * rec['C_s'] + a[1] * rec['C_f']) / np.where(f, k, 1.0)
    low = np.where(f, b[1], b[0]) * rec['U']
    np.testing.assert_allclose(w, (high - low) / (N * rec['density']), rtol=1e-13)
    # Linear in (a, b)
    w2 = rmc.residual_weights(rec, N, (2 * a[0], 2 * a[1]), (2 * b[0], 2 * b[1]), k)
    np.testing.assert_allclose(w2, 2 * w, rtol=1e-13)
