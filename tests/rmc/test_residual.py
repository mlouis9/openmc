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
- sampler-route particles: their mean is the sampler-route production
  4 pi V int Sigma_samp q_s / Sigma_t dE', with either incident proposal;
- the same-path control: M on the J directions and its restored mean R on
  L MG-only directions have equal expectations;
- weights: the C++ weight formula against NumPy, and linear in a and b;
- every channel on the sampler route (all_sampler), with K incident
  energies: the sampled particles' mean is the whole production of the tag,
  4 pi V int Sigma_P,t q_s / Sigma_t dE', for scattering and fission;
- the flux-difference control: the term-I and region-term parts of Z have
  opposite means, and the sampled production is unchanged.

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
def inf(infinite_files):
    import openmc.lib.rmc as rmc
    f = infinite_files
    with pipeline.ce_session(f['model'], f['mesh'], f['work'] / 'ce_residual'):
        rmc.field_load(f['field'], f['lifts'])
        # Quadrature grid, fine enough for the U-238 resonances
        g = np.geomspace(*E_RANGE, 400_001)
        region = np.full(g.size, int(rmc.field_locate(np.zeros(3))[0]))
        q_s = rmc.field_evaluate(region, g, 's')
        q_f = rmc.field_evaluate(region, g, 'f')
        sigma_t, sigma_el, nu_sigma_f = rmc.xs(problems.MAT_FUEL, g)
        _, by_route = rmc.channel_production(problems.MAT_FUEL, g)
        V = problems.PITCH**2 * problems.HEIGHT
        yield dict(rmc=rmc, g=g, q_s=q_s, q_f=q_f, t=sigma_t, el=sigma_el,
                   nf=nu_sigma_f, samp=by_route[('s', 'sampler')],
                   prod_s=by_route[('s', 'evaluator')] + by_route[('s', 'sampler')], V=V)
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
    rec = rmc.residual_sample('s', 20_000, seed=3, n_directions=1, E_range=E_RANGE,
                              sampler_route=False)
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
    rec = rmc.residual_sample('s', 40_000, seed=5, E_range=E_RANGE, sampler_route=False)
    exact = 4 * math.pi * inf['V'] * np.trapezoid(
        inf['el'] * inf['q_s'] / inf['t'], inf['g'])
    _check(rec['C_s'] / rec['density'], exact)


@pytest.mark.slow
@pytest.mark.parametrize('incident', ['log_uniform', 'flux'])
def test_sampler_route_term(inf, incident):
    """Sampler-route particles (U-238 and O-16 inelastic, (n,2n)): the sum of
    their C_s / (N density) per sample has mean
    4 pi V int Sigma_samp q_s / Sigma_t dE', with Sigma_samp the registry's
    sampler-route production. With the flux proposal the incident energies
    follow the direction's own MG flux."""
    rmc = inf['rmc']
    N = 20_000
    rec = rmc.residual_sample('s', N, seed=13, n_directions=4, E_range=E_RANGE,
                              incident=incident, Ein_range=EIN_RANGE)
    smp = rec['sampler'] == 1
    # Sum of sampler particles per sample: N samples of the destination
    per_sample = np.zeros(N)
    # Sampler records precede their sample's density record
    sample_of = np.cumsum(np.r_[0, (rec['sampler'][:-1] == 0)])
    np.add.at(per_sample, sample_of[smp], rec['C_s'][smp] / rec['density'][smp])
    exact = 4 * math.pi * inf['V'] * np.trapezoid(inf['samp'] * inf['q_s'] / inf['t'],
                                                   inf['g'])
    _check(per_sample, exact, rel=5e-3)
    # Outgoing directions are unit vectors
    np.testing.assert_allclose(np.linalg.norm(rec['u'][smp], axis=1), 1.0, rtol=1e-10)


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
    rec = rmc.residual_sample(tag, 100_000, seed=9, n_directions=1, E_range=E_RANGE,
                              sampler_route=False)
    q = inf['q_s'] if tag == 's' else inf['q_f']
    exact = 4 * math.pi * inf['V'] * np.trapezoid(q, inf['g'])
    _check(rec['U'] / rec['density'], exact)


@pytest.mark.slow
@pytest.mark.parametrize('tag', ['s', 'f'])
def test_control_mean_is_restored(inf, tag):
    """With the same-path control, the MG term on the J directions and its
    restored mean on L MG-only directions have the same expectation: the
    mean of (M - R) / density is zero."""
    rmc = inf['rmc']
    rec = rmc.residual_sample(tag, 4000, seed=21, n_directions=4, control=True,
                              n_control_directions=4, E_range=E_RANGE,
                              sampler_route=False)
    for y in 'sf':
        x = (rec['M_' + y] - rec['R_' + y]) / rec['density']
        scale = np.abs(rec['R_' + y] / rec['density']).mean()
        if scale == 0.0:
            continue
        mean, sem = _mean_and_sem(x)
        assert abs(mean) <= 5 * sem + 1e-12 * scale, (y, mean / scale, sem / scale)


def _numpy_weights(rec, N, a, b, k, c, lam):
    f = rec['tag'] == 1
    ik = np.where(f, 1.0 / k, 1.0)
    bt = np.where(f, b[1], b[0])
    high = a[0] * (rec['C_s'] - c * rec['M_s'] + c * rec['R_s']) + \
        a[1] * (rec['C_f'] - c * rec['M_f'] + c * rec['R_f'])
    swept = (a[0] * rec['R_s'] + a[1] * rec['R_f']) * ik
    dens = high * ik - bt * ((1 - lam) * rec['U'] + lam * swept)
    smp = (a[0] * rec['C_s'] + a[1] * rec['C_f']) * ik
    return np.where(rec['sampler'] == 1, smp, dens) / (N * rec['density'])


@pytest.mark.slow
def test_weights(inf):
    """G4.7 in miniature: weight_records equals the formula in NumPy for
    several scalars, including a transient a_f, and is linear in (a, b)."""
    rmc = inf['rmc']
    recs = [rmc.residual_sample(t, 50, seed=11, n_directions=2, control=True,
                                n_control_directions=2) for t in 'sf']
    rec = {k: np.concatenate([r[k] for r in recs]) for k in recs[0]}
    N = 50
    for a, b, k, c, lam in [((0.7, 1.3), (0.9, 1.1), 1.25, 0.0, 0.0),
                            ((1.0, 0.96), (1.0, 1.0), 0.64, 1.0, 0.0),
                            ((1.1, 0.9), (1.05, 0.95), 0.8, 0.5, 0.5)]:
        w = rmc.residual_weights(rec, N, a, b, k, c, lam)
        np.testing.assert_allclose(w, _numpy_weights(rec, N, a, b, k, c, lam),
                                   rtol=1e-13, atol=1e-300)
    w = rmc.residual_weights(rec, N, (0.7, 1.3), (0.9, 1.1), 1.25, 0.5)
    w2 = rmc.residual_weights(rec, N, (1.4, 2.6), (1.8, 2.2), 1.25, 0.5)
    np.testing.assert_allclose(w2, 2 * w, rtol=1e-13)


def _per_sample(rec, N, value, mask):
    sample_of = np.cumsum(np.r_[0, (rec['sampler'][:-1] == 0)])
    out = np.zeros(N)
    np.add.at(out, sample_of[mask], value[mask])
    return out


@pytest.mark.slow
@pytest.mark.parametrize('tag', ['s', 'f'])
@pytest.mark.parametrize('common_numbers', [False, True])
def test_all_sampler_high_term(inf, tag, common_numbers):
    rmc = inf['rmc']
    N = 20_000
    rec = rmc.residual_sample(tag, N, seed=17, n_directions=4, E_range=E_RANGE,
                              all_sampler=True, n_energies=3,
                              common_numbers=common_numbers)
    smp = rec['sampler'] == 1
    assert np.all(rec['kind'][smp] == 1) and np.all(rec['channel'][smp] >= 0)
    # No density-branch production
    assert np.all(rec['C_s'][~smp] == 0.0)
    per = _per_sample(rec, N, rec['C_s'] / rec['density'], smp)
    prod = inf['prod_s'] if tag == 's' else inf['nf']
    exact = 4 * math.pi * inf['V'] * np.trapezoid(prod * inf['q_s'] / inf['t'], inf['g'])
    _check(per, exact, rel=5e-3)


@pytest.mark.slow
def test_flux_difference_control(inf):
    rmc = inf['rmc']
    N = 20_000
    rec = rmc.residual_sample('s', N, seed=19, n_directions=4, E_range=E_RANGE,
                              all_sampler=True, n_energies=2,
                              sampler_control='region', region_samples=4)
    k1, k2 = rec['kind'] == 1, rec['kind'] == 2
    assert k2.any()
    assert np.all(rec['C_s'][k2] == 0.0) and np.all(rec['C_f'][k2] == 0.0)
    z1 = _per_sample(rec, N, rec['Z'] / rec['density'], k1)
    z2 = _per_sample(rec, N, rec['Z'] / rec['density'], k2)
    assert z1.mean() < 0 < z2.mean()
    _check(z1 + z2, 0.0)
    per = _per_sample(rec, N, rec['C_s'] / rec['density'], k1)
    exact = 4 * math.pi * inf['V'] * np.trapezoid(
        inf['prod_s'] * inf['q_s'] / inf['t'], inf['g'])
    _check(per, exact, rel=5e-3)
