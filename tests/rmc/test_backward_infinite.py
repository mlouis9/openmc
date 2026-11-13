"""The backward CE characteristic in the infinite medium (gates G2.2, G2.3).

In a homogeneous medium with reflective faces and a flat field, the angular
flux has a closed form at every energy: psi = q(E) / Sigma_t(E), and after a
fixed length L, q (1 - exp(-Sigma_t L)) / Sigma_t, however the path is cut
into segments by reflections.

- fixed length: matches the closed form to round-off;
- roulette: the mean matches q / Sigma_t for three cutoffs (G2.3), and the
  variance falls as the cutoff falls;
- vacuum: refused, since no path reaches a vacuum boundary.
"""
import math

import numpy as np
import pytest

import openmc.lib

import mg
import pipeline
import problems

# Thermal, a U-238 resonance peak and its wing, epithermal, fast
ENERGIES = np.array([0.0253, 6.67, 7.5, 1e3, 1e6])
A = (1.0, 1.0)


@pytest.fixture(scope='module')
def inf(field_files, tmp_path_factory):
    import openmc.lib.rmc as rmc
    model, mesh = problems.infinite_medium()
    work = tmp_path_factory.mktemp('infinite_backward')
    field_path, _, _, _ = mg.run_random_ray(
        model, mesh, work / 'rr', mgxs_particles=2000, rays=100, batches=10,
        inactive=5)
    with pipeline.ce_session(model, mesh, work / 'ce'):
        rmc.field_load(field_path, field_files['lifts'])
        r = np.array([0.1, -0.2, 0.3])
        region = int(rmc.field_locate(r)[0])
        q = (A[0] * rmc.field_evaluate(np.full(ENERGIES.size, region), ENERGIES, 's')
             + A[1] * rmc.field_evaluate(np.full(ENERGIES.size, region), ENERGIES, 'f'))
        sigma = rmc.xs_total(problems.MAT_FUEL, ENERGIES)
        yield dict(rmc=rmc, r=r, q=q, sigma=sigma)
        rmc.field_free()


U = np.array([0.3, -0.5, 0.81])
U = U / np.linalg.norm(U)


@pytest.mark.slow
def test_fixed_length_closed_form(inf):
    rmc = inf['rmc']
    for L in (0.05, 1.0, 7.0):
        got, n = rmc.backward_psi(inf['r'], U, ENERGIES, a=A, tail='fixed_length',
                                  length=L)
        want = inf['q'] * (-np.expm1(-inf['sigma'] * L)) / inf['sigma']
        np.testing.assert_allclose(got, want, rtol=1e-12)


@pytest.mark.slow
@pytest.mark.parametrize('cutoff', [math.exp(-4), math.exp(-8), math.exp(-16)])
def test_roulette_mean_is_q_over_sigma(inf, cutoff):
    rmc = inf['rmc']
    samples = np.array([rmc.backward_psi(inf['r'], U, ENERGIES, a=A, tail='roulette',
                                         cutoff=cutoff, seed=s)[0]
                        for s in range(1, 401)])
    exact = inf['q'] / inf['sigma']
    mean = samples.mean(axis=0)
    sem = samples.std(axis=0, ddof=1) / math.sqrt(len(samples))
    # Where roulette barely acts the spread is at round-off, so allow that
    z = (mean - exact) / np.maximum(sem, 1e-12 * np.abs(exact))
    assert np.all(np.abs(z) < 5), z
    assert np.all(np.abs(mean / exact - 1) < 0.2)


@pytest.mark.slow
def test_roulette_variance_falls_with_cutoff(inf):
    rmc = inf['rmc']

    def rel_var(cutoff):
        s = np.array([rmc.backward_psi(inf['r'], U, ENERGIES, a=A, tail='roulette',
                                       cutoff=cutoff, seed=k)[0] for k in range(1, 201)])
        return np.var(s, axis=0) / np.mean(s, axis=0)**2

    v4, v16 = rel_var(math.exp(-4)), rel_var(math.exp(-16))
    assert np.all(v16 <= v4 + 1e-12)
    assert np.all(v16 < 1e-6)   # e^-16 leaves almost nothing to roulette


@pytest.mark.slow
def test_vacuum_tail_refused_without_vacuum(inf):
    with pytest.raises(Exception, match='vacuum'):
        inf['rmc'].backward_psi(inf['r'], U, ENERGIES, tail='vacuum', max_segments=2000)
