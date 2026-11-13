"""The backward CE characteristic in the slab box (gates G2.4-style exactness
and the three tails), rmc::BackwardFlux.

The slab box has vacuum faces in x, so every backward path ends. That makes
the vacuum tail exact and deterministic, which these tests use as the
reference for the other two tails:

- vacuum: equals an independent Python sum over the tracer's segments, with
  cross sections from the probe and the field from FrozenField, to 1e-12;
- roulette: its mean over many seeds equals the vacuum value (unbiased);
- fixed length: a length beyond the path equals the vacuum value exactly; a
  shorter one equals the partial sum up to that length (the biased dead-length
  approximation, by design).
"""
import math

import numpy as np
import pytest

import openmc.lib

import mg
import pipeline
import problems

ENERGIES = np.array([0.0253, 1.0, 6.67, 1e3, 1e5, 2e6])
A = (0.8, 1.3)


@pytest.fixture(scope='module')
def slab(field_files, tmp_path_factory):
    import openmc.lib.rmc as rmc
    model, mesh = problems.slab_box()
    work = tmp_path_factory.mktemp('slab_backward')
    field_path, _, _, _ = mg.run_random_ray(
        model, mesh, work / 'rr', mgxs_path=field_files['mgxs'],
        rays=200, batches=12, inactive=4)
    with pipeline.ce_session(model, mesh, work / 'ce'):
        rmc.field_load(field_path, field_files['lifts'])
        id_of_index = {m._index: m.id for m in openmc.lib.materials.values()}
        yield dict(rmc=rmc, id_of_index=id_of_index)
        rmc.field_free()


def python_psi(rmc, id_of_index, r, u, E, a, length=None):
    """Independent sum of eq. rmc-backward-psi over the tracer's segments."""
    seg = rmc.trace(r, u)
    assert seg['ends_in_vacuum']
    psi = np.zeros_like(E)
    T = np.ones_like(E)
    travelled = 0.0
    for l, m, reg in zip(seg['length'], seg['material'], seg['region']):
        if length is not None:
            if travelled >= length:
                break
            l = min(l, length - travelled)
        travelled += l
        sigma = rmc.xs_total(id_of_index[int(m)], E)
        q = np.zeros_like(E)
        if reg >= 0:
            q = (a[0] * rmc.field_evaluate(np.full(E.size, reg), E, 's')
                 + a[1] * rmc.field_evaluate(np.full(E.size, reg), E, 'f'))
        psi += T * q * (-np.expm1(-sigma * l)) / sigma
        T *= np.exp(-sigma * l)
    return psi


def _paths(n, seed):
    rng = np.random.default_rng(seed)
    L, Y, Z = problems.SLAB_HALF_X, problems.PITCH / 2, problems.HEIGHT / 2
    out = []
    while len(out) < n:
        u = rng.normal(size=3)
        u /= np.linalg.norm(u)
        if abs(u[0]) > 0.2:
            out.append((rng.uniform([-L, -Y, -Z], [L, Y, Z]), u))
    return out


@pytest.mark.slow
def test_vacuum_tail_matches_independent_sum(slab):
    rmc = slab['rmc']
    for r, u in _paths(10, 1):
        got, _ = rmc.backward_psi(r, u, ENERGIES, a=A, tail='vacuum')
        want = python_psi(rmc, slab['id_of_index'], r, u, ENERGIES, A)
        np.testing.assert_allclose(got, want, rtol=1e-12, atol=0)
        assert np.all(got > 0)


@pytest.mark.slow
def test_roulette_is_unbiased(slab):
    """Mean of the roulette tail over seeds equals the exact vacuum value,
    with a cutoff high enough that roulette happens on most paths."""
    rmc = slab['rmc']
    r, u = _paths(1, 2)[0]
    exact, _ = rmc.backward_psi(r, u, ENERGIES, a=A, tail='vacuum')
    samples = np.array([rmc.backward_psi(r, u, ENERGIES, a=A, tail='roulette',
                                         cutoff=0.3, seed=s)[0]
                        for s in range(1, 3001)])
    mean = samples.mean(axis=0)
    sem = samples.std(axis=0, ddof=1) / math.sqrt(len(samples))
    # Energies where roulette barely acts have a spread at round-off level
    z = (mean - exact) / np.maximum(sem, 1e-12 * np.abs(exact))
    assert np.all(np.abs(z) < 5), z
    assert np.sum(sem > 1e-6 * np.abs(exact)) >= 2   # roulette did act


@pytest.mark.slow
def test_fixed_length_tail(slab):
    rmc = slab['rmc']
    r, u = _paths(1, 3)[0]
    exact, n_seg = rmc.backward_psi(r, u, ENERGIES, a=A, tail='vacuum')
    long, _ = rmc.backward_psi(r, u, ENERGIES, a=A, tail='fixed_length', length=1e3)
    np.testing.assert_allclose(long, exact, rtol=1e-14)
    for length in (0.3, 1.0, 2.5):
        got, _ = rmc.backward_psi(r, u, ENERGIES, a=A, tail='fixed_length',
                                  length=length)
        want = python_psi(rmc, slab['id_of_index'], r, u, ENERGIES, A, length=length)
        np.testing.assert_allclose(got, want, rtol=1e-12)
        # The truncated tail is non-negative, so the bias is downward
        assert np.all(got <= exact * (1 + 1e-12))
