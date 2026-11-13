"""Forward correction flights in the infinite medium (gates G5.1, G5.3).

A flight carries a parent of a signed bank, chosen by |w|, to its next
collision with OpenMC's cross sections and geometry, and returns the neutron
that collision produces through OpenMC's own scatter() or
sample_fission_neutron(), with signed weight sgn(w) W_abs / N times the
realized yield (scatter) or nu / k (fission)
(docs/source/methods/rmc.rst, methods_rmc_flights).

- G5.1: from one parent energy at a time, the mean child weight per tag is
  Sigma_s^yield / Sigma_t and nu Sigma_f / (k Sigma_t), from the channel
  registry's production and the total cross section.
- G5.3: a bank of (+w, x), (-w, x) pairs gives children whose binned means
  are zero.

The comparison with OpenMC is in test_flights_pincell.py.
"""
import math

import numpy as np
import pytest

import openmc.lib

import pipeline
import problems
import residual_gates as gates

E_PARENT = [0.0253, 6.67, 1e3, 1e5, 2e6, 1.4e7]
E_EDGES = np.array([1e-5, 0.1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6, 3e6, 2e7])
K = 1.3


@pytest.fixture(scope='module')
def inf(infinite_files):
    import openmc.lib.rmc as rmc
    f = infinite_files
    with pipeline.ce_session(f['model'], f['mesh'], f['work'] / 'ce_flights'):
        yield rmc


@pytest.mark.slow
@pytest.mark.parametrize('E', E_PARENT)
def test_homogeneous_expectation(inf, E):
    rmc = inf
    n = 40_000
    parent = rmc.Bank(r=[[0.1, -0.2, 0.3]], u=[[0.0, 0.6, 0.8]], E=[E], w=[1.0],
                      tag=['s'])
    out, st, _ = rmc.propagate(parent, n, K, seed=7)
    assert st['leaked'] == 0 and st['flights'] == n
    ch = out.arrays()
    total, _, nu_sigma_f = rmc.xs(problems.MAT_FUEL, [E])
    by_kind, _ = rmc.channel_production(problems.MAT_FUEL, np.array([E]))
    want = {0: (by_kind['elastic'] + by_kind['thermal_elastic'] + by_kind['inelastic'])[0]
            / total[0],
            1: nu_sigma_f[0] / (K * total[0])}
    for t in (0, 1):
        x = np.zeros(n)
        w = ch['w'][ch['tag'] == t] * n
        x[:w.size] = w
        mean, sem = x.mean(), x.std(ddof=1) / math.sqrt(n)
        # Where a channel is rare (fission at the 6.67 eV capture resonance),
        # few or no children are seen: bound the error by the Poisson error
        # of the expected count, with a per-child weight of at most 3
        sem = max(sem, math.sqrt(3 * want[t] / n))
        assert abs(mean - want[t]) <= 5 * sem + 1e-12, (t, mean, want[t], sem)
    out.free()
    parent.free()


@pytest.mark.slow
def test_signed_pairs_cancel(inf):
    rmc = inf
    rng = np.random.default_rng(3)
    m = 200
    r = rng.uniform(-0.6, 0.6, size=(m, 3))
    u = rng.normal(size=(m, 3))
    u /= np.linalg.norm(u, axis=1)[:, None]
    E = 10**rng.uniform(-2, 7, m)
    w = rng.uniform(0.5, 1.5, m)
    bank = rmc.Bank(r=np.r_[r, r], u=np.r_[u, u], E=np.r_[E, E], w=np.r_[w, -w],
                    tag=['s'] * (2 * m))
    n = 40_000
    out, _, _ = rmc.propagate(bank, n, K, seed=9)
    ch = out.arrays()
    h = gates.children_histogram(ch, n, lambda x: np.zeros(len(x), int), E_EDGES, 1)
    for t in 'sf':
        mean, sem = h[t]
        live = sem > 0
        z = mean[live] / sem[live]
        assert np.all(np.abs(z) < 5), (t, z)
        assert abs(ch['w'][ch['tag'] == {'s': 0, 'f': 1}[t]].sum()) <= \
            5 * math.sqrt((sem**2).sum()) + 1e-12
    out.free()
    bank.free()


