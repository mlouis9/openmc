"""The RMC driver against the deterministic toy (gate G6.1).

The driver (examples/rmc/driver.py) sees its backend only through a few
operations. With ToyBackend, which applies the toy's operators in expected
values, the driver must reproduce the toy's own iteration
(examples/rmc/normalization_toy.py, run_emission) to round-off: k, the
field amplitudes a, the chosen amplitudes b, the bank's L1 norm, and the
field's share of fission production rho, for Scheme C with b = (1, 1), with
fixed b different from 1, and with per-tag mass matching (undamped and
damped), which converges to the CE/MG mass ratios of each tag.
"""
import numpy as np
import pytest

import driver
import normalization_toy as toy

N_GEN = 60


def _compare(rows, ref):
    for d, r in zip(rows, ref):
        for key_d, key_r in (('k', 'k'), ('a_s', 'a_s'), ('a_f', 'a_f'),
                             ('b_s', 'b_s'), ('b_f', 'b_f'), ('rho', 'rho')):
            assert d[key_d] == pytest.approx(r[key_r], rel=1e-12, abs=1e-14), \
                (d['n'], key_d, d[key_d], r[key_r])


def _l1(state):
    return float(np.abs(state.bank['s'] + state.bank['f']).sum())


@pytest.mark.parametrize('b', [(1.0, 1.0), (0.97, 0.995), (1.02, 0.98)])
def test_fixed_amplitudes(b):
    backend = driver.ToyBackend()
    cfg = driver.Config(amplitudes=driver.FixedAmplitudes(b))
    l1 = []
    _, rows = driver.run(backend, cfg, toy.kE_mg, N_GEN,
                         callback=lambda s, d: l1.append(_l1(s)))
    ref = toy.run_emission(*b, n_gen=N_GEN)
    _compare(rows, ref)
    # The bank after generation n is the toy's dQ at generation n + 1
    np.testing.assert_allclose(l1, [r['l1'] for r in ref[1:N_GEN + 1]], rtol=1e-11)


@pytest.mark.parametrize('damping', [1.0, 0.5])
def test_mass_matching(damping):
    backend = driver.ToyBackend()
    cfg = driver.Config(amplitudes=driver.MassMatching(damping))
    _, rows = driver.run(backend, cfg, toy.kE_mg, N_GEN)
    ref = toy.run_emission(rule=toy.mass_matching(damping), n_gen=N_GEN)
    _compare(rows, ref)
    # Fixed point: the field carries each tag's CE mass
    assert rows[-1]['b_s'] == pytest.approx(toy.Qs_ce.sum() / toy.us.sum(), rel=1e-6)
    assert rows[-1]['b_f'] == pytest.approx(toy.Qf_ce.sum() / toy.uf.sum(), rel=1e-6)


def test_eigenvalue_converges_to_ce():
    """The iteration converges to the CE eigenvalue whatever b is."""
    backend = driver.ToyBackend()
    for b in [(1.0, 1.0), (0.9, 1.1)]:
        state, _ = driver.run(backend, driver.Config(amplitudes=driver.FixedAmplitudes(b)),
                              toy.kE_mg, 400)
        assert state.k == pytest.approx(toy.kE_ce, rel=1e-10)


def test_nonpositive_production_stops():
    """f <= 0 raises; it is never clipped."""
    backend = driver.ToyBackend()

    class Broken(driver.ToyBackend):
        def residual(self, t, n, a, b, k, seed):
            out = super().residual(t, n, a, b, k, seed)
            if t == 'f':
                out['f'] = out['f'] - 10.0
            return out
    with pytest.raises(driver.EstimatorFailure):
        driver.run(Broken(), driver.Config(), toy.kE_mg, 1)
