"""Forward correction flights against OpenMC on the graphite pincell (gates
G5.2, G5.4).

- G5.2: the children of a signed test bank, binned by material and outgoing
  energy, equal the first-collision production of the same bank in OpenMC
  (positive and negative parts as two source files, one-batch eigenvalue
  runs with an energy cutoff;
  examples/rmc/residual_gates.py).
- G5.4: the flights' signed track-length flux equals OpenMC's uncollided
  flux of the same sources.

The reference runs as separate OpenMC processes, so no openmc.lib session
may be open while it runs (Model.run would use it).
"""
import math

import numpy as np
import pytest

import openmc.lib

import pipeline
import problems
import residual_gates as gates

E_EDGES = np.array([1e-5, 0.1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6, 3e6, 2e7])
K = 1.3


def _test_bank(rng, m=2000):
    """A signed bank over the pincell: uniform positions, log-uniform
    energies, isotropic directions, 60 % positive weights."""
    lo = -np.array([problems.PITCH / 2, problems.PITCH / 2, problems.HEIGHT / 2])
    r = rng.uniform(lo, -lo, size=(m, 3))
    u = rng.normal(size=(m, 3))
    u /= np.linalg.norm(u, axis=1)[:, None]
    E = 10**rng.uniform(-3, 7, m)
    w = rng.uniform(0.2, 1.0, m) * np.where(rng.random(m) < 0.6, 1.0, -1.0)
    return dict(r=r, u=u, E=E, w=w, tag=np.zeros(m, dtype=np.int32))


def _material_of(r):
    return np.where(np.hypot(r[:, 0], r[:, 1]) < problems.R_FUEL, 0, 1)


@pytest.fixture(scope='module')
def pin(pincell, field_files):
    import openmc.lib.rmc as rmc
    model, mesh = pincell
    bank = _test_bank(np.random.default_rng(5))
    ref = gates.bank_reference(model, bank, field_files['work'] / 'flight_reference',
                               E_EDGES, [problems.MAT_FUEL, problems.MAT_MOD],
                               particles=200_000, runs=5, k=K)
    n = 400_000
    with pipeline.ce_session(model, mesh, field_files['work'] / 'ce_flight_pin'):
        b = rmc.Bank(**bank)
        out, st, flux = rmc.propagate(b, n, K, seed=13,
                                      tally_materials=[problems.MAT_FUEL, problems.MAT_MOD],
                                      tally_edges=E_EDGES)
        ch = out.arrays()
        out.free()
        # The flux in independent batches, for its standard error
        fluxes = []
        for i in range(8):
            o, _, f = rmc.propagate(b, n // 8, K, seed=100 + i,
                                    tally_materials=[problems.MAT_FUEL, problems.MAT_MOD],
                                    tally_edges=E_EDGES)
            o.free()
            fluxes.append(f)
        b.free()
    fluxes = np.array(fluxes)
    return dict(ref=ref, children=ch, n=n, stats=st, flux=fluxes.mean(0),
                flux_sem=fluxes.std(0, ddof=1) / math.sqrt(len(fluxes)))


def _compare(est, sem, want, want_std):
    err = np.hypot(sem, want_std)
    live = want_std > 0
    z = (est[live] - want[live]) / err[live]
    dof = int(live.sum())
    chi2 = float(np.sum(z**2))
    assert chi2 < dof + 5 * math.sqrt(2 * dof) and np.all(np.abs(z) < 5), \
        f'chi2 = {chi2:.1f} for {dof} bins, max |z| = {np.abs(z).max():.1f}'


@pytest.mark.slow
@pytest.mark.parametrize('tag', ['s', 'f'])
def test_children_match_openmc(pin, tag):
    h = gates.children_histogram(pin['children'], pin['n'], _material_of, E_EDGES, 2)
    _compare(*h[tag], *pin['ref'][tag])


@pytest.mark.slow
def test_track_length_flux_matches_uncollided(pin):
    """The flights' flux (mean and standard error over 8 independent
    batches) against OpenMC's uncollided track-length flux."""
    want, want_std = pin['ref']['flux']
    _compare(pin['flux'], pin['flux_sem'], want, want_std)
