"""Production densities against OpenMC's own collision samplers (gates G3.2
and G3.3).

Each evaluator is the density of what an OpenMC sampler does, so the test
of each is the same: sample with OpenMC on a scratch particle, histogram,
and compare with the evaluator integrated over the same bins.

- Fission (rmc::fission_energy_density, ContinuousTabular::evaluate,
  MaxwellEnergy::evaluate): the outgoing-energy law of U-235 and U-238 at
  incident energies on and between the tabulated incident grid, where the
  sampler's random choice of table and its rescaling of the outgoing energy
  both matter, and of U-236, whose prompt spectrum is a Maxwell law. The
  density integrates to its CDF.
- Elastic scattering off a stationary target (rmc::elastic_stationary): the
  joint (E, mu0) density for a 1/E incident spectrum, for C, O-16 (strongly
  forward-peaked above a few MeV), and U-238, including the kinematic edge
  E = alpha E'.
"""
import math

import numpy as np
import pytest

import openmc.lib

import pipeline

K_B = 8.617333262e-5          # eV/K
KT = 294.0 * K_B


def chi2_ok(observed, expected, min_expected=20):
    """chi^2 over bins with enough expected counts, within 5 sigma of its dof."""
    ok = expected > min_expected
    chi2 = float(np.sum((observed[ok] - expected[ok])**2 / expected[ok]))
    dof = int(ok.sum())
    return chi2 < dof + 5 * math.sqrt(2 * dof), chi2, dof


@pytest.fixture(scope='module')
def lib(pincell, field_files):
    """A CE session (the pincell's materials load C, O, and U data)."""
    import openmc.lib.rmc as rmc
    model, mesh = pincell
    with pipeline.ce_session(model, mesh, field_files['work'] / 'production'):
        yield rmc


# ---------------------------------------------------------------------------
# Fission
# ---------------------------------------------------------------------------

# On the incident grid (1.5 MeV) and between grid points where neighboring
# tables differ strongly (17.5 MeV, multi-chance fission; 0.25 MeV). U-236's
# prompt spectrum is a Maxwell law (MaxwellEnergy), not a table. Its
# restriction energy is -30 MeV, so the truncation at E - U never acts below
# energy_max, and only the conditioning on E_out < energy_max is exercised.
FISSION_CASES = [('U235', 1.5e6), ('U235', 1.75e7), ('U235', 2.5e5),
                 ('U238', 1.5e6), ('U238', 2.5e7), ('U238', 1e3),
                 ('U236', 1e3), ('U236', 1.5e6), ('U236', 1.2e7)]


@pytest.mark.slow
@pytest.mark.parametrize('nuclide,E_in', FISSION_CASES)
def test_fission_energy_law_matches_sampler(lib, nuclide, E_in):
    n = 400_000
    x = lib.sample_fission_energy(nuclide, E_in, n, seed=11)
    edges = np.concatenate([[0.0], np.geomspace(1e3, 2e7, 151), [np.inf]])
    observed, _ = np.histogram(x, edges)
    _, F = lib.fission_density(nuclide, E_in, edges[1:-1])
    F = np.concatenate([[0.0], F, [1.0]])
    ok, chi2, dof = chi2_ok(observed, n * np.diff(F))
    assert ok, f'chi2 = {chi2:.1f} for {dof} bins'


@pytest.mark.slow
@pytest.mark.parametrize('nuclide,E_in', FISSION_CASES[:3] + FISSION_CASES[-2:])
def test_fission_density_integrates_to_cdf(lib, nuclide, E_in):
    """The density and the CDF are the same law: trapezoid integrals of the
    density on a fine grid equal CDF differences."""
    for a, b in ((1e3, 1e5), (1e5, 1e6), (1e6, 5e6), (5e6, 1.9e7)):
        g = np.geomspace(a, b, 20_001)
        f, F = lib.fission_density(nuclide, E_in, g)
        assert np.trapezoid(f, g) == pytest.approx(F[-1] - F[0], rel=1e-5, abs=1e-12)
    _, F = lib.fission_density(nuclide, E_in, np.array([1e-12, 1e9]))
    assert F[0] == pytest.approx(0.0, abs=1e-15) and F[1] == pytest.approx(1.0, abs=1e-12)


@pytest.fixture(scope='module')
def synthetic_law(tmp_path_factory):
    """A ContinuousTabular law with two incident tables whose outgoing ranges
    and shapes differ ([1, 5] keV triangle at 100 keV, [4, 20] keV ramp at
    200 keV). In OpenMC's fission data every table spans the same range, so
    the sampler's rescaling is the identity there; this law exercises it."""
    import h5py
    import openmc.data
    import openmc.stats
    t0 = openmc.stats.Tabular([1e3, 2e3, 5e3], [0.0, 1.0, 0.0], 'linear-linear')
    t1 = openmc.stats.Tabular([4e3, 6e3, 2e4], [1.0, 0.5, 0.0], 'linear-linear')
    for t in (t0, t1):
        t.normalize()
        t.c = t.cdf()
    law = openmc.data.ContinuousTabular([2], [2], [1e5, 2e5], [t0, t1])
    path = tmp_path_factory.mktemp('law') / 'law.h5'
    with h5py.File(path, 'w') as f:
        law.to_hdf5(f.create_group('law'))
    return path


@pytest.mark.slow
@pytest.mark.parametrize('E_in', [1e5, 1.3e5, 1.7e5, 2e5])
def test_tabular_law_rescaling(lib, synthetic_law, E_in):
    """Between the incident grid points (1.3 and 1.7 x 1e5 eV) the sampler
    rescales the outgoing energy; the density and CDF must follow it. Without
    the rescaling or its Jacobian, this fails between grid points and passes
    on them."""
    edges = np.linspace(500, 2.1e4, 121)
    f, F, x = lib.tabular_law(synthetic_law, E_in, edges, n_samples=400_000, seed=7)
    observed, _ = np.histogram(x, edges)
    ok, chi2, dof = chi2_ok(observed, 400_000 * np.diff(F))
    assert ok, f'chi2 = {chi2:.1f} for {dof} bins'
    g = np.linspace(500, 2.1e4, 200_001)
    fg, _, _ = lib.tabular_law(synthetic_law, E_in, g)
    assert np.trapezoid(fg, g) == pytest.approx(1.0, abs=5e-5)


# ---------------------------------------------------------------------------
# Elastic scattering, stationary target
# ---------------------------------------------------------------------------

def _expected_elastic(lib, nuclide, Ea, Eb, E_edges, mu_edges, q=16):
    """Expected fraction in each (E, mu0) bin for incident density 1/E on
    [Ea, Eb], by Gauss-Legendre quadrature of the evaluator in (ln E, mu0)."""
    xg, wg = np.polynomial.legendre.leggauss(q)
    out = np.zeros((len(E_edges) - 1, len(mu_edges) - 1))
    for i in range(len(E_edges) - 1):
        a, b = math.log(E_edges[i]), math.log(E_edges[i + 1])
        lnE = 0.5 * (a + b) + 0.5 * (b - a) * xg
        wE = 0.5 * (b - a) * wg
        for j in range(len(mu_edges) - 1):
            c, d = mu_edges[j], mu_edges[j + 1]
            mu = 0.5 * (c + d) + 0.5 * (d - c) * xg
            wm = 0.5 * (d - c) * wg
            L, M = np.meshgrid(lnE, mu, indexing='ij')
            E = np.exp(L).ravel()
            E_in, f = lib.elastic_image(nuclide, KT, E, M.ravel())
            s = np.where((E_in >= Ea) & (E_in <= Eb), 1 / (E_in * math.log(Eb / Ea)), 0)
            out[i, j] = np.sum(s * f * E * np.outer(wE, wm).ravel())
    return out


ELASTIC_CASES = [('C0', 1e4, 1e5), ('C0', 2e6, 8e6), ('O16', 3e6, 9e6),
                 ('U238', 1e3, 2e3)]


@pytest.mark.slow
@pytest.mark.parametrize('nuclide,Ea,Eb', ELASTIC_CASES)
def test_elastic_kernel_matches_sampler(lib, nuclide, Ea, Eb):
    rng = np.random.default_rng(17)
    n = 400_000
    E_in = Ea * (Eb / Ea)**rng.random(n)
    E, mu = lib.sample_elastic(nuclide, KT, E_in, seed=23)
    E_edges = np.geomspace(E.min() * 0.999, Eb * 1.0001, 21)
    mu_edges = np.linspace(-1, 1, 21)
    observed, _, _ = np.histogram2d(E, mu, [E_edges, mu_edges])
    expected = n * _expected_elastic(lib, nuclide, Ea, Eb, E_edges, mu_edges)
    # The quadrature is coarse across the kinematic edge E = alpha E'
    assert expected.sum() / n == pytest.approx(1.0, abs=1e-3)
    ok, chi2, dof = chi2_ok(observed, expected)
    assert ok, f'chi2 = {chi2:.1f} for {dof} bins'


@pytest.mark.slow
def test_elastic_image_is_zero_outside_data(lib):
    E_in, f = lib.elastic_image('C0', KT, np.array([1e9, 1e-9]), np.array([0.3, 0.3]))
    assert np.all(f == 0.0)
    E_in, f = lib.elastic_image('C0', KT, np.array([1e5]), np.array([1.0]))
    assert E_in[0] == pytest.approx(1e5) and f[0] > 0
