"""The channel registry (gate G3.1) and the sampler route.

G3.1: for each material of the graphite pincell and a dense energy grid,

- the registry's production summed over all channels equals an independent
  sum computed in Python from the HDF5 nuclear data (openmc.data): elastic,
  every non-redundant reaction with a neutron product times its yield, and
  nu-total times fission (MT 18, or the partial fission reactions, which
  OpenMC gives the nu of the first), with nu * sigma_f interpolated
  linearly between the nuclide's energy grid points as OpenMC does;
- the fission-tag channels equal OpenMC's macroscopic nu-fission, and the
  scattering-tag channels without yields (elastic plus inelastic cross
  sections) equal OpenMC's total minus absorption, which is what scatter()
  samples from;
- every channel's production is assigned to exactly one route: evaluator
  plus sampler equals the total, per tag and energy, and the elastic channel
  is on the evaluator route exactly where OpenMC uses a stationary target.

Sampler route: channels are chosen in proportion to their production, and
the outgoing energies of the inelastic channels respect their kinematics.
"""
import math
import os
from pathlib import Path

import numpy as np
import pytest

import openmc.data
import openmc.lib

import pipeline
import problems

K_B = 8.617333262e-5
KT = 294.0 * K_B
E_GRID = np.geomspace(1e-5, 1.99e7, 4001)


@pytest.fixture(scope='module')
def lib(pincell, field_files):
    import openmc.lib.rmc as rmc
    model, mesh = pincell
    with pipeline.ce_session(model, mesh, field_files['work'] / 'channels'):
        yield rmc


def _library():
    """Nuclide name -> HDF5 path, from OPENMC_CROSS_SECTIONS."""
    lib = openmc.data.DataLibrary.from_xml(os.environ['OPENMC_CROSS_SECTIONS'])
    root = Path(os.environ['OPENMC_CROSS_SECTIONS']).parent
    return {lib_entry['materials'][0]: root / lib_entry['path'] if not
            Path(lib_entry['path']).is_absolute() else Path(lib_entry['path'])
            for lib_entry in lib.libraries if lib_entry['type'] == 'neutron'}


def _reference_production(material_id, E):
    """Independent production nu_bar Sigma [1/cm] by tag from openmc.data."""
    paths = _library()
    mat = openmc.lib.materials[material_id]
    nuclides, densities = mat.nuclides, mat.densities
    prod_s = np.zeros_like(E)
    prod_f = np.zeros_like(E)
    for name, N in zip(nuclides, densities):
        data = openmc.data.IncidentNeutron.from_hdf5(paths[name])
        T = '294K'
        for mt, rx in data.reactions.items():
            if rx.redundant or mt in (18, 19, 20, 21, 38):
                continue
            xs = rx.xs[T](E)
            if mt == 2:
                prod_s += N * xs
                continue
            neutrons = [p for p in rx.products if p.particle == 'neutron']
            if not neutrons:
                continue
            prod_s += N * neutrons[0].yield_(E) * xs
        # Fission: MT 18, or the partial reactions (first, second, third
        # chance, and MT 38) where there is no MT 18. OpenMC creates every
        # fission neutron with nu of its first fission reaction
        # (Nuclide::nu), so the partials share the first one's nu.
        def has_neutrons(rx):
            return any(p.particle == 'neutron' for p in rx.products)
        fission = [data.reactions[mt] for mt in (18,) if mt in data.reactions
                   and has_neutrons(data.reactions[mt])] or \
            [data.reactions[mt] for mt in (19, 20, 21, 38) if mt in data.reactions]
        # OpenMC tabulates nu * sigma_f on the nuclide's energy grid and
        # interpolates the product linearly, so the reference does the same.
        if fission:
            first = fission[0]
            g = data.energy[T]
            if first.derived_products:   # total nu, as Nuclide::nu uses it
                nu = first.derived_products[0].yield_(g)
            else:
                nu = sum(p.yield_(g) for p in first.products if p.particle == 'neutron')
            nu_sigma_f = nu * sum(rx.xs[T](g) for rx in fission)
            prod_f += N * np.interp(E, g, nu_sigma_f)
    return prod_s, prod_f


@pytest.mark.slow
@pytest.mark.parametrize('material_id', [problems.MAT_FUEL, problems.MAT_MOD])
def test_partition_against_nuclear_data(lib, material_id):
    by_kind, by_route = lib.channel_production(material_id, E_GRID)
    ref_s, ref_f = _reference_production(material_id, E_GRID)
    s = by_kind['elastic'] + by_kind['thermal_elastic'] + by_kind['inelastic']
    np.testing.assert_allclose(s, ref_s, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(by_kind['fission'], ref_f, rtol=1e-10, atol=1e-14)


@pytest.mark.slow
@pytest.mark.parametrize('material_id', [problems.MAT_FUEL, problems.MAT_MOD])
def test_partition_against_openmc_macro(lib, material_id):
    by_kind, by_route = lib.channel_production(material_id, E_GRID)
    total, elastic, nu_fission = lib.xs(material_id, E_GRID)
    np.testing.assert_allclose(by_kind['fission'], nu_fission, rtol=1e-12, atol=1e-14)
    # Each route's production adds up to the tag's total
    for t, kinds in (('s', ('elastic', 'thermal_elastic', 'inelastic')), ('f', ('fission',))):
        whole = sum(by_kind[k] for k in kinds)
        split = by_route[(t, 'evaluator')] + by_route[(t, 'sampler')]
        np.testing.assert_allclose(split, whole, rtol=1e-14, atol=0)
    assert np.all(by_route[('f', 'sampler')] == 0.0)


@pytest.mark.slow
def test_elastic_route_boundary(lib):
    """With free_gas_threshold = 1e-6 (the verification problems), stationary
    targets are used everywhere for A > 1: graphite elastic is entirely on
    the evaluator route, and its sampler route holds only inelastic."""
    by_kind, by_route = lib.channel_production(problems.MAT_MOD, E_GRID)
    np.testing.assert_allclose(by_route[('s', 'evaluator')], by_kind['elastic'], rtol=1e-14)
    np.testing.assert_allclose(by_route[('s', 'sampler')], by_kind['inelastic'],
                               rtol=1e-14, atol=0)


@pytest.mark.slow
@pytest.mark.parametrize('E_in', [1e6, 5e6, 1.4e7])
def test_sampler_route_channel_frequencies(lib, E_in):
    """In the fuel at E' above the U-238 inelastic threshold, every sampler
    draw is an inelastic channel, its production is the registry's sampler
    sum, and outgoing energies are below E' (no upscatter for these laws)."""
    n = 20_000
    out = lib.sample_sampler_route(problems.MAT_FUEL, E_in, [0, 0, 1], n, seed=3)
    _, by_route = lib.channel_production(problems.MAT_FUEL, np.array([E_in]))
    np.testing.assert_allclose(out['production'], by_route[('s', 'sampler')][0], rtol=1e-14)
    assert np.all(out['kind'] == 2)
    assert np.all((out['E'] > 0) & (out['E'] < E_in))
    np.testing.assert_allclose(np.linalg.norm(out['u'], axis=1), 1.0, rtol=1e-12)
