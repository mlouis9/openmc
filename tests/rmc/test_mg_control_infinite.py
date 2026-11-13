"""The multigroup control against random ray in the infinite medium (gate
G3.6, deterministic part).

The MG control evaluates multigroup production at a destination from MG
angular fluxes accumulated along a backward path (MGTables, MGControl, and
the MG channels of BackwardFlux::evaluate_all). Its inputs are the field and
the exported MG tables only, so it can be checked against the random ray
solution that produced the field: at amplitudes a = (1, 1) and k = k_MG, the
MG production of a converged field, averaged over directions, is the field.

In the infinite medium psi^MG_g = Q_g / (4 pi Sigma_t,g) on every path, so
the check is deterministic (fixed-length tail long enough that the
truncation is below round-off): psi^MG times 4 pi Sigma_t from the field
file's own tables equals Q, which fixes the group order of MGTables, and the
production from one direction equals Q_{t,g} for both tags.
"""
import math

import h5py
import numpy as np
import pytest

import openmc.lib

import pipeline
import problems


def _group_points(rmc, region, tag):
    """One energy per group where the tag's shape is positive, and the shape
    there."""
    edges = rmc.field_groups()
    E = np.sqrt(edges[:-1] * edges[1:])
    out_E, out_h = [], []
    for g in range(len(E)):
        grid = np.geomspace(max(edges[g], 1e-6 * edges[g + 1]), edges[g + 1], 402)[1:-1]
        h = rmc.field_shape(np.full(grid.size, region), grid, tag)
        i = int(np.argmax(h))
        out_E.append(grid[i])
        out_h.append(h[i])
    return np.array(out_E), np.array(out_h)


def _production_by_group(rmc, r, u, region, tag, k, **tail):
    """Group emission estimated from one direction: the direction's weight
    4 pi times q^MG, integrated over the outgoing solid angle (4 pi) and the
    outgoing group (q is h(E) / (4 pi) times a constant per group), for the
    field Q_s + Q_f, with the 1/k of fission production. Its mean over
    isotropic directions is sum_g' K(g' -> g) phi^MG_g'."""
    E, h = _group_points(rmc, region, tag)
    q_s, q_f, _, _ = rmc.mg_control(r, u, E, tag, **tail)
    x = (4 * math.pi)**2 * (q_s + q_f) / np.where(h > 0, h, np.inf)
    return x / k if tag == 'f' else x


@pytest.fixture(scope='module')
def inf(infinite_files):
    import openmc.lib.rmc as rmc
    f = infinite_files
    with pipeline.ce_session(f['model'], f['mesh'], f['work'] / 'ce_mg'):
        rmc.field_load(f['field'], f['lifts'])
        with h5py.File(f['field']) as h:
            k = float(h.attrs['k_mg'])
            ids = list(h['materials/material_id'][()])
            G = int(h.attrs['n_groups'])
            m = ids.index(problems.MAT_FUEL)
            # The file's tables: [material][temperature][group], ascending
            sigma_t = h['materials/sigma_t'][()].reshape(len(ids), -1, G)[m, 0]
        yield dict(rmc=rmc, k=k, sigma_t=sigma_t)
        rmc.field_free()


@pytest.mark.slow
def test_mg_flux_infinite_medium(inf):
    rmc = inf['rmc']
    r = np.array([0.1, -0.2, 0.3])
    u = np.array([0.3, -0.5, 0.81])
    u /= np.linalg.norm(u)
    region = int(rmc.field_locate(r)[0])
    _, _, psi_s, psi_f = rmc.mg_control(r, u, [1.0], 's', tail='fixed_length',
                                        length=400.0)
    for psi, tag in ((psi_s, 's'), (psi_f, 'f')):
        Q = rmc.field_emission(region, tag)
        np.testing.assert_allclose(4 * math.pi * inf['sigma_t'] * psi, Q,
                                   rtol=1e-10, atol=1e-14 * Q.max())


@pytest.mark.slow
@pytest.mark.parametrize('tag', ['s', 'f'])
def test_mg_production_infinite_medium(inf, tag):
    rmc = inf['rmc']
    r = np.array([0.1, -0.2, 0.3])
    u = np.array([-0.6, 0.2, 0.77])
    u /= np.linalg.norm(u)
    region = int(rmc.field_locate(r)[0])
    x = _production_by_group(rmc, r, u, region, tag, inf['k'], tail='fixed_length',
                             length=400.0)
    Q = rmc.field_emission(region, tag)
    ok = Q > 1e-6 * Q.max()
    # Random ray's power iteration on one region converges to its own
    # eigenvector; the residual is its convergence error
    np.testing.assert_allclose(x[ok], Q[ok], rtol=1e-4)


