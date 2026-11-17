"""The continuous forward residual (forward_points) and its band partner on
the graphite pincell.

G8.16: the forward points' signed weights, summed by (material, outgoing
group), equal the forward rays' CE-minus-multigroup bin totals: the pointwise
estimator has the expectation of the binned one.

G8.17: the band partner's expected emission, computed from a band
first-flight sweep and elastic transfer tables (the closure), equals the
partner's own weights on forward points drawn with the band proposal; and the
CE terms keep their expectation under that proposal.
"""
import math

import numpy as np
import pytest

import cancellation as canc
import pipeline


@pytest.fixture(scope='module')
def lib(pincell, field_files):
    import openmc.lib.rmc as rmc
    model, mesh = pincell
    with pipeline.ce_session(model, mesh, field_files['work'] / 'ce_forward_points'):
        rmc.field_load(field_files['field'], field_files['lifts'], field_files['volumes'])
        yield rmc, mesh
        rmc.field_free()


def _k_mg(field_files):
    import field as fld
    return float(fld.FieldFile.load(field_files['field']).k_mg)


def _batches(fn, n):
    x = np.array([fn(i) for i in range(n)])
    return x.mean(axis=0), x.std(axis=0, ddof=1) / math.sqrt(n)


@pytest.mark.slow
def test_forward_points_have_the_forward_bins_expectation(lib, field_files):
    """G8.16."""
    rmc, mesh = lib
    k = _k_mg(field_files)
    rep = canc.Representation(rmc, field_files, mesh)
    tb = rep.forward_tables()
    edges = np.array(rmc.field_groups())
    G = len(edges) - 1
    mats = sorted(set(int(m) for m in rep.material_id if m >= 0))
    zmat = np.array([int(rep.material_id[rep.regions_of[z][0]]) if len(rep.regions_of[z]) else -1
                     for z in range(rep.Z)])

    def aggregate_points(tag, parts, i):
        acc = np.zeros((len(mats), G))
        for part in parts:
            pts, _ = rmc.forward_points(tag, part, 20_000, k=k, lam=0.5, seed=7000 + 10 * i + part)
            mid = zmat[rep.zone[np.maximum(pts['region'], 0)]]
            go = np.clip(np.searchsorted(edges, pts['E'], side='right') - 1, 0, G - 1)
            for a, m in enumerate(mats):
                sel = mid == m
                acc[a] += np.bincount(go[sel], weights=(pts['w_ce'] - pts['w_mg'])[sel], minlength=G)
        return acc

    def aggregate_bins(i):
        diff, _ = rmc.forward_residual_bins(20_000, tb, n_E=8, seed=8000 + i)
        D = diff.sum(axis=0)
        key = np.arange(D.size)
        t = key // (rep.D * rep.NE * rep.Z)
        z = (key // (rep.D * rep.NE)) % rep.Z
        e = (key // rep.D) % rep.NE
        out = np.zeros((2, len(mats), G))
        for a, m in enumerate(mats):
            for tt in (0, 1):
                sel = (t == tt) & (zmat[z] == m)
                out[tt, a] = np.bincount(rep.e2g[e[sel]], weights=D[sel], minlength=G)
        out[1] /= k   # tag f deposits are productions
        return out

    Ps, sPs = _batches(lambda i: aggregate_points('s', (0, 1, 2), i), 12)
    Pf, sPf = _batches(lambda i: aggregate_points('f', (0,), i), 12)
    B, sB = _batches(aggregate_bins, 12)
    for P, sP, t in ((Ps, sPs, 0), (Pf, sPf, 1)):
        z = (P - B[t]) / np.sqrt(sP**2 + sB[t]**2 + 1e-30)
        big = np.abs(B[t]) + np.abs(P) > 1e-3
        assert np.abs(z[big]).max() < 4.5, (t, z)


@pytest.mark.slow
def test_band_partner_closure_matches_its_forward_points(lib, field_files):
    """G8.17."""
    import h5py
    import openmc.lib
    import resolve_basis as rb
    rmc, mesh = lib
    k = _k_mg(field_files)
    rep = canc.Representation(rmc, field_files, mesh)
    edges = np.array(rmc.field_groups())
    edges[0] = 1e-5
    G = len(edges) - 1
    mats = {i: int(m) for i, m in enumerate(openmc.lib.materials)}
    id2i = {v: i for i, v in mats.items()}
    with h5py.File(field_files['field']) as f:
        region_mat = np.array([id2i.get(int(x), -1) for x in f['regions/material_id'][()]])
    nuclides = sorted({n for m in mats.values() for n in openmc.lib.materials[m].nuclides})
    grids = {g: rb.union_grid(nuclides, edges[g], edges[g + 1]) for g in range(G)}
    blocks, cross, sources, info = rb.build_subgroup(rmc, mats, region_mat, grids, 8, 8,
                                                     spacing='log')
    bands = {key: (v['band'], v['mass'], v['sigma_el']) for key, v in info.items()}
    for slot in (0, 1):
        rmc.resolve_load(len(mats), G, blocks, cross, sources, slot=slot, bands=bands)
    T = rb.elastic_band_transfer(rmc, mats, grids, info, edges)
    zmat = np.array([region_mat[rep.regions_of[z][0]] if len(rep.regions_of[z]) else -1
                     for z in range(rep.Z)])

    def by_material(C):
        return np.array([C[zmat == m].sum(axis=0) for m in mats])

    C, sC = _batches(lambda i: by_material(rb.band_closure(
        *rmc.band_sweep(20, dead=40.0, active=200.0, seed=900 + i), T, region_mat,
        rep.zone, rep.Z)), 6)

    def points(i):
        pts, d = rmc.forward_points('s', 8, 20_000, k=k, lam=0.45, seed=9100 + i)
        z = rep.zone[np.maximum(pts['region'], 0)]
        go = np.clip(np.searchsorted(edges, pts['E'], side='right') - 1, 0, G - 1)
        acc = np.bincount(z * G + go, weights=pts['w_mg'], minlength=rep.Z * G).reshape(rep.Z, G)
        return np.r_[by_material(acc).ravel(), d[0]]

    P, sP = _batches(points, 12)
    ce0, sce0 = _batches(lambda i: rmc.forward_points('s', 0, 20_000, k=k, lam=0.5,
                                                      seed=9300 + i)[1][0], 12)
    Pm, sPm = P[:-1].reshape(len(mats), G), sP[:-1].reshape(len(mats), G)
    z = (Pm - C) / np.sqrt(sPm**2 + sC**2 + 1e-30)
    big = (np.abs(C) > 0.05)
    assert big.sum() >= 4
    assert np.abs(z[big]).max() < 4.5, z
    assert abs(P[-1] - ce0) < 4.5 * math.sqrt(sP[-1]**2 + sce0**2)
