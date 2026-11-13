"""The multigroup control against random ray in the graphite pincell (gate
G3.6, statistical part).

Uniform points in a fuel and a moderator region and isotropic directions
with the roulette tail: the mean MG production of the field Q_s + Q_f (with
1/k_MG for fission) equals Q_{t,R,g} within the Monte Carlo error of the
average or the random ray solution's own error (2 %). The deterministic
part of the gate is in test_mg_control_infinite.py.
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
def pin(pincell, field_files):
    import openmc.lib.rmc as rmc
    model, mesh = pincell
    with pipeline.ce_session(model, mesh, field_files['work'] / 'ce_mg'):
        rmc.field_load(field_files['field'], field_files['lifts'], field_files['volumes'])
        with h5py.File(field_files['field']) as h:
            k = float(h.attrs['k_mg'])
        yield dict(rmc=rmc, k=k)
        rmc.field_free()


def _points_in_region(rmc, region, n, rng):
    lo = -np.array([problems.PITCH / 2, problems.PITCH / 2, problems.HEIGHT / 2])
    out = []
    while len(out) < n:
        x = rng.uniform(lo, -lo, size=(4 * n, 3))
        out.extend(x[rmc.field_locate(x) == region])
    return np.array(out[:n])


@pytest.mark.slow
@pytest.mark.parametrize('where', ['fuel', 'moderator'])
def test_mg_production_pincell(pin, where):
    rmc = pin['rmc']
    point = {'fuel': [0.05, 0.05, 0.0], 'moderator': [0.58, 0.58, 0.0]}[where]
    region = int(rmc.field_locate(np.array(point))[0])
    rng = np.random.default_rng(7)
    n = 400
    pts = _points_in_region(rmc, region, n, rng)
    dirs = rng.normal(size=(n, 3))
    dirs /= np.linalg.norm(dirs, axis=1)[:, None]
    for tag in ('s', 'f'):
        Q = rmc.field_emission(region, tag)
        if Q.max() == 0.0:
            continue
        x = np.array([_production_by_group(rmc, p, u, region, tag, pin['k'],
                                           seed=i + 1)
                      for i, (p, u) in enumerate(zip(pts, dirs))])
        mean = x.mean(axis=0)
        sem = x.std(axis=0, ddof=1) / math.sqrt(n)
        ok = Q > 1e-3 * Q.max()
        err = np.abs(mean - Q)[ok]
        allowed = np.maximum(5 * sem[ok], 0.02 * Q[ok])
        assert np.all(err <= allowed), (tag, mean[ok] / Q[ok] - 1, sem[ok] / Q[ok])
