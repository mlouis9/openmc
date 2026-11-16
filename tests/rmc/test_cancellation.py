"""Sign cancellation and the correction tally on the graphite pincell
(gates G8.0-G8.4; examples/rmc/cancellation.py).

- G8.0 (near kernel): fission children of isotropic parents at one point
  and energy, with flight paths below d_max, are distributed in the fuel
  like Sigma_f(E) K(x | x_p, E), K the image-summed near kernel: the
  counts in shells around the parent match the kernel integrated by Monte
  Carlo over the same shells (chi^2 with the normalization fitted).
- G8.1 (regional cancellation is exact): the children of a bank of tag-f
  parents, binned by fuel ring and energy, have the same mean with and
  without regional cancellation, over independent flights; and the pool
  carries weight.
- G8.2 (destination pooling is exact): the fission residual with every
  channel on the sampler route, binned by region and energy, has the same
  mean with and without pooling.
- G8.3 (binned netting): with the low term exact, the netted bank's bin
  totals equal the input's plus the closed-form low term, in expectation;
  the absolute weight falls.
- G8.4 (field part of the correction tally): the field's CE fission rate by
  fuel region from backward traces equals the collision estimate from
  forward flights of a weighted sample of the field.
- G8.5 (history flights): a flight that continues through its scatterings
  ends in capture, fission, or leakage and banks only fission children;
  their total weight equals the fission weight that collision-by-collision
  flights of the same parents produce over all generations.
"""
import math

import numpy as np
import pytest

import openmc.lib

import cancellation as canc
import pipeline
import problems
import controlled_residual_study as cs


@pytest.fixture(scope='module')
def lib(pincell, field_files):
    import openmc.lib.rmc as rmc
    model, mesh = pincell
    with pipeline.ce_session(model, mesh, field_files['work'] / 'ce_cancellation'):
        rmc.field_load(field_files['field'], field_files['lifts'], field_files['volumes'])
        yield rmc, mesh
        rmc.field_free()


def _iso(rng, n):
    mu = 2 * rng.random(n) - 1
    ph = 2 * np.pi * rng.random(n)
    st = np.sqrt(1 - mu**2)
    return np.column_stack([st * np.cos(ph), st * np.sin(ph), mu])


def _fuel_parents(rng, n, E):
    rr = problems.R_FUEL * np.sqrt(rng.random(n))
    th = 2 * np.pi * rng.random(n)
    z = problems.HEIGHT * (rng.random(n) - 0.5)
    r = np.column_stack([rr * np.cos(th), rr * np.sin(th), z])
    return dict(r=r, u=_iso(rng, n), E=np.full(n, E), w=np.ones(n),
                tag=np.ones(n, dtype=np.int32))


@pytest.mark.slow
def test_near_kernel_matches_flights(lib):
    rmc, _ = lib
    rng = np.random.default_rng(1)
    xp = np.array([0.11, -0.07, 0.23])
    E, d_max, n = 2.0e6, 1.5, 400_000
    parents = dict(r=np.tile(xp, (4000, 1)), u=_iso(rng, 4000), E=np.full(4000, E),
                   w=np.ones(4000), tag=np.ones(4000, dtype=np.int32))
    b = rmc.Bank(**parents)
    out, _, aux = rmc.propagate_aux(b, n, 1.0, seed=3)
    ch = out.arrays()
    out.free()
    b.free()
    sel = (ch['tag'] == 1) & (aux['path'] < d_max)
    # Shells in the fuel by distance to the parent (direct image only)
    edges = np.array([0.0, 0.1, 0.2, 0.3, 0.45, 0.6])
    d = np.linalg.norm(ch['r'][sel] - xp, axis=1)
    counts, _ = np.histogram(d, edges)
    # Monte Carlo integral of K over the same shells within the fuel
    m = 200_000
    pts = _fuel_parents(rng, m, E)['r']
    K = rmc.near_kernel(pts, np.tile(xp, (m, 1)), np.full(m, E), d_max)
    V = math.pi * problems.R_FUEL**2 * problems.HEIGHT
    dp = np.linalg.norm(pts - xp, axis=1)
    expect = np.array([V * np.mean(K * ((dp >= lo) & (dp < hi)))
                       for lo, hi in zip(edges[:-1], edges[1:])])
    scale = counts.sum() / expect.sum()
    z = (counts - scale * expect) / np.sqrt(np.maximum(counts, 1))
    assert np.all(np.abs(z) < 5), (counts, scale * expect)
    assert np.sum(z**2) < len(z) - 1 + 5 * math.sqrt(2 * len(z))


def _bin_fuel(bank, E_edges):
    rho = np.hypot(bank['r'][:, 0], bank['r'][:, 1])
    ring = np.minimum((rho / problems.R_FUEL * 3).astype(int), 3)
    e = np.searchsorted(E_edges, bank['E'], side='right') - 1
    ok = (e >= 0) & (e < len(E_edges) - 1) & (bank['tag'] == 1)
    return np.bincount(ring[ok] * (len(E_edges) - 1) + e[ok], weights=bank['w'][ok],
                       minlength=4 * (len(E_edges) - 1))


@pytest.mark.slow
def test_regional_cancellation_is_exact(lib):
    rmc, _ = lib
    E_edges = np.array([1e3, 1e5, 5e5, 1e6, 2e6, 5e6, 2e7])
    reg = canc.RegionalCancellation(rmc, d_max=3.0)
    plain, cancelled, pooled = [], [], []
    for rep in range(12):
        rng = np.random.default_rng(100 + rep)
        par = _fuel_parents(rng, 3000, 1.5e6)
        par['w'] = np.where(rng.random(3000) < 0.5, 1.0, -1.0)
        b = rmc.Bank(**par)
        out, _, aux = rmc.propagate_aux(b, 60_000, 1.0, seed=7 + rep)
        ch = out.arrays()
        out.free()
        b.free()
        res, info = reg(par, ch, aux, np.random.default_rng(rep))
        plain.append(_bin_fuel(ch, E_edges))
        cancelled.append(_bin_fuel(res, E_edges))
        pooled.append(info['eligible'])
    plain, cancelled = np.array(plain), np.array(cancelled)
    diff = cancelled - plain
    sem = diff.std(axis=0, ddof=1) / math.sqrt(len(diff))
    live = sem > 0
    z = diff.mean(axis=0)[live] / sem[live]
    assert np.all(np.abs(z) < 5), z
    assert min(pooled) > 0


@pytest.mark.slow
def test_destination_pooling_is_exact(lib, field_files):
    rmc, _ = lib
    pool = canc.DestinationPooling(rmc, field_files)
    E_edges = np.array([1e3, 1e5, 5e5, 1e6, 2e6, 5e6, 2e7])
    plain, pooled = [], []
    for rep in range(10):
        N = 400
        rec = rmc.residual_sample('f', N, seed=50 + rep, all_sampler=True, n_energies=2,
                                  n_directions=4, tail='roulette', cutoff=math.exp(-4))
        w = rmc.residual_weights(rec, N, (1, 1), (1, 1), 0.65, 0.0, 0.0)
        bank = dict(r=rec['r'], u=rec['u'], E=rec['E'], w=w,
                    tag=rec['tag'].astype(np.int32))
        out, info = pool(rec, w, N, 1.0, np.random.default_rng(rep))
        plain.append(_bin_fuel(bank, E_edges))
        pooled.append(_bin_fuel(out, E_edges))
        assert info['W_out'] < info['W_in']
    plain, pooled = np.array(plain), np.array(pooled)
    diff = pooled - plain
    sem = diff.std(axis=0, ddof=1) / math.sqrt(len(diff))
    live = sem > 0
    z = diff.mean(axis=0)[live] / sem[live]
    assert np.all(np.abs(z) < 5), z


@pytest.mark.slow
def test_binned_netting_exact_low(lib, field_files):
    rmc, mesh = lib
    rep = canc.Representation(rmc, field_files, mesh)
    net = canc.BinnedNetting(rep, 20_000, exact_low=True)
    rng = np.random.default_rng(5)
    n = 30_000
    pts = rng.uniform(-np.array([problems.PITCH / 2, problems.PITCH / 2,
                                 problems.HEIGHT / 2]),
                      np.array([problems.PITCH / 2, problems.PITCH / 2,
                                problems.HEIGHT / 2]), size=(n, 3))
    bank = dict(r=pts, u=_iso(rng, n), E=10**rng.uniform(-3, 7, n),
                w=rng.uniform(0.0, 2e-3, n), tag=(rng.random(n) < 0.3).astype(np.int32))
    want = np.bincount(rep.bins(bank), weights=bank['w'], minlength=rep.n_bins()) + \
        rep.low_term((1e-4, 1e-4))
    totals = []
    for s in range(8):
        out, info = net(bank, (1e-4, 1e-4), seed=s)
        totals.append(np.bincount(rep.bins(out), weights=out['w'], minlength=rep.n_bins()))
        assert info['W_after'] <= info['W_before'] + np.abs(rep.low_term((1e-4, 1e-4))).sum()
    totals = np.array(totals)
    # Each bin holds n_b particles of weight m_b / E[n_b], with n_b the floor or
    # the ceiling of E[n_b]: every resampled particle must land in its own bin,
    # and the bin's mean over seeds must be within the binomial error of m_b
    expect = net.n * np.abs(want) / np.abs(want).sum()
    live = expect > 0
    unit = want[live] / expect[live]
    count = totals[:, live] / unit
    assert np.allclose(count, np.round(count), atol=1e-6)
    lo = np.floor(expect[live])
    assert np.all((np.round(count) == lo) | (np.round(count) == lo + 1))
    p_up = expect[live] - lo
    sd = np.abs(unit) * np.sqrt(p_up * (1 - p_up) / len(totals))
    assert np.all(np.abs(totals[:, live].mean(axis=0) - want[live]) <= 5 * sd + 1e-12)
    # The total is preserved in expectation
    assert abs(totals.sum(axis=1).mean() - want.sum()) < 5 * totals.sum(axis=1).std() + 1e-9


@pytest.mark.slow
def test_field_tally_matches_flights(lib, field_files):
    rmc, mesh = lib
    k = 0.65
    f_back, _, sq = rmc.field_tally(200_000, n_E=4, seed=11)
    X = f_back.sum(axis=1)
    n_pts = 200_000
    var = (sq.sum(axis=1) / n_pts - X**2) / n_pts
    # Forward: flights from a weighted sample of the field (both components)
    rng = np.random.default_rng(2)
    bins = cs.Bins(rmc, field_files)
    fwd = []
    for rep in range(6):
        parts = []
        for comp in 'sf':
            r, u, E, w = cs.field_parents(rmc, bins, comp, 200_000, rng)
            parts.append(dict(r=r, u=u, E=E, w=w, tag=np.zeros(len(w), dtype=np.int32)))
        bank = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
        b = rmc.Bank(**bank)
        out, _, aux = rmc.propagate_aux(b, 400_000, k, seed=30 + rep, children=False)
        out.free()
        b.free()
        fwd.append(aux['region_fission'])
    fwd = np.array(fwd)
    mean, sem = fwd.mean(axis=0), fwd.std(axis=0, ddof=1) / math.sqrt(len(fwd))
    fuel = X > 0
    z = (X[fuel] - mean[fuel]) / np.sqrt(var[fuel] + sem[fuel]**2)
    assert np.all(np.abs(z) < 5), z
    assert abs(X.sum() / mean.sum() - 1) < 0.02


@pytest.mark.slow
def test_history_flights_match_collision_generations(lib):
    rmc, _ = lib
    n, k = 20_000, 0.65
    parents = _fuel_parents(np.random.default_rng(11), 2000, 1.0e6)

    def fission_weight(arr):
        return float(arr['w'][arr['tag'] == 1].sum())

    hist, coll = [], []
    for rep in range(4):
        # History flights: one generation, fission children only
        b = rmc.Bank(**parents)
        out, st, _ = rmc.propagate_aux(b, n, k, seed=100 + rep, children=False,
                                       region_tally=False, history=True)
        arr = out.arrays()
        out.free()
        b.free()
        assert np.all(arr['tag'] == 1)
        assert st['captured'] + st['fissions'] + st['leaked'] == st['flights'] == n
        assert st['collisions'] > 10 * n
        hist.append(fission_weight(arr))

        # Collision flights: fission children summed over generations, the
        # scattering children carried on until their weight is negligible
        bank, total = parents, 0.0
        for g in range(2000):
            b = rmc.Bank(**bank)
            out, st, _ = rmc.propagate_aux(b, n, k, seed=10_000 * rep + g,
                                           children=False, region_tally=False)
            arr = out.arrays()
            out.free()
            b.free()
            total += fission_weight(arr)
            s_ = arr['tag'] == 0
            if arr['w'][s_].sum() < 1e-4 * len(parents['w']):
                break
            bank = {key: v[s_] for key, v in arr.items()}
        coll.append(total)
    hist, coll = np.array(hist), np.array(coll)
    sem = math.hypot(hist.std(ddof=1), coll.std(ddof=1)) / 2
    assert abs(hist.mean() - coll.mean()) < 5 * sem, (hist, coll)


@pytest.mark.slow
def test_field_production_matches_field_tally(lib):
    """G8.9: the field's CE nu-fission production per source tag from points
    drawn from its lifted fission rate (field_production) equals the sum over
    regions of the uniform-point field tally, and is less noisy (much less
    with lifts that resolve the fission resonances)."""
    rmc, _ = lib
    X, var = rmc.field_production(20_000, 4, seed=3)
    ref = np.array([rmc.field_tally(20_000, 4, seed=10 + i)[1].sum(axis=0)
                    for i in range(6)])
    mean, sem = ref.mean(axis=0), ref.std(axis=0, ddof=1) / math.sqrt(len(ref))
    assert np.all(np.abs(X - mean) < 5 * np.hypot(sem, np.sqrt(var))), (X, mean, sem)
    assert np.sqrt(var.sum()) < 0.5 * ref.sum(axis=1).std(ddof=1)
