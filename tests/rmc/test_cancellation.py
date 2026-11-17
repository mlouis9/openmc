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
- G8.10 (grouped cancellation at fission births is exact): parents fly once
  and their children (emission records: fission births and stationary
  elastic scatterings) fly again; the fission grandchildren, binned by
  emitted energy, have the same mean total weight with ordinary weights and
  with the grouped weights sum_j J_j / sum_j m_j q_j, over independent
  repetitions of both flights; and the grouped weights carry less absolute
  weight.
- G8.11 (net loss of the field's flux): the backward estimate of the net
  loss rate per source tag (absorption less the gain of multiplying
  scatterings; field_production with score='loss') equals the flights'
  collision estimate for a weighted sample of the field, and the flights'
  loss and scattered weight add up to the weight flown.
- G8.12 (forward rays): the first-collision rates of the field's emission
  from forward rays integrated analytically (field_first_flight_rates)
  equal the backward estimates of the production and the net loss, per
  source tag.
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


@pytest.mark.slow
def test_grouped_fission_cancellation_is_exact(lib, field_files):
    rmc, mesh = lib
    rng = np.random.default_rng(41)
    n_par, n1, n2, reps, k = 2000, 20_000, 40_000, 120, 0.7
    parents = _fuel_parents(rng, n_par, 2.0e6)
    parents['w'] = rng.choice([-1.0, 1.0], n_par) * (0.5 + rng.random(n_par))
    rep = canc.Representation(rmc, field_files, mesh)
    grouped = canc.GroupedFissionCancellation(rmc, rep, d_max=3.0, group_size=8)
    edges = np.array([0.0, 5e5, 1.5e6, 3e6, 2e7])
    diff, absw = [], []
    for i in range(reps):
        b = rmc.Bank(**parents)
        out, _, aux = rmc.propagate_aux(b, n1, k, seed=1000 + i, children=True)
        ch = out.arrays()
        out.free()
        b.free()
        bank = canc.child_records(ch, aux)
        gid, off, mem = grouped.groups(bank, np.random.default_rng(i))
        b = rmc.Bank(**{key: bank[key] for key in ('r', 'u', 'E', 'w', 'tag')})
        out, _, aux2 = rmc.propagate_aux(b, n2, k, seed=5000 + i, children=True)
        gc = out.arrays()
        out.free()
        b.free()
        new, info = grouped(bank, gc, aux2, gid, off, mem, n2, k)
        f = gc['tag'] == 1
        e = np.digitize(gc['E'][f], edges) - 1
        diff.append(np.bincount(e, weights=new['w'][f] - gc['w'][f], minlength=4))
        absw.append((info['W_eligible_before'], info['W_eligible_after'],
                     info['eligible'], info['fission_children']))
    diff = np.array(diff)
    absw = np.array(absw).mean(axis=0)
    mean, sem = diff.mean(axis=0), diff.std(axis=0, ddof=1) / math.sqrt(reps)
    scale = absw[0] / 4                      # eligible weight per energy bin
    print('grouped: paired difference / sem', mean / sem, ' sem/scale', sem / scale,
          ' eligible', absw[2], 'of', absw[3], ' |w|', absw[0], '->', absw[1])
    assert np.all(np.abs(mean) < 4.5 * sem), (mean, sem)
    assert np.all(sem < 0.2 * scale)         # the comparison has power
    assert absw[1] < 0.8 * absw[0]


@pytest.mark.slow
def test_field_loss_matches_flights(lib, field_files):
    rmc, _ = lib
    L, var = rmc.field_production(60_000, n_E=4, seed=17, cutoff=math.exp(-3), score='loss')
    rng = np.random.default_rng(5)
    bins = cs.Bins(rmc, field_files)
    for i, comp in enumerate('sf'):
        est = []
        for rep in range(6):
            r, u, E, w = cs.field_parents(rmc, bins, comp, 100_000, rng)
            b = rmc.Bank(r=r, u=u, E=E, w=w, tag=np.zeros(len(w), dtype=np.int32))
            out, st, _ = rmc.propagate_aux(b, 200_000, 1.0, seed=70 + rep, children=False)
            ch = out.arrays()
            out.free()
            b.free()
            loss = st['absorption'] - st['multiplied']
            est.append(loss)
            # Analog closure of the same flights: what is not lost scatters
            scattered = ch['w'][ch['tag'] == 0].sum()
            flown = w.sum()
            assert abs(scattered + loss - flown) < 0.02 * abs(flown)
        est = np.array(est)
        mean, sem = est.mean(), est.std(ddof=1) / math.sqrt(len(est))
        z = (L[i] - mean) / math.hypot(sem, math.sqrt(var[i]))
        assert abs(z) < 5, (comp, L[i], mean, sem)
        assert abs(L[i] / mean - 1) < 0.05


@pytest.mark.slow
def test_forward_rays_match_backward_estimates(lib):
    rmc, _ = lib
    Xb, vXb = rmc.field_production(200_000, n_E=4, seed=31, cutoff=math.exp(-3))
    Lb, vLb = rmc.field_production(200_000, n_E=4, seed=32, cutoff=math.exp(-3), score='loss')
    Xf, Lf, vXf, vLf = rmc.field_first_flight_rates(400_000, n_E=2, seed=33)
    for name, b, vb, f, vf in (('X', Xb, vXb, Xf, vXf), ('L', Lb, vLb, Lf, vLf)):
        z = (f - b) / np.sqrt(vb + vf)
        assert np.all(np.abs(z) < 5), (name, b, f, z)
        # the comparison resolves one per cent of the larger component
        assert math.sqrt(vb[0] + vf[0]) < 0.01 * b[0]


@pytest.mark.slow
def test_multigroup_rates_on_forward_rays_are_the_fixed_point(lib, field_files):
    """G8.13: scored with the multigroup cross sections of each energy's
    group, the forward rays give the field's own balance: its net loss is
    the fission emission mass <1, Q_f> and its production is k_MG times
    that."""
    import field as fld
    rmc, _ = lib
    k_mg = float(fld.FieldFile.load(field_files['field']).k_mg)
    X, L, vX, vL, Xm, Lm, _, _ = rmc.field_first_flight_rates(400_000, n_E=4, seed=41,
                                                              control=True)
    M_f = rmc.field_mass('f')
    # The multigroup scores' own spread is close to the CE one
    tol = 5 * math.sqrt(vL.sum()) + 2e-3 * M_f
    assert abs(Lm.sum() - M_f) < tol, (Lm.sum(), M_f)
    assert abs(Xm.sum() - k_mg * M_f) < 5 * math.sqrt(vX.sum()) + 2e-3 * k_mg * M_f


@pytest.mark.slow
def test_forward_residual_bins_reproduce_the_field(lib, field_files):
    """G8.14: the multigroup deposits of the forward rays, summed over the
    source tags, are the field's own emission by netting bin (tag s) and
    k_MG times it (tag f, as productions); the CE-minus-multigroup deposits
    are small against them."""
    import field as fld
    rmc, mesh = lib
    rep = canc.Representation(rmc, field_files, mesh)
    k_mg = float(fld.FieldFile.load(field_files['field']).k_mg)
    h = rep.n_bins() // 2
    F_s, F_f = -rep.low_term((1.0, 0.0)), -rep.low_term((0.0, 1.0))
    diff, low = rmc.forward_residual_bins(300_000, rep.zone, rep.Z, n_E=8, seed=51)
    L = low.sum(axis=0)
    assert L[:h].sum() == pytest.approx(F_s.sum(), rel=3e-3)
    assert L[h:].sum() == pytest.approx(k_mg * F_f.sum(), rel=3e-3)
    assert np.abs(L[:h] - F_s[:h]).max() < 0.02 * F_s[:h].max()
    assert np.abs(L[h:] - k_mg * F_f[h:]).max() < 0.02 * k_mg * F_f[h:].max()
    D = diff.sum(axis=0)
    assert np.abs(D[:h]).sum() < 0.1 * F_s.sum()
    assert abs(D[:h].sum()) < 0.01 * F_s.sum()
