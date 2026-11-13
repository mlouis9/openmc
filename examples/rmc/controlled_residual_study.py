"""Controlled residual estimators without consistent densities: a binned
study against an independent reference (graphite pincell).

    python controlled_residual_study.py --work rmc_study_work            # smoke
    python controlled_residual_study.py --work rmc_study_work --n 40000 \\
        --n-scan 10000 --ref-flights 4000000 --figures ../../docs/source/devguide/rmc_figures

Theory: docs/source/methods/rmc.rst, methods_rmc_sampler_control. Report:
docs/source/devguide/rmc_controlled_residual.rst.

The residual of the frozen field Q_a = a_s Q_s + a_f Q_f is estimated by
residual_sample() records with weights from residual_weights(). It is
binned three ways:

- energy: by material and 36 logarithmic outgoing-energy bins;
- space: by field region (the fuel and the graphite are each split into
  regions by the source-region mesh), shown against the region's centroid
  radius; the field integral uses geometric region volumes from a fine grid,
  since the field's sampled volumes V_A are noisy in the small regions cut
  by the pin surface;
- angle: by material, energy range, and the radial cosine mu_r = Omega .
  e_rho of the outgoing direction (e_rho the unit vector away from the pin
  axis).

Reference. Parents are drawn from each component of the field (Q_s alone,
Q_f alone) in Python: a uniform point in the box, a group in proportion to
the region's emission, and a log-uniform energy in the group, weighted by
4 pi V_box q / p. One forward flight of each, with OpenMC's collision
routines (rmc.propagate, Stage 5), gives first-collision production by tag.
By linearity the reference for any amplitudes is a_s Ref_s + a_f Ref_f,
minus the field a_t F_t, which is integrated exactly from the lifts. The
reference shares no code with the residual sampler beyond the field and
OpenMC's physics; its uncertainty comes from independent chunks.

Estimator classes (VARIANTS; methods_rmc_sampler_control):

- density route: elastic scattering (stationary target) and fission by
  their densities at the destination's outgoing state; every other channel
  by OpenMC's samplers, with or without the multigroup control on the
  density branch (c = 1) and the flux-difference control on the sampler
  route;
- density free: every channel on the sampler route (all_sampler), with K
  incident energies per direction (independent or common random numbers),
  the flux-difference control with M_II region samples, and the biased net
  assignment for comparison.

Parameter scans (density free, flux-difference control unless noted): K
with and without common random numbers, J, M_II, and the roulette cutoff
of the backward paths.

High term alone. The same records with b = 0 give the high term, whose
noise is what remains if the low term is removed exactly (in a tally, by
subtracting F; in the iteration, by cancelling the low-term particles
against the high-term ones).

Noise scaling. (1) Per bin, the per-sample standard deviation against the
reference residual and against the first-collision production (high term).
(2) An amplitude scan: the weights of the same records for a = b = (1,
lambda) and (lambda, 1) give estimates of residuals of different size, whose
reference is known by linearity; the binned noise is plotted against the
residual's norm. (3) Per channel, the noise of each channel's records
against that channel's production.

Outputs: controlled_residual_study.npz (every binned estimate and the
reference) and .json (summary tables); figures with --figures.
"""
import argparse
import json
import math
import time
from pathlib import Path

import h5py
import numpy as np

import openmc.lib

import pipeline
import problems
import residual_gates as gates
import estimator_study as es

CHANNELS = ('elastic', 'thermal_elastic', 'inelastic', 'fission')
KINDS = ('density', 'sampler', 'region')
E_EDGES = np.logspace(-5, math.log10(2e7), 37)
E_RANGES = np.array([1e-5, 1.0, 1e5, 2e7])
RANGE_NAMES = ('thermal (< 1 eV)', 'epithermal (1 eV - 100 keV)', 'fast (> 100 keV)')
MU_EDGES = np.linspace(-1.0, 1.0, 11)
MATERIAL_NAMES = ('fuel', 'graphite')
TAIL = dict(tail='roulette', cutoff=math.exp(-4))

FD = dict(sampler_control='region')
ALL = dict(all_sampler=True)

# name: (class, residual_sample options, c, lambda)
VARIANTS = {
    'density route, uncontrolled': (
        'density route', dict(directions='isotropic', n_directions=8), 0.0, 0.0),
    'density route, MG control': (
        'density route', dict(directions='isotropic', n_directions=4, control=True,
                              n_control_directions=4), 1.0, 0.0),
    'density route, MG control + flux difference': (
        'density route', dict(directions='isotropic', n_directions=4, control=True,
                              n_control_directions=4, region_samples=8, **FD), 1.0, 0.0),
    'density route, tandem + flux difference': (
        'density route', dict(directions='sweep', n_directions=8, control=True,
                              region_samples=8, **FD), 1.0, 1.0),
    'density free, uncontrolled, K=1': (
        'density free', dict(directions='isotropic', n_directions=8, **ALL), 0.0, 0.0),
    'density free, K=4': (
        'density free', dict(directions='isotropic', n_directions=4, n_energies=4,
                             **ALL), 0.0, 0.0),
    'density free, K=4 common numbers': (
        'density free', dict(directions='isotropic', n_directions=4, n_energies=4,
                             common_numbers=True, **ALL), 0.0, 0.0),
    'density free, flux difference, K=1': (
        'density free', dict(directions='isotropic', n_directions=8, region_samples=8,
                             **FD, **ALL), 0.0, 0.0),
    'density free, flux difference, K=4 common numbers': (
        'density free', dict(directions='isotropic', n_directions=4, n_energies=4,
                             common_numbers=True, region_samples=8, **FD, **ALL), 0.0, 0.0),
    'density free, flux difference, K=4 common numbers, net (biased)': (
        'density free', dict(directions='isotropic', n_directions=4, n_energies=4,
                             common_numbers=True, region_samples=8, net=True, **FD,
                             **ALL), 0.0, 0.0),
}

# Variants shown in the binned comparison figures
SHOWN = ['density route, MG control',
         'density route, MG control + flux difference',
         'density free, uncontrolled, K=1',
         'density free, flux difference, K=4 common numbers']

_BASE = dict(directions='isotropic', **ALL)
SCANS = {
    'K, independent': [(K, dict(n_directions=4, n_energies=K, region_samples=8, **FD))
                       for K in (1, 2, 4, 8, 16)],
    'K, common numbers': [(K, dict(n_directions=4, n_energies=K, common_numbers=True,
                                   region_samples=8, **FD)) for K in (1, 2, 4, 8, 16)],
    'K, uncontrolled': [(K, dict(n_directions=4, n_energies=K)) for K in (1, 2, 4, 8, 16)],
    'J': [(J, dict(n_directions=J, region_samples=8, **FD)) for J in (1, 2, 4, 8, 16)],
    'M_II': [(M, dict(n_directions=4, region_samples=M, **FD))
             for M in (1, 2, 8, 32, 128)],
    # Roulette cutoff exp(-x) of the backward paths
    'cutoff, uncontrolled': [(x, dict(n_directions=4, cutoff=math.exp(-x)))
                             for x in (2, 4, 8, 12)],
    'cutoff, flux difference (M_II = 128)': [
        (x, dict(n_directions=4, cutoff=math.exp(-x), region_samples=128, **FD))
        for x in (2, 4, 8, 12)],
}

AMPLITUDES = (0.0, 0.5, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0, 3.0)


# ---------------------------------------------------------------------------
# Geometry and binning
# ---------------------------------------------------------------------------

class Bins:
    """The three binnings and their field integrals."""

    def __init__(self, rmc, files):
        with h5py.File(files['field']) as f:
            self.region_material_id = f['regions/material_id'][()]
        with h5py.File(files['volumes']) as f:
            self.V = f['V_A'][()]
        self.R = len(self.V)
        self.region_mat = np.where(self.region_material_id == problems.MAT_FUEL, 0, 1)
        # Region centroids and geometric volumes from a grid of points (the
        # problem is uniform in z). The field's own volumes V_A come from
        # point sampling and carry a few per cent of noise in the small
        # regions cut by the pin surface; the estimators and the reference
        # both see the geometry, so the exact field integrals use these.
        n = 4001
        x = np.linspace(-problems.PITCH / 2, problems.PITCH / 2, n + 1)
        xc = 0.5 * (x[:-1] + x[1:])
        X, Y = np.meshgrid(xc, xc, indexing='ij')
        pts = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])
        reg = rmc.field_locate(pts)
        ok = reg >= 0
        cnt = np.bincount(reg[ok], minlength=self.R)
        cx = np.bincount(reg[ok], weights=pts[ok, 0], minlength=self.R) / np.maximum(cnt, 1)
        cy = np.bincount(reg[ok], weights=pts[ok, 1], minlength=self.R) / np.maximum(cnt, 1)
        self.centroid_radius = np.hypot(cx, cy)
        self.V_geom = cnt * (problems.PITCH / n)**2 * problems.HEIGHT
        self.sizes = {'energy': 2 * (len(E_EDGES) - 1), 'space': self.R,
                      'angle': 2 * 3 * (len(MU_EDGES) - 1)}

    def index(self, rmc, r, u, E):
        """Bin of each state in every binning (-1 outside)."""
        reg = rmc.field_locate(r)
        rho = np.hypot(r[:, 0], r[:, 1])
        mat = np.where(rho < problems.R_FUEL, 0, 1)
        nE = len(E_EDGES) - 1
        b = np.searchsorted(E_EDGES, E, side='right') - 1
        e_ok = (b >= 0) & (b < nE)
        out = {'energy': np.where(e_ok, mat * nE + b, -1),
               'space': np.where(e_ok & (reg >= 0), reg, -1)}
        er = np.searchsorted(E_RANGES, E, side='right') - 1
        mu = (u[:, 0] * r[:, 0] + u[:, 1] * r[:, 1]) / np.maximum(rho, 1e-12)
        nm = len(MU_EDGES) - 1
        bm = np.clip(np.searchsorted(MU_EDGES, mu, side='right') - 1, 0, nm - 1)
        a_ok = (er >= 0) & (er < 3)
        out['angle'] = np.where(a_ok, (mat * 3 + er) * nm + bm, -1)
        return out

    def field(self, rmc, files, tag):
        """Exact field integral F_t in every binning."""
        regions = [np.flatnonzero(self.region_mat == m) for m in (0, 1)]
        V = self.V_geom
        F = {'energy': gates.field_integral(rmc, files, regions, E_EDGES, tag,
                                            volumes=V).ravel()}
        per_region = gates.field_integral(rmc, files, [[r] for r in range(self.R)],
                                          np.array([E_EDGES[0], E_EDGES[-1]]), tag,
                                          volumes=V)
        F['space'] = per_region[:, 0]
        by_range = gates.field_integral(rmc, files, regions, E_RANGES, tag,
                                        volumes=V)                            # [2, 3]
        dmu = np.diff(MU_EDGES) / 2.0
        F['angle'] = (by_range[:, :, None] * dmu[None, None, :]).ravel()
        return F


def binned(idx, w, sample, N, size):
    """Sum of weights per bin and its standard error, each sample (a
    destination with all its records) one independent draw."""
    ok = idx >= 0
    key = sample[ok].astype(np.int64) * size + idx[ok]
    uk, inv = np.unique(key, return_inverse=True)
    tot = np.bincount(inv, weights=w[ok])                 # per (sample, bin)
    b = uk % size
    s1 = np.bincount(b, weights=tot, minlength=size)
    s2 = np.bincount(b, weights=(N * tot)**2, minlength=size)
    var = np.maximum(s2 / N - s1**2, 0.0) / N
    return s1, np.sqrt(var)


# ---------------------------------------------------------------------------
# Reference
# ---------------------------------------------------------------------------

def field_parents(rmc, bins, component, n, rng):
    """n weighted parents from one component of the field."""
    lo = np.array([-problems.PITCH / 2, -problems.PITCH / 2, -problems.HEIGHT / 2])
    hi = -lo
    V_box = float(np.prod(hi - lo))
    r = lo + (hi - lo) * rng.random((n, 3))
    reg = rmc.field_locate(r)
    groups = rmc.field_groups()
    G = len(groups) - 1
    Q = np.array([rmc.field_emission(R, component) for R in range(bins.R)])  # [R, G]
    keep = reg >= 0
    r, reg = r[keep], reg[keep]
    Qr = Q[reg]
    tot = Qr.sum(axis=1)
    keep = tot > 0
    r, reg, Qr, tot = r[keep], reg[keep], Qr[keep], tot[keep]
    cdf = np.cumsum(Qr, axis=1) / tot[:, None]
    g = (rng.random(len(reg))[:, None] > cdf).sum(axis=1)
    g = np.minimum(g, G - 1)
    elo = np.maximum(groups[g], E_EDGES[0])
    ehi = groups[g + 1]
    E = elo * np.exp(np.log(ehi / elo) * rng.random(len(reg)))
    p = Qr[np.arange(len(reg)), g] / tot / (E * np.log(ehi / elo))
    q = 4 * math.pi * rmc.field_evaluate(reg, E, component)
    w = V_box * q / p / n
    mu = 2 * rng.random(len(reg)) - 1
    ph = 2 * math.pi * rng.random(len(reg))
    st = np.sqrt(1 - mu**2)
    u = np.column_stack([st * np.cos(ph), st * np.sin(ph), mu])
    return r, u, E, w


def reference(rmc, bins, k, flights, chunks, seed=1):
    """{component: {tag: {binning: (mean, sem)}}} of first-collision
    production, and the field masses for a check."""
    rng = np.random.default_rng(seed)
    out, mass = {}, {}
    per = max(flights // chunks, 1)
    for comp in 'sf':
        acc = {t: {b: [] for b in bins.sizes} for t in 'sf'}
        masses = []
        for c in range(chunks):
            r, u, E, w = field_parents(rmc, bins, comp, per, rng)
            masses.append(w.sum())
            bank = rmc.Bank(r=r, u=u, E=E, w=w, tag=['s'] * len(w))
            children, st, _ = rmc.propagate(bank, per, k, seed=seed + 1000 * c + 17)
            ch = children.arrays()
            bank.free()
            children.free()
            idx = bins.index(rmc, ch['r'], ch['u'], ch['E'])
            for it, t in enumerate('sf'):
                m = ch['tag'] == it
                for b, size in bins.sizes.items():
                    i = idx[b][m]
                    ok = i >= 0
                    acc[t][b].append(np.bincount(i[ok], weights=ch['w'][m][ok],
                                                 minlength=size))
            print(f'  reference {comp}: chunk {c + 1}/{chunks}, {per} flights', flush=True)
        out[comp] = {t: {b: (np.mean(v, axis=0), np.std(v, axis=0, ddof=1) / math.sqrt(chunks))
                         for b, v in acc[t].items()} for t in 'sf'}
        mass[comp] = (float(np.mean(masses)), float(rmc.field_mass(comp)))
    return out, mass


def reference_residual(ref, F, t, a, binning):
    """Reference residual of tag t for amplitudes a = b, and the high term."""
    hi = a[0] * ref['s'][t][binning][0] + a[1] * ref['f'][t][binning][0]
    sd = np.hypot(a[0] * ref['s'][t][binning][1], a[1] * ref['f'][t][binning][1])
    return hi - a['sf'.index(t)] * F[t][binning], sd, hi


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------

def run_variant(rmc, bins, tag, N, opts, seed):
    t0 = time.perf_counter()
    rec = rmc.residual_sample(tag, N, seed=seed, E_range=(1e-5, 2e7), **{**TAIL, **opts})
    dt = time.perf_counter() - t0
    dens = rec['sampler'] == 0
    rec['sample'] = np.cumsum(np.r_[0, dens[:-1]])
    rec['bins'] = bins.index(rmc, rec['r'], rec['u'], rec['E'])
    rec['time'] = dt
    return rec


def binned_all(bins, rec, w, N, mask=None):
    out = {}
    ww = w if mask is None else np.where(mask, w, 0.0)
    for b, size in bins.sizes.items():
        out[b] = binned(rec['bins'][b], ww, rec['sample'], N, size)
    return out


def summary(est, ref_r, ref_sd, N):
    e, s = est
    err = np.hypot(s, ref_sd)
    # A bin is compared only where both sides have a standard error: the
    # reference has events there and the estimator has more than one sample
    live = (s > 0) & (ref_sd > 0)
    z = (e[live] - ref_r[live]) / err[live]
    return dict(chi2=float(np.sum(z**2)), dof=int(live.sum()),
                max_abs_z=float(np.abs(z).max()) if z.size else 0.0,
                per_sample_variance=float(N * np.sum(s**2)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--work', default='rmc_study_work')
    ap.add_argument('--n', type=int, default=4000, help='samples per tag, main variants')
    ap.add_argument('--n-scan', type=int, default=2000, help='samples per tag, scans')
    ap.add_argument('--ref-flights', type=int, default=400_000,
                    help='reference flights per field component')
    ap.add_argument('--ref-chunks', type=int, default=8)
    ap.add_argument('--out', default='controlled_residual_study')
    ap.add_argument('--figures', default=None, help='directory for the figures')
    args = ap.parse_args()

    w, files, k = es._files(args.work)
    import openmc.lib.rmc as rmc
    res = dict(k=k, n=args.n, n_scan=args.n_scan, ref_flights=args.ref_flights,
               variants={}, scans={}, amplitude={}, channels={})
    arrays = {}
    with pipeline.sweep_session(w.model, w.mesh, files['mgxs'], files['field'],
                                files['work'] / 'controlled_study', rays=500):
        rmc.field_load(files['field'], files['lifts'], files['volumes'])
        rmc.sweep_init()
        rmc.sweep_iterate(3)
        rmc.sweep_field(batch=4)
        bins = Bins(rmc, files)
        F = {t: bins.field(rmc, files, t) for t in 'sf'}
        cache = Path(args.out + '_reference.npz')
        key = f'{args.ref_flights}/{args.ref_chunks}'
        if cache.exists() and str(np.load(cache)['key']) == key:
            c = np.load(cache)
            ref = {cp: {t: {b: tuple(c[f'{cp}/{t}/{b}']) for b in bins.sizes} for t in 'sf'}
                   for cp in 'sf'}
            mass = json.loads(str(c['mass']))
        else:
            print('reference flights', flush=True)
            ref, mass = reference(rmc, bins, k, args.ref_flights, args.ref_chunks)
            np.savez(cache, key=key, mass=json.dumps(mass),
                     **{f'{cp}/{t}/{b}': np.array(ref[cp][t][b]) for cp in 'sf'
                        for t in 'sf' for b in bins.sizes})
        res['field_mass_check'] = mass
        for c in 'sf':
            for t in 'sf':
                for b in bins.sizes:
                    arrays[f'ref/{c}/{t}/{b}'] = np.array(ref[c][t][b])
        for t in 'sf':
            for b in bins.sizes:
                arrays[f'F/{t}/{b}'] = F[t][b]
        arrays['centroid_radius'] = bins.centroid_radius
        arrays['region_mat'] = bins.region_mat
        arrays['V'] = bins.V_geom
        arrays['V_A'] = bins.V

        # Main variants
        one = (1.0, 1.0)
        for name, (cls, opts, c, lam) in VARIANTS.items():
            for it, t in enumerate('sf'):
                rec = run_variant(rmc, bins, t, args.n, opts, seed=101 + 13 * it)
                wts = rmc.residual_weights(rec, args.n, one, one, k, c, lam)
                est = binned_all(bins, rec, wts, args.n)
                entry = dict(cls=cls, c=c, lam=lam, time=rec['time'],
                             records=int(len(wts)), binnings={})
                for b in bins.sizes:
                    rr, sd, hi = reference_residual(ref, F, t, one, b)
                    # The tandem's low term is the sweep's emission, not the
                    # field: compare its residual with lambda = 0 weights
                    if lam > 0:
                        e0 = binned_all(bins, rec, rmc.residual_weights(
                            rec, args.n, one, one, k, c, 0.0), args.n)[b]
                    else:
                        e0 = est[b]
                    entry['binnings'][b] = summary(e0, rr, sd, args.n)
                    arrays[f'est/{name}/{t}/{b}'] = np.array(est[b])
                    arrays[f'est0/{name}/{t}/{b}'] = np.array(e0)
                # The high term alone (b = 0): what remains if the low term
                # is removed exactly, e.g. by cancellation against it
                wh = rmc.residual_weights(rec, args.n, one, (0, 0), k, c, 0.0)
                hi_est = binned_all(bins, rec, wh, args.n)
                for b in bins.sizes:
                    arrays[f'high/{name}/{t}/{b}'] = np.array(hi_est[b])
                entry['high_only_variance'] = {
                    b: float(args.n * np.sum(hi_est[b][1]**2)) for b in bins.sizes}
                # Decomposition by record kind and channel (density free and
                # density route alike)
                dec = {}
                for ik, kn in enumerate(KINDS):
                    m = rec['kind'] == ik
                    if m.any():
                        dec[f'kind:{kn}'] = float(args.n * np.sum(
                            binned_all(bins, rec, wts, args.n, m)['energy'][1]**2))
                for ic, cn in enumerate(CHANNELS):
                    m = rec['channel'] == ic
                    if m.any():
                        dec[f'channel:{cn}'] = float(args.n * np.sum(
                            binned_all(bins, rec, wts, args.n, m)['energy'][1]**2))
                entry['decomposition'] = dec
                res['variants'][f'{name} | {t}'] = entry
                print(f'{t}  {name:62s} t {rec["time"]:.1f}s  '
                      + '  '.join(f'{b}: chi2 {v["chi2"]:.0f}/{v["dof"]} '
                                  f'var {v["per_sample_variance"]:.3e}'
                                  for b, v in entry['binnings'].items()), flush=True)

                # Amplitude scan (noise against the residual's size)
                if name in SHOWN or 'uncontrolled' in name:
                    amp = []
                    for which in (0, 1):
                        for x in AMPLITUDES:
                            a = (x, 1.0) if which == 0 else (1.0, x)
                            wa = rmc.residual_weights(rec, args.n, a, a, k, c, 0.0)
                            e, s = binned_all(bins, rec, wa, args.n)['energy']
                            rr, sd, hi = reference_residual(ref, F, t, a, 'energy')
                            amp.append(dict(which='a_s' if which == 0 else 'a_f', x=x,
                                            residual_norm=float(np.linalg.norm(rr)),
                                            high_norm=float(np.linalg.norm(hi)),
                                            noise=float(math.sqrt(args.n) * np.linalg.norm(s)),
                                            error=float(np.linalg.norm(e - rr))))
                    res['amplitude'][f'{name} | {t}'] = amp

                # Per-channel production (the sampled particles without the
                # control) and noise (the channel's records with it, b = 0)
                if cls == 'density free' and 'net' not in name:
                    wh = rmc.residual_weights(rec, args.n, one, (0, 0), k, c, 0.0)
                    wp = rmc.residual_weights({**rec, 'Z': np.zeros_like(rec['Z'])},
                                              args.n, one, (0, 0), k, c, 0.0)
                    ch = {}
                    for ic, cn in enumerate(CHANNELS):
                        m = rec['channel'] == ic
                        if not m.any():
                            continue
                        e1, s1 = binned_all(bins, rec, wp, args.n, m & (rec['kind'] == 1))['energy']
                        e2, s2 = binned_all(bins, rec, wh, args.n, m)['energy']
                        ch[cn] = dict(production=e1.tolist(),
                                      noise=(math.sqrt(args.n) * s2).tolist())
                    res['channels'][f'{name} | {t}'] = ch

        # Parameter scans (energy binning)
        for scan, points in SCANS.items():
            rows = []
            for x, opts in points:
                row = dict(x=x)
                for it, t in enumerate('sf'):
                    rec = run_variant(rmc, bins, t, args.n_scan, {**_BASE, **opts},
                                      seed=501 + 13 * it)
                    wts = rmc.residual_weights(rec, args.n_scan, one, one, k, 0.0, 0.0)
                    est = binned_all(bins, rec, wts, args.n_scan)
                    rr, sd, _ = reference_residual(ref, F, t, one, 'energy')
                    s = summary(est['energy'], rr, sd, args.n_scan)
                    wh = rmc.residual_weights(rec, args.n_scan, one, (0, 0), k, 0.0, 0.0)
                    _, sh = binned_all(bins, rec, wh, args.n_scan)['energy']
                    row[t] = dict(time_per_sample=rec['time'] / args.n_scan,
                                  high_only_variance=float(args.n_scan * np.sum(sh**2)), **s)
                rows.append(row)
                print(f'scan {scan} = {x}: ' + '  '.join(
                    f'{t} var {row[t]["per_sample_variance"]:.3e} chi2 {row[t]["chi2"]:.0f}'
                    f'/{row[t]["dof"]}' for t in 'sf'), flush=True)
            res['scans'][scan] = rows
        rmc.sweep_finalize()
        rmc.field_free()

    np.savez_compressed(args.out + '.npz', **arrays)
    Path(args.out + '.json').write_text(json.dumps(res, indent=1))
    print('wrote', args.out + '.npz', args.out + '.json')
    if args.figures:
        import controlled_residual_figures as figs
        figs.make_all(args.out, args.figures)


if __name__ == '__main__':
    main()
