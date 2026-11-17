"""Partners of the continuous forward residual (rmc.forward_points) on the
graphite pincell: ray-wise mismatch, noise by part, and the band partner's
closure.

    python forward_partner_study.py rho --partner lanes --b-m 16x1 32x2 --start src
    python forward_partner_study.py noise --partner bands --b-m 128x16 --parts s:0 s:8 f:7
    python forward_partner_study.py closure --b-m 32x16 --sweeps 8 --rays 20

Partners (forward_points parts):
  0  multigroup partner;   3  the CE law applied to the multigroup flux;
  5  RESOLVE lanes (resolve_basis.build) or bands (build_subgroup), with the
     CE law on the decoded flux, and its P1/P3/P7 truncations as diagnostics;
  7  the same, banked, without the moment diagnostics;
  8  part 7 (tag s) with a third proposal component drawn from the band
     library of slot 1 (the same bands as the partner here).

The importance is residual_efficiency.importance_table (field region x 60
energy bins); the bins are the control variate's (cancellation.Representation).
Noise is the standard deviation of the importance-weighted total times the
root of the CPU seconds that produced it, split into the bins' part and the
within-bin part.
"""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

FINE = [200, 200, 200, 200, 200, 4000, 800, 200]
NUCLIDES = ['U234', 'U235', 'U236', 'U238', 'O16', 'O17', 'C0']


def seed(i, label):
    return int.from_bytes(hashlib.sha256(f'{label}/{i}'.encode()).digest()[:8], 'little') >> 1


def setup(rmc, files):
    import h5py
    import openmc.lib
    import resolve_basis as rb
    edges = np.array(rmc.field_groups())
    edges[0] = 1e-5
    mats = {i: int(m) for i, m in enumerate(openmc.lib.materials)}
    id2i = {v: k for k, v in mats.items()}
    with h5py.File(files['field']) as f:
        mid = f['regions/material_id'][()]
    region_mat = np.array([id2i.get(int(x), -1) for x in mid])
    grids = {g: rb.union_grid(NUCLIDES, edges[g], edges[g + 1]) for g in range(len(edges) - 1)}
    return edges, mats, region_mat, grids


def load_partner(rmc, args, edges, mats, region_mat, grids):
    """Build and load the partner (slot 0) and, for bands, the same bands as
    the proposal library (slot 1). Returns the library info."""
    import resolve_basis as rb
    b, m = map(int, args.b_m.split('x'))
    G = len(edges) - 1
    if args.partner == 'bands':
        blocks, cross, sources, info = rb.build_subgroup(rmc, mats, region_mat, grids, b, m,
                                                         spacing=args.spacing)
        bands = {k: (v['band'], v['mass'], v['sigma_el']) for k, v in info.items()}
        for slot in (0, 1):
            rmc.resolve_load(len(mats), G, blocks, cross, sources, slot=slot, bands=bands)
    else:
        generic = {'tuned': False, 'generic': True, 'src': 'src'}[args.start]
        blocks, cross, sources, info = rb.build(rmc, mats, region_mat, list(edges), grids, b, m,
                                                generic=generic)
        rmc.resolve_load(len(mats), G, blocks, cross, sources)
    return info


def importance(rmc, files, rep):
    """Importance table and the bins' mean importance under the field's shape."""
    import cancellation as canc
    import residual_efficiency as reff
    table = np.load(files['work'] / 'importance_table.npy')
    keys = np.arange(rep.n_bins())
    z = (keys // (rep.D * rep.NE)) % rep.Z
    e = (keys // rep.D) % rep.NE
    t = keys // (rep.D * rep.NE * rep.Z)
    occ = np.array([len(rep.regions_of[zz]) > 0 and
                    (rep.V[rep.regions_of[zz]] * rep.Q['sf'[tt]][rep.regions_of[zz], rep.e2g[ee]]).sum() > 0
                    for zz, ee, tt in zip(z, e, t)])
    cp = canc.control_particles(rep, occ.astype(float), 600_000, np.random.default_rng(3))
    kb = rep.bins(cp)
    eta = reff.lookup(table, rmc, cp['r'], cp['E'])
    ok = kb >= 0
    cnt = np.bincount(kb[ok], minlength=rep.n_bins())
    etab = np.bincount(kb[ok], weights=eta[ok], minlength=rep.n_bins()) / np.maximum(cnt, 1)
    return table, etab


def cmd_rho(rmc, args, files, k_mg):
    edges, mats, region_mat, grids = setup(rmc, files)
    out = {}
    for bm in args.b_m_list:
        args.b_m = bm
        t0 = time.time()
        info = load_partner(rmc, args, edges, mats, region_mat, grids)
        res = {'build_s': time.time() - t0, 'L': {f'{m},{g}': v['L'] for (m, g), v in info.items()}}
        for tag in 'sf':
            D = np.array([rmc.forward_points(tag, 5, args.n, k=k_mg, lam=0.5, seed=seed(i, f'rho{tag}'))[1]
                          for i in range(args.batches)])
            m_, se = D.mean(0), D.std(0, ddof=1) / math.sqrt(len(D))
            res[tag] = {name: [m_[i] / m_[0], se[i] / m_[0]]
                        for name, i in (('exact', 17), ('P1', 18), ('P3', 19), ('P7', 20),
                                        ('kinematic_mg', 7))}
        out[bm] = res
        print(bm, json.dumps({t: {k: round(v[0], 4) for k, v in res[t].items()} for t in 'sf'}), flush=True)
    return out


def cmd_noise(rmc, args, files, k_mg):
    import cancellation as canc
    import residual_efficiency as reff
    from pipeline import ce_session  # noqa: F401  (session opened by caller)
    edges, mats, region_mat, grids = setup(rmc, files)
    load_partner(rmc, args, edges, mats, region_mat, grids)
    rep = canc.Representation(rmc, files, args.mesh)
    table, etab = importance(rmc, files, rep)
    out = {}
    for spec in args.parts:
        tag, part = spec.split(':')
        part = int(part)
        rows, T = [], 0.0
        for b in range(args.batches):
            t0 = time.perf_counter()
            pts, d = rmc.forward_points(tag, part, args.n, k=k_mg, lam=0.45 if part == 8 else 0.5,
                                        seed=seed(b, f'noise{tag}{part}'))
            T += time.perf_counter() - t0
            bank = dict(r=pts['r'], u=pts['u'], E=pts['E'],
                        tag=np.full(len(pts['E']), 0 if tag == 's' else 1))
            w = pts['w_ce'] - pts['w_mg']
            key = rep.bins(bank)
            ok = key >= 0
            eta = reff.lookup(table, rmc, pts['r'], pts['E'])
            etb = np.where(ok, etab[np.maximum(key, 0)], eta)
            rows.append([(w * eta).sum(), (w * etb).sum(), (w * (eta - etb)).sum(),
                         np.abs(w).sum(), d[0], d[1]])
        rows = np.array(rows)
        m_, se = rows.mean(0), rows.std(0, ddof=1) / math.sqrt(len(rows))
        r = dict(us=T / (len(rows) * args.n) * 1e6, W=m_[3], total=se[0] * math.sqrt(T),
                 bin=se[1] * math.sqrt(T), within=se[2] * math.sqrt(T),
                 sum_ce=[m_[4], se[4]], sum_partner=[m_[5], se[5]])
        out[spec] = r
        print(spec, json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()}),
              flush=True)
    return out


def cmd_closure(rmc, args, files, k_mg):
    import cancellation as canc
    import resolve_basis as rb
    args.partner = 'bands'
    edges, mats, region_mat, grids = setup(rmc, files)
    info = load_partner(rmc, args, edges, mats, region_mat, grids)
    rep = canc.Representation(rmc, files, args.mesh)
    _, etab = importance(rmc, files, rep)
    G = len(edges) - 1
    T = rb.elastic_band_transfer(rmc, mats, grids, info, edges)
    # importance by (zone, group), tag s, summed over direction bins
    eb = etab[:rep.n_bins() // 2].reshape(rep.Z, rep.NE, rep.D).mean(axis=2)
    C, times = [], []
    for i in range(args.sweeps):
        t0 = time.time()
        Phi, off = rmc.band_sweep(args.rays, dead=args.dead, active=args.active,
                                  seed=seed(i, 'sweep'), n_sample=args.n_sample)
        times.append(time.time() - t0)
        C.append(rb.band_closure(Phi, off, T, region_mat, rep.zone, rep.Z))
    C = np.array(C)
    Y = (C * eb[None, :, :G]).sum(axis=(1, 2))
    P = []
    for b in range(args.batches):
        pts, _ = rmc.forward_points('s', 8, args.n, k=k_mg, lam=0.45, seed=seed(b, 'closure'))
        z = rep.zone[np.maximum(pts['region'], 0)]
        go = np.clip(np.searchsorted(edges, pts['E'], side='right') - 1, 0, G - 1)
        P.append(np.bincount(z * G + go, weights=pts['w_mg'], minlength=rep.Z * G).reshape(rep.Z, G))
    P = np.array(P)
    cm, cs = C.mean(0), C.std(0, ddof=1) / math.sqrt(len(C))
    pm, ps = P.mean(0), P.std(0, ddof=1) / math.sqrt(len(P))
    sel = (np.abs(cm) + np.abs(pm)) > 1e-3
    z = ((cm - pm) / np.sqrt(cs**2 + ps**2 + 1e-30))[sel]
    out = dict(sweep_s=float(np.mean(times)), closure_total=float(cm.sum()),
               noise=float(Y.std(ddof=1) * math.sqrt(np.mean(times))),
               max_abs_z=float(np.abs(z).max()), mean_z2=float(np.mean(z**2)), cells=int(sel.sum()))
    print(json.dumps(out), flush=True)
    if args.verbose:
        zmat = np.array([region_mat[rep.regions_of[q][0]] if len(rep.regions_of[q]) else -1
                         for q in range(rep.Z)])
        for mi in mats:
            for g in range(G):
                a, b = cm[zmat == mi, g].sum(), pm[zmat == mi, g].sum()
                print(f'  material {mi} group {g}: closure {a:.4f}  points {b:.4f}')
        print('  by zone, group 6:', np.round(cm[:, 6], 3), np.round(pm[:, 6], 3))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command', choices=['rho', 'noise', 'closure'])
    p.add_argument('--work', default='pincell')
    p.add_argument('--partner', choices=['lanes', 'bands'], default='bands')
    p.add_argument('--start', choices=['tuned', 'generic', 'src'], default='src',
                   help='starting block of the lanes')
    p.add_argument('--spacing', choices=['log', 'quantile'], default='log')
    p.add_argument('--b-m', nargs='+', default=['16x8'],
                   help='lanes: width x depth; bands: windows x levels')
    p.add_argument('--parts', nargs='+', default=['s:0', 's:8', 'f:7'])
    p.add_argument('--n', type=int, default=10000)
    p.add_argument('--batches', type=int, default=16)
    p.add_argument('--sweeps', type=int, default=8)
    p.add_argument('--rays', type=int, default=20)
    p.add_argument('--dead', type=float, default=40.0)
    p.add_argument('--active', type=float, default=200.0)
    p.add_argument('--n-sample', type=int, default=0)
    p.add_argument('--out', default=None)
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args()
    import fine_lifts as fl
    import pipeline
    w, files, k_mg = fl.fine_lift_files(args.work, FINE)
    args.mesh = w.mesh
    import openmc.lib.rmc as rmc
    args.b_m_list = args.b_m
    args.b_m = args.b_m[0]
    with pipeline.ce_session(w.model, w.mesh, files['work'] / f'partner_{args.command}'):
        rmc.field_load(files['field'], files['lifts'], files['volumes'])
        out = {'rho': cmd_rho, 'noise': cmd_noise, 'closure': cmd_closure}[args.command](
            rmc, args, files, k_mg)
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1, default=float))


if __name__ == '__main__':
    main()
