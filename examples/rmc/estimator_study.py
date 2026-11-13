"""Measurements that choose among the residual estimators (gates G3.7, G4.5,
G4.6). None of them is pass/fail; each prints a table and writes it as JSON.

    python estimator_study.py --work rmc_study_work --n 2000          # smoke run
    python estimator_study.py --work rmc_study_work --n 40000 --rays 500 2000 8000

- G3.7 (same-path correlation): for isotropic directions with the same-path
  control, the per-direction CE estimate q_CE and MG estimate q_MG at each
  sample's outgoing state. Reported by tag and outgoing-energy range: the
  correlation of their deviations from the sample's mean, and
  Var(q_CE - q_MG) / Var(q_CE). A ratio well below 1 means the P0 MG
  control removes most of the directional noise.
- G4.5 (weight distribution, a baseline): sum |w|, the 99th percentile of
  |w|, and the variance of the per-sample residual, for the density branch
  and the sampler-route particles separately.
- G4.6 (choosing the estimator): for each variant of the estimator family
  (direction source, J, L, c, lambda) and each number of rays per sweep, the
  variance of the per-sample residual summed over (material, energy) bins,
  the time per sample, and their product. Lower variance x time is better.
  For tandem and partial variants the full iteration (Stage 6) must also be
  run at fixed CE work, since the sweep's noise enters the field; that is
  not part of this script.

The graphite pincell's field, lifts, and MGXS are built once in --work (as
in visual_checks.py), and the region volumes with the C++ volume function.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

import openmc.lib

import pipeline
import problems
import residual_gates as gates
import visual_checks as vc

E_EDGES = np.array([1e-5, 0.1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6, 3e6, 2e7])
E_RANGES = {'thermal (< 1 eV)': (1e-5, 1.0), 'epithermal (1 eV - 100 keV)': (1.0, 1e5),
            'fast (> 100 keV)': (1e5, 2e7)}

# name: (residual_sample options, c, lambda)
VARIANTS = {
    'frozen, uncontrolled, J=8': (dict(directions='isotropic', n_directions=8), 0.0, 0.0),
    'frozen, same path, J=4 L=4': (dict(directions='isotropic', n_directions=4,
                                        control=True, n_control_directions=4), 1.0, 0.0),
    'frozen, same path, J=8 L=2': (dict(directions='isotropic', n_directions=8,
                                        control=True, n_control_directions=2), 1.0, 0.0),
    'frozen, sweep control, J=8': (dict(directions='sweep', n_directions=8,
                                        control=True), 1.0, 0.0),
    'tandem, J=8': (dict(directions='sweep', n_directions=8, control=True), 1.0, 1.0),
    'partial (0.5), J=8': (dict(directions='sweep', n_directions=8, control=True), 0.5, 0.5),
    'tandem, J=8, same energy, flux proposals': (
        dict(directions='sweep', n_directions=8, control=True,
             control_energy='same_energy', incident='flux', beta=0.5), 1.0, 1.0),
}


def _material_of(r):
    return np.where(np.hypot(r[:, 0], r[:, 1]) < problems.R_FUEL, 0, 1)


def _files(work):
    w = vc.Work(work)
    P = w.pincell()
    vol = Path(P['work']) / 'rmc_volumes.h5'
    if not vol.exists():
        with pipeline.ce_session(w.model, w.mesh, Path(P['work']) / 'ce_volumes'):
            pipeline.build_volumes(P['field'], P['lifts'], w.mesh, vol, n_per_voxel=2000)
    import h5py
    with h5py.File(P['field']) as f:
        k = float(f.attrs['k_mg'])
    return w, dict(field=P['field'], lifts=P['lifts'], volumes=vol, mgxs=P['mgxs'],
                   work=Path(P['work'])), k


def same_path_correlation(rmc, n, k):
    out = {}
    for tag in 'sf':
        J = 8
        rec = rmc.residual_sample(tag, n, seed=7, directions='isotropic', n_directions=J,
                                  control=True, n_control_directions=1,
                                  sampler_route=False, keep_directions=True)
        dens = rec['sampler'] == 0
        # Directions are kept only for samples whose material produces the
        # tag (every material for s; the fuel for f)
        has = np.ones(int(dens.sum()), bool) if tag == 's' else \
            _material_of(rec['r'][dens]) == 0
        E = rec['E'][dens][has]
        ce = rec['ce'].reshape(-1, J)
        mg = rec['mg'].reshape(-1, J)
        active = np.ones(len(E), bool)
        for name, (lo, hi) in E_RANGES.items():
            m = active & (E >= lo) & (E < hi)
            if m.sum() < 10:
                continue
            dce = (ce[m] - ce[m].mean(axis=1, keepdims=True)).ravel()
            dmg = (mg[m] - mg[m].mean(axis=1, keepdims=True)).ravel()
            corr = float(np.corrcoef(dce, dmg)[0, 1]) if dce.std() > 0 and dmg.std() > 0 else float('nan')
            ratio = float(np.var(dce - dmg) / np.var(dce)) if np.var(dce) > 0 else float('nan')
            out[f'{tag}: {name}'] = dict(samples=int(m.sum()), correlation=corr,
                                         var_ratio=ratio)
    return out


def weight_baseline(rmc, rec, N, k, c, lam):
    w = rmc.residual_weights(rec, N, (1, 1), (1, 1), k, c, lam)
    out = {}
    for branch, m in (('density', rec['sampler'] == 0), ('sampler', rec['sampler'] == 1)):
        if m.sum() == 0:
            continue
        a = np.abs(w[m])
        out[branch] = dict(sum_abs=float(a.sum()), p99_abs=float(np.percentile(a, 99)),
                           records=int(m.sum()))
    dens = rec['sampler'] == 0
    sample_of = np.cumsum(np.r_[0, dens[:-1]])
    per = np.zeros(N)
    np.add.at(per, sample_of, w * N)
    out['per_sample_variance'] = float(per.var(ddof=1))
    return out, w


def binned_variance(rec, w):
    _, sem = gates.residual_histogram(rec, w, _material_of, E_EDGES, 2)
    return float((sem**2).sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--work', default='rmc_study_work')
    ap.add_argument('--n', type=int, default=2000, help='samples per tag and variant')
    ap.add_argument('--rays', type=int, nargs='+', default=[500],
                    help='rays per field sweep')
    ap.add_argument('--out', default='estimator_study.json')
    args = ap.parse_args()

    w, files, k = _files(args.work)
    import openmc.lib.rmc as rmc
    results = {'n': args.n, 'G3.7': None, 'variants': []}
    for rays in args.rays:
        with pipeline.sweep_session(w.model, w.mesh, files['mgxs'], files['field'],
                                    files['work'] / f'study_{rays}', rays=rays):
            rmc.field_load(files['field'], files['lifts'], files['volumes'])
            rmc.sweep_init()
            rmc.sweep_iterate(3)
            rmc.sweep_field(batch=4)
            if results['G3.7'] is None:
                results['G3.7'] = same_path_correlation(rmc, max(args.n // 4, 200), k)
            for name, (opts, c, lam) in VARIANTS.items():
                if opts['directions'] == 'isotropic' and rays != args.rays[0]:
                    continue   # does not depend on the sweep
                for tag in 'sf':
                    t0 = time.perf_counter()
                    rec = rmc.residual_sample(tag, args.n, seed=11, E_range=(1e-5, 2e7),
                                              tail='roulette', cutoff=math.exp(-4), **opts)
                    dt = (time.perf_counter() - t0) / args.n
                    base, wts = weight_baseline(rmc, rec, args.n, k, c, lam)
                    var = binned_variance(rec, wts)
                    results['variants'].append(dict(
                        variant=name, tag=tag, rays=rays if opts['directions'] == 'sweep' else None,
                        c=c, lam=lam, time_per_sample=dt, binned_variance=var,
                        variance_x_time=var * dt * args.n, **base))
            rmc.sweep_finalize()
            rmc.field_free()

    print('\nG3.7 same-path correlation (isotropic directions, J = 8)')
    for key, v in results['G3.7'].items():
        print(f"  {key:40s} samples {v['samples']:6d}  corr {v['correlation']:+.3f}"
              f"  Var(CE-MG)/Var(CE) {v['var_ratio']:.3f}")
    print('\nG4.5 / G4.6 by variant (variance summed over bins, time per sample)')
    for v in results['variants']:
        rays = '' if v['rays'] is None else f"{v['rays']} rays"
        print(f"  {v['tag']}  {v['variant']:44s} {rays:10s} var {v['binned_variance']:.3e}"
              f"  t {1e3 * v['time_per_sample']:.2f} ms  var*T {v['variance_x_time']:.3e}")
    Path(args.out).write_text(json.dumps(results, indent=1))
    print('wrote', args.out)


if __name__ == '__main__':
    main()
