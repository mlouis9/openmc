"""The RMC eigenvalue iteration on a verification problem (gates G6.3, G6.4).

    python eigenvalue_run.py --problem infinite --inactive 20 --active 40 --replicas 10
    python eigenvalue_run.py --problem infinite --inactive 3 --active 5 --replicas 1   # smoke

Builds the field (random ray), its lifts and volumes in --work, runs
independent replicas of the driver (examples/rmc/driver.py) with the OpenMC
backend, and compares the replica-mean k of the active generations with a
CE Monte Carlo reference under identical physics. Per-generation diagnostics
of every replica (k, f, a, b, rho, W_abs, tag sums, bank size, times) are
written to --out as JSON.

Without sign cancellation (Stage 8), W_abs grows roughly linearly with the
generation, so runs must be short and populations large; these gates test
the mean, not the efficiency.
"""
import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np

import openmc

import driver
import mg
import pipeline
import problems
import visual_checks as vc


# Frozen-field estimators with isotropic directions: (options, c)
ESTIMATORS = {
    'uncontrolled': (dict(), 0.0),
    'same_path': (dict(control=True, n_control_directions=2), 1.0),
    'same_energy': (dict(control=True, n_control_directions=2,
                         control_energy='same_energy', incident='flux', beta=0.5), 1.0),
}


def _files(problem, work):
    work = Path(work)
    if problem == 'pincell':
        w = vc.Work(work)
        P = w.pincell()
        return w.model, w.mesh, dict(field=P['field'], lifts=P['lifts'],
                                     volumes=w.volumes(), work=Path(P['work']))
    # The infinite medium: one region, volume of the box, the pincell's lifts
    w = vc.Work(work)
    lifts = w.pincell()['lifts']
    model, mesh = problems.infinite_medium()
    d = work / 'infinite'
    field = d / 'rr' / 'rmc_field.h5'
    if not field.exists():
        mg.run_random_ray(model, mesh, d / 'rr', mgxs_particles=2000, rays=100,
                          batches=600, inactive=500)
    vol = d / 'rmc_volumes.h5'
    if not vol.exists():
        with pipeline.ce_session(model, mesh, d / 'ce_volumes'):
            pipeline.build_volumes(field, lifts, mesh, vol, n_per_voxel=1000)
    return model, mesh, dict(field=field, lifts=lifts, volumes=vol, work=d)


def reference_k(model, workdir, particles=20_000, batches=150, inactive=50):
    """CE Monte Carlo k under the problem's physics."""
    m = openmc.Model(geometry=model.geometry, materials=model.materials,
                     settings=model.settings)
    m.settings.particles, m.settings.batches = particles, batches
    m.settings.inactive = inactive
    sp_path = m.run(cwd=Path(workdir), output=False)
    with openmc.StatePoint(sp_path) as sp:
        return sp.keff.n, sp.keff.s


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--problem', choices=['infinite', 'pincell'], default='infinite')
    ap.add_argument('--work', default='rmc_eigen_work')
    ap.add_argument('--inactive', type=int, default=20)
    ap.add_argument('--active', type=int, default=40)
    ap.add_argument('--replicas', type=int, default=10)
    ap.add_argument('--N', type=int, default=2000, help='residual samples per tag')
    ap.add_argument('--N-delta', type=int, default=4000, help='flights per generation')
    ap.add_argument('--N-B', type=int, default=20_000, help='bank size after resampling')
    ap.add_argument('--directions', type=int, default=4)
    ap.add_argument('--estimator', choices=list(ESTIMATORS), default='same_energy',
                    help='residual estimator (frozen field, isotropic directions)')
    ap.add_argument('--mass-matching', type=float, default=None,
                    help='damping of per-tag mass matching (default: b = 1)')
    ap.add_argument('--no-reference', action='store_true')
    ap.add_argument('--out', default='eigenvalue_run.json')
    args = ap.parse_args()

    model, mesh, files = _files(args.problem, args.work)
    ref = None
    if not args.no_reference:
        ref = reference_k(model, files['work'] / 'reference')
        print(f'reference CE k = {ref[0]:.5f} +/- {ref[1]:.5f}')

    with h5py.File(files['field']) as f:
        k_mg = float(f.attrs['k_mg'])
    amplitudes = (driver.MassMatching(args.mass_matching) if args.mass_matching
                  else driver.FixedAmplitudes())
    results = []
    import openmc.lib.rmc as rmc
    with pipeline.ce_session(model, mesh, files['work'] / 'ce_eigen'):
        rmc.field_load(files['field'], files['lifts'], files['volumes'])
        opts, c = ESTIMATORS[args.estimator]
        backend = driver.OpenMCBackend(dict(n_directions=args.directions, tail='roulette',
                                            cutoff=math.exp(-4), **opts), c=c)
        for rep in range(args.replicas):
            cfg = driver.Config(N={'s': args.N, 'f': args.N}, N_delta=args.N_delta,
                                N_B=args.N_B, amplitudes=amplitudes, seed=1000 + rep)
            _, rows = driver.run(
                backend, cfg, k_mg, args.inactive + args.active,
                callback=lambda s, d: print(
                    f"  replica {rep} gen {d['n']:3d}  k {d['k_next']:.5f}  "
                    f"f {d['f']:.5f}  rho {d['rho']:.4f}  W_abs {d['W_abs']:.4g}  "
                    f"bank {d['size']} -> {d['size_resampled']}  {d['time']:.1f} s"))
            k_active = np.array([r['k_next'] for r in rows[args.inactive:]])
            results.append(dict(replica=rep, rows=rows, k_mean=float(k_active.mean())))
        rmc.field_free()

    km = np.array([r['k_mean'] for r in results])
    mean = km.mean()
    sem = km.std(ddof=1) / math.sqrt(len(km)) if len(km) > 1 else float('nan')
    print(f'\nk_MG = {k_mg:.5f}; RMC replica-mean k = {mean:.5f} +/- {sem:.5f} '
          f'({len(km)} replicas)')
    if ref is not None:
        print(f'reference CE k = {ref[0]:.5f} +/- {ref[1]:.5f}; difference '
              f'{1e5 * (mean - ref[0]):.0f} pcm')
    Path(args.out).write_text(json.dumps(dict(args=vars(args), k_mg=k_mg, reference=ref,
                                              replicas=results), indent=1,
                                         default=float))
    print('wrote', args.out)


if __name__ == '__main__':
    main()
