"""The multigroup first-collision emission of the frozen field by netting
bin, from random ray transport sweeps of the field's own sources.

    python multigroup_deposits.py --work rmc_visual_work \\
        --fine-lifts 200 200 200 200 200 4000 800 200 --batches 200 \\
        --out multigroup_flux.npz

The binned control variate (stage8.py --variant bcv, cancellation.
Representation) needs, per source tag t, M_t = S T Q_t by bin: the multigroup
scattering and fission emission of the uncollided multigroup flux T Q_t of
the field's flat source Q_t. One random ray sweep with a fixed source
(rmc.sweep_field) is that transport, and its track-length estimate of the
region-averaged flux is unbiased for it (up to the dead zone's truncation of
the incoming flux), with far less variance than the forward rays'
first-collision estimate. This script averages the region fluxes of many
sweeps, phi_t [R, G] in ascending groups, and stores them with their
standard errors and the time spent; Representation.multigroup_bins() turns
them into bins.
"""
import argparse
import math
import time
from pathlib import Path

import numpy as np

import controlled_residual_study as cs
import pipeline


def sweep_fluxes(rmc, n_batches, first_batch=4):
    """Region-averaged uncollided multigroup fluxes of Q_s and Q_f per
    sweep, [n_batches, 2, R, G] (ascending groups)."""
    R = rmc.field_n_regions()
    out = []
    for b in range(n_batches):
        rmc.sweep_field(batch=first_batch + b)
        phi = np.zeros((2, R, len(rmc.field_groups()) - 1))
        for r in range(R):
            seg = rmc.sweep_field_region(r)
            L = seg['length'].sum()
            if not L > 0:
                continue
            for it, key in enumerate(('psi_s', 'psi_f')):
                phi[it, r] = 4 * math.pi * (seg['length'][:, None] * seg[key]).sum(axis=0) / L
        out.append(phi)
    return np.array(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--work', default='rmc_visual_work')
    ap.add_argument('--fine-lifts', type=int, nargs='+', default=None)
    ap.add_argument('--batches', type=int, default=200)
    ap.add_argument('--rays', type=int, default=500, help='rays per sweep')
    ap.add_argument('--first-batch', type=int, default=4,
                    help='sweeps use the rays of batches first-batch, first-batch + 1, ... '
                         '(independent files need disjoint ranges)')
    ap.add_argument('--out', default='multigroup_flux.npz')
    args = ap.parse_args()
    if args.fine_lifts:
        import fine_lifts as fl
        w, files, _ = fl.fine_lift_files(args.work, args.fine_lifts)
    else:
        w, files, _ = cs.exact_volume_files(args.work)
    import openmc.lib.rmc as rmc
    with pipeline.sweep_session(w.model, w.mesh, files['mgxs'], files['field'],
                                files['work'] / 'multigroup_deposits', rays=args.rays,
                                batches=args.first_batch + args.batches + 10):
        rmc.field_load(files['field'], files['lifts'], files['volumes'])
        rmc.sweep_init()
        rmc.sweep_iterate(3)
        t0 = time.perf_counter()
        phi = sweep_fluxes(rmc, args.batches, args.first_batch)
        T = time.perf_counter() - t0
        rmc.sweep_finalize()
    np.savez(args.out, phi=phi.mean(axis=0),
             se=phi.std(axis=0, ddof=1) / math.sqrt(len(phi)), time=T,
             batches=args.batches, rays=args.rays)
    print(f'{args.batches} sweeps of {args.rays} rays in {T:.0f} s')


if __name__ == '__main__':
    main()
