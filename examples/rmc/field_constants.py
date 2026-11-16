"""The constants of a frozen field that the balance eigenvalue needs: the CE
nu-fission production X and the net loss rate L of the field's first-flight
flux, per source tag.

    python field_constants.py --work rmc_visual_work --fine-lifts 200 ... \\
        --calls 80 --points 100000 --out field_constants.json

Each call of rmc.field_production is an independent estimate with its own
hashed seed; the file holds the means, their standard errors from the spread
of the calls, and the time spent, so that a run that reads the file
(stage8.py --field-constants) can count that time.
"""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

import controlled_residual_study as cs
import pipeline


def call_seed(label, score, i):
    """An independent 63-bit seed per call (small consecutive seeds of a
    linear congruential generator give related streams)."""
    h = hashlib.sha256(f'{label}/{score}/{i}'.encode()).digest()
    return int.from_bytes(h[:8], 'little') >> 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--work', default='rmc_visual_work')
    ap.add_argument('--fine-lifts', type=int, nargs='+', default=None)
    ap.add_argument('--field-groups', default=None,
                    help='group structure of a field variant (field_variants.py)')
    ap.add_argument('--mgxs-particles', type=int, default=20_000)
    ap.add_argument('--field-mesh', type=int, default=12,
                    help='source-region mesh per side of the field variant')
    ap.add_argument('--calls', type=int, default=80, help='calls for the net loss')
    ap.add_argument('--production-calls', type=int, default=10)
    ap.add_argument('--points', type=int, default=100_000, help='backward points per call')
    ap.add_argument('--loss-estimator', choices=['backward', 'forward'], default='forward',
                    help='net loss from backward points, or from forward rays integrated '
                         'analytically (four times as many rays as points per call)')
    ap.add_argument('--label', default='A', help='seeds differ between labels')
    ap.add_argument('--out', default='field_constants.json')
    args = ap.parse_args()

    base = None
    if args.field_groups:
        import field_variants as fv
        base = fv.field_files(args.work, args.field_groups, args.mgxs_particles,
                              n_per_side=args.field_mesh)
    if args.fine_lifts:
        import fine_lifts as fl
        w, files, _ = fl.fine_lift_files(args.work, args.fine_lifts, base=base)
    else:
        w, files, _ = base or cs.exact_volume_files(args.work)
    import openmc.lib.rmc as rmc
    out = dict(label=args.label, points=args.points)
    with pipeline.ce_session(w.model, w.mesh, files['work'] / f'field_constants_{args.label}_{args.field_mesh}'):
        rmc.field_load(files['field'], files['lifts'], files['volumes'])
        for key, score, n in (('X', 'nu-fission', args.production_calls),
                              ('L', 'loss', args.calls)):
            t0 = time.perf_counter()
            if key == 'L' and args.loss_estimator == 'forward':
                v = np.array([rmc.field_first_flight_rates(
                    4 * args.points, 1, seed=call_seed(args.label, 'forward', i))[1]
                    for i in range(n)])
            else:
                v = np.array([rmc.field_production(args.points, 4, cutoff=math.exp(-3),
                                                   seed=call_seed(args.label, score, i),
                                                   score=score)[0] for i in range(n)])
            out[key] = v.mean(axis=0).tolist()
            out[key + '_se'] = (v.std(axis=0, ddof=1) / math.sqrt(n)).tolist()
            out[key + '_time'] = time.perf_counter() - t0
            out[key + '_calls'] = v.tolist()
            print(f'{key} = {out[key]} +- {out[key + "_se"]}  ({out[key + "_time"]:.0f} s)',
                  flush=True)
    Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
