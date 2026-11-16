"""Lifts on an energy grid refined where the CE emission has fine structure
(the resolved-resonance groups), for the field of the controlled study.

    python fine_lifts.py --work rmc_visual_work --bins 200 200 200 200 200 4000 800 200

One count of fine bins per group (ascending energy). The field itself (its
group emission) is unchanged: only the within-group shapes are resolved more
finely, so that the low term follows the resonance structure of the
scattering emission in the fuel.
"""
import argparse
from pathlib import Path

import numpy as np

import controlled_residual_study as cs
import field as fld
import mg

LIFTS = dict(particles=20_000, batches=210, inactive=10)


def fine_grid(group_edges, bins):
    out = []
    for a, b, n in zip(group_edges[:-1], group_edges[1:], bins):
        out.append(np.geomspace(max(a, 1e-5), b, n + 1)[:-1])
    return np.concatenate(out + [[group_edges[-1]]])


def fine_lift_files(work, bins, lifts=LIFTS):
    w, files, k = cs.exact_volume_files(work)
    tag = '_'.join(str(b) for b in bins)
    d = Path(files['work']) / f'fine_lifts_{tag}'
    out = d / 'rmc_lifts.h5'
    if not out.exists():
        F = fld.FieldFile.load(files['field'])
        fine = fine_grid(F.edges, bins)
        sp = mg.run_lift_tallies(w.model, fine, d / 'lift', **lifts)
        fld.build_lifts(sp, fine, F.edges, [m.id for m in w.model.materials], out)
    return w, dict(files, lifts=out), k


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--work', default='rmc_visual_work')
    ap.add_argument('--bins', type=int, nargs='+', required=True)
    args = ap.parse_args()
    _, files, _ = fine_lift_files(args.work, args.bins)
    print(files['lifts'])


if __name__ == '__main__':
    main()
