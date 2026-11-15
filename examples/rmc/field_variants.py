"""Fields of the graphite pincell with other group structures and MGXS
statistics, each with prescribed exact volumes, its own lifts, and a volume
file, for comparing how the field's accuracy sets the residual's noise.

    python field_variants.py --work rmc_visual_work --groups CASMO-40

field_files(work, groups, ...) returns (Work, files, k_MG) like
controlled_residual_study.exact_volume_files, building what is missing in
work/pincell_<groups>_<mgxs particles>.
"""
import argparse
from pathlib import Path

import numpy as np

import field as fld
import mg
import problems
import visual_checks as vc
import volumes as vol

# MGXS and lift statistics of the variants (the 8-group field of the visual
# checks uses vc.RR and vc.LIFTS)
RR = dict(rays=500, batches=500, inactive=300)
LIFTS = dict(particles=20_000, batches=60, inactive=10)


class _Problem:
    """The pincell's CE model and source-region mesh (as visual_checks.Work)."""

    def __init__(self, n_per_side):
        self.model, self.mesh = problems.graphite_pincell(n_per_side)


def field_files(work, groups='CASMO-40', mgxs_particles=20_000, fine_bins=1600,
                n_per_side=12):
    w = vc.Work(work) if n_per_side == 12 else _Problem(n_per_side)
    tag = '' if n_per_side == 12 else f'_mesh{n_per_side}'
    d = Path(work) / f'pincell_{groups}_{mgxs_particles}{tag}'
    d.mkdir(parents=True, exist_ok=True)
    sr = d / 'sr_volumes.h5'
    cid, inst, b, V = vol.pincell_volumes(w.mesh)
    if not sr.exists():
        vol.write_source_region_volumes(sr, cid, inst, b, V)
    field = d / 'rr' / 'rmc_field.h5'
    if not field.exists():
        rr = {'volume_estimator': 'prescribed', 'source_region_volumes': str(sr)}
        mg.run_random_ray(w.model, w.mesh, d / 'rr', groups=groups,
                          mgxs_particles=mgxs_particles, random_ray=rr, **RR)
    F = fld.FieldFile.load(field)
    lifts = d / 'rmc_lifts.h5'
    if not lifts.exists():
        per_group = max(1, fine_bins // (len(F.edges) - 1))
        fine = mg.fine_grid(F.edges, per_group)
        sp = mg.run_lift_tallies(w.model, fine, d / 'lift', **LIFTS)
        fld.build_lifts(sp, fine, F.edges, [m.id for m in w.model.materials], lifts)
    exact = {(c, i, m): v for c, i, m, v in zip(cid.tolist(), inst.tolist(), b.tolist(), V)}
    VR = np.array([exact[key] for key in F.keys()])
    volumes = d / 'rmc_volumes_exact.h5'
    if not volumes.exists():
        zeros = np.zeros(len(VR), dtype=np.int64)
        fld.write_volumes(volumes, VR, VR, zeros, zeros, fld.field_scale(F, VR), 0, (0, 0))
    return w, dict(field=field, lifts=lifts, volumes=volumes, mgxs=d / 'rr' / 'mgxs.h5',
                   work=d), float(F.k_mg)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--work', default='rmc_visual_work')
    ap.add_argument('--groups', default='CASMO-40')
    ap.add_argument('--mgxs-particles', type=int, default=20_000)
    ap.add_argument('--mesh', type=int, default=12, help='source-region mesh per side')
    args = ap.parse_args()
    _, files, k = field_files(args.work, args.groups, args.mgxs_particles,
                              n_per_side=args.mesh)
    print(files, 'k_MG', k)


if __name__ == '__main__':
    main()
