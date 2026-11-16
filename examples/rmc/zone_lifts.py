"""Lifts resolved in space: one within-group shape per zone of field regions
(radial rings of the fuel, rings of the graphite) instead of one per
material, from a CE eigenvalue run tallying emission and flux spectra by
source region.

    python zone_lifts.py --work rmc_visual_work --rings 4 2

The lift file keeps the material rows (so every material still has a lift)
and adds one row per zone; region_rows maps every field region to its
zone's row (rmc::FrozenField). A zone is a set of regions of one cell whose
mesh-bin centres lie in one ring around the pin axis, so a zone collects
enough events per fine energy bin.
"""
import argparse
import copy
from pathlib import Path

import h5py
import numpy as np

import openmc

import controlled_residual_study as cs
import field as fld
import problems

LIFTS = dict(particles=20_000, batches=110, inactive=10)


def run_region_tallies(model, mesh, fine_edges, workdir, particles, batches, inactive):
    """CE run tallying nu-scatter and nu-fission by outgoing energy and the
    flux by energy, per (cell, mesh bin)."""
    m = copy.deepcopy(model)
    cells = [c for c in m.geometry.get_all_cells().values() if c.fill is not None]
    f_cell = openmc.CellFilter(cells)
    f_mesh = openmc.MeshFilter(mesh)
    t_e = openmc.Tally(name='zone_emission')
    t_e.filters = [f_cell, f_mesh, openmc.EnergyoutFilter(fine_edges)]
    t_e.scores = ['nu-scatter', 'nu-fission']
    t_p = openmc.Tally(name='zone_flux')
    t_p.filters = [f_cell, f_mesh, openmc.EnergyFilter(fine_edges)]
    t_p.scores = ['flux']
    m.tallies = openmc.Tallies([t_e, t_p])
    m.settings.particles, m.settings.batches, m.settings.inactive = particles, batches, inactive
    Path(workdir).mkdir(parents=True, exist_ok=True)
    return Path(m.run(cwd=workdir, output=False)), [c.id for c in cells]


def zone_of_regions(F, mesh, rings):
    """Zone index of every field region: (cell, ring of its mesh-bin centre)."""
    ll = np.array(mesh.lower_left)
    dim = np.array(mesh.dimension)
    w = (np.array(mesh.upper_right) - ll) / dim
    b = F.mesh_bin
    cx = ll[0] + (b % dim[0] + 0.5) * w[0]
    cy = ll[1] + ((b // dim[0]) % dim[1] + 0.5) * w[1]
    rad = np.hypot(cx, cy)
    zone = np.empty(len(b), dtype=np.int64)
    keys = {}
    for i, (cell, r) in enumerate(zip(F.cell_id, rad)):
        if cell == problems.CELL_FUEL:
            ring = min(int(rings[0] * r / problems.R_FUEL), rings[0] - 1)
        else:
            rmax = np.hypot(*(np.array(mesh.upper_right[:2])))
            x = (r - problems.R_FUEL) / (rmax - problems.R_FUEL)
            ring = min(max(int(rings[1] * x), 0), rings[1] - 1)
        zone[i] = keys.setdefault((int(cell), ring), len(keys))
    return zone, keys


def build_zone_lifts(sp_path, F, mesh, cell_ids, base_lifts, zone, out_path):
    with h5py.File(base_lifts) as f:
        fine = f['fine_edges'][()]
        group_edges = f['group_edges'][()]
        mat_ids = f['material_ids'][()]
        base = {k: f[k][()] for k in ('h_s', 'h_f', 'h_phi')}
    dE = np.diff(fine)
    gof = np.searchsorted(group_edges, fine[:-1], side='right') - 1
    nF = len(fine) - 1
    nbins = int(np.prod(mesh.dimension))
    with openmc.StatePoint(sp_path) as sp:
        te, tp = sp.get_tally(name='zone_emission'), sp.get_tally(name='zone_flux')
        # mean arrays shaped [cell, mesh bin, energy, score]
        em = te.mean.reshape(len(cell_ids), nbins, nF, 2)
        fl = tp.mean.reshape(len(cell_ids), nbins, nF)
    rows = {k: [base[k].reshape(len(mat_ids), nF)[i] for i in range(len(mat_ids))]
            for k in base}
    ids = list(mat_ids)
    region_rows = np.empty(len(zone), dtype=np.int32)
    for z in range(zone.max() + 1):
        regs = np.flatnonzero(zone == z)
        ci = cell_ids.index(int(F.cell_id[regs[0]]))
        bins = F.mesh_bin[regs]
        s = em[ci, bins, :, 0].sum(axis=0)
        f_ = em[ci, bins, :, 1].sum(axis=0)
        p = fl[ci, bins, :].sum(axis=0)
        mrow = list(mat_ids).index(int(F.material_id[regs[0]]))
        for key, v in (('h_s', s), ('h_f', f_), ('h_phi', p)):
            h = fld._normalize_per_group(v / dE, dE, gof, fine)
            # groups with no tallied content keep the material's shape
            for g in np.unique(gof):
                sel = gof == g
                if not np.sum(v[sel]) > 0:
                    h[sel] = rows[key][mrow][sel]
            rows[key].append(h)
        ids.append(int(F.material_id[regs[0]]))
        region_rows[regs] = len(ids) - 1
    with h5py.File(out_path, 'w') as f:
        f.attrs['version'] = 1
        f['fine_edges'] = fine
        f['group_edges'] = group_edges
        f['material_ids'] = np.asarray(ids, dtype=np.int32)
        for key in ('h_s', 'h_f', 'h_phi'):
            f[key] = np.concatenate(rows[key])
        f['region_rows'] = region_rows


def zone_lift_files(work, rings=(4, 2), lifts=LIFTS):
    w, files, k = cs.exact_volume_files(work)
    d = Path(files['work']) / f'zone_lifts_{rings[0]}_{rings[1]}'
    out = d / 'rmc_lifts_zones.h5'
    if not out.exists():
        F = fld.FieldFile.load(files['field'])
        with h5py.File(files['lifts']) as f:
            fine = f['fine_edges'][()]
        sp, cell_ids = run_region_tallies(w.model, w.mesh, fine, d / 'ce', **lifts)
        zone, _ = zone_of_regions(F, w.mesh, rings)
        build_zone_lifts(sp, F, w.mesh, cell_ids, files['lifts'], zone, out)
    files = dict(files, lifts=out)
    return w, files, k


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--work', default='rmc_visual_work')
    ap.add_argument('--rings', type=int, nargs=2, default=[4, 2])
    args = ap.parse_args()
    _, files, _ = zone_lift_files(args.work, tuple(args.rings))
    print(files['lifts'])


if __name__ == '__main__':
    main()
