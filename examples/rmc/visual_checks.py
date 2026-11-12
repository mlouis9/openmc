"""Figures that show each residual Monte Carlo (RMC) component working.

Run from this directory (needs OPENMC_CROSS_SECTIONS and an OpenMC build
whose openmc.lib loads the RMC library):

    python visual_checks.py --out ../../docs/source/devguide/rmc_figures

Files are built once in --work (default ./rmc_visual_work) and reused on
later runs; delete it to rebuild. Each fig_* function makes one figure and is
included in docs/source/devguide/rmc_visual.rst, which explains what to look
for.
"""
import argparse
import math
import os
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import openmc
import openmc.lib
import openmc.lib.rmc as rmc

import field as fld
import mg
import pipeline
import problems
import volumes as vol

# Settings: the graphite pincell's random ray needs a few hundred inactive
# batches; the lift tallies use a fine grid so that resonances are visible.
RR = dict(mgxs_particles=5_000, rays=500, batches=500, inactive=300)
LIFTS = dict(particles=20_000, batches=40, inactive=10)
LIFT_BINS_PER_GROUP = 200
U238_RESONANCES = (6.67, 20.87, 36.68, 66.03, 80.75, 102.6, 116.9, 165.3, 189.6, 208.5)


# ---------------------------------------------------------------------------
# Files, built once
# ---------------------------------------------------------------------------

class Work:
    """Builds and caches the files the figures need."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.model, self.mesh = problems.graphite_pincell()

    def pincell(self):
        """Field, lifts, MGXS, statepoint, and MG model of the pincell."""
        w = self.root / 'pincell'
        field = w / 'rr' / 'rmc_field.h5'
        lifts = w / 'rmc_lifts.h5'
        if not (field.exists() and lifts.exists()):
            pipeline.build_mg_files(self.model, self.mesh, w, rr_kwargs=RR,
                                    lift_kwargs=LIFTS, per_group=LIFT_BINS_PER_GROUP)
        _, sp, mgxs, mg_model = self._mg_model(w)
        return dict(field=field, lifts=lifts, mgxs=mgxs, statepoint=sp,
                    mg_model=mg_model, work=w)

    def _mg_model(self, w):
        # The MG model is rebuilt (cheaply) from the cached MGXS library
        import copy
        m = copy.deepcopy(self.model)
        m.convert_to_multigroup(method='material_wise', groups='CASMO-8',
                                mgxs_path=w / 'rr' / 'mgxs.h5')
        m.convert_to_random_ray()
        m.settings.random_ray['source_region_meshes'] = mg.source_region_meshes(m, self.mesh)
        m.settings.particles, m.settings.batches = RR['rays'], RR['batches']
        m.settings.inactive = RR['inactive']
        sp = sorted((w / 'rr').glob('statepoint.*.h5'))[-1]
        return None, sp, w / 'rr' / 'mgxs.h5', m

    def split(self):
        """Field of the split-mesh pincell (short run; geometry only)."""
        model, pairs = problems.graphite_pincell_split_meshes()
        w = self.root / 'split'
        field = w / 'rr' / 'rmc_field.h5'
        if not field.exists():
            mg.run_random_ray(model, pairs, w / 'rr',
                              mgxs_path=self.pincell()['mgxs'],
                              rays=400, batches=30, inactive=10)
        return dict(model=model, pairs=pairs, field=field, work=w)

    def slab(self):
        model, mesh = problems.slab_box()
        w = self.root / 'slab'
        field = w / 'rr' / 'rmc_field.h5'
        if not field.exists():
            mg.run_random_ray(model, mesh, w / 'rr', mgxs_path=self.pincell()['mgxs'],
                              rays=200, batches=12, inactive=4)
        return dict(model=model, mesh=mesh, field=field, work=w)


def _save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / name, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print('wrote', out / name)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_lifts(work, out):
    """Within-group emission shapes (lifts) and the flux spectrum, per material."""
    P = work.pincell()
    with h5py.File(P['lifts'], 'r') as f:
        E = f['fine_edges'][()]
        ge = f['group_edges'][()]
        ids = list(f['material_ids'][()])
        nF = len(E) - 1
        h = {k: f[k][()].reshape(len(ids), nF) for k in ('h_s', 'h_f', 'h_phi')}
    mid = np.sqrt(E[:-1] * E[1:])
    names = {problems.MAT_FUEL: 'UO$_2$ fuel', problems.MAT_MOD: 'graphite'}

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for col, mat in enumerate((problems.MAT_FUEL, problems.MAT_MOD)):
        i = ids.index(mat)
        ax = axes[0, col]
        for key, label in (('h_s', 'scattering emission $h_s$'),
                           ('h_f', 'fission emission $h_f$'),
                           ('h_phi', 'flux shape')):
            y = mid * h[key][i]          # per unit lethargy within each group
            if np.any(y > 0):
                ax.step(E[:-1], y, where='post', lw=0.8, label=label)
        for e in ge[1:-1]:
            ax.axvline(e, color='0.7', lw=0.5, ls='--')
        ax.set(xscale='log', yscale='log', xlim=(1e-3, 2e7),
               title=f'{names[mat]}: lifts, per unit lethargy (each group sums to 1)',
               xlabel='E [eV]', ylabel='$E\\,h(E)$')
        ax.legend(fontsize=8)

        ax = axes[1, col]
        sel = (mid > 1) & (mid < 300)
        ax.step(E[:-1][sel], (mid * h['h_phi'][i])[sel], where='post', lw=0.8,
                label='flux shape')
        ax.step(E[:-1][sel], (mid * h['h_s'][i])[sel], where='post', lw=0.8,
                label='scattering emission')
        for e in U238_RESONANCES:
            ax.axvline(e, color='C3', lw=0.6, alpha=0.6)
        ax.set(xscale='log', yscale='log', xlim=(1, 300),
               title=f'{names[mat]}, 1-300 eV (red: U-238 resonance energies)',
               xlabel='E [eV]', ylabel='$E\\,h(E)$')
        ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, out, 'lifts.png')


def _grid(lo, hi, n):
    x = np.linspace(lo[0], hi[0], n)
    y = np.linspace(lo[1], hi[1], n)
    X, Y = np.meshgrid(x, y)
    pts = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])
    return X, Y, pts


def fig_field_maps(work, out):
    """The frozen field: Q_s and Q_f by region, in a fast and a thermal group."""
    P = work.pincell()
    F = fld.FieldFile.load(P['field'])
    with pipeline.ce_session(work.model, work.mesh, P['work'] / 'ce_maps'):
        rmc.field_load(P['field'], P['lifts'])
        lo, hi = F.box
        X, Y, pts = _grid(lo, hi, 300)
        R = rmc.field_locate(pts).reshape(X.shape)
        rmc.field_free()
    G = F.n_groups
    panels = (('$Q^{MG}_s$', F.Q_s[:, G - 1], f'group [{F.edges[G-1]:.3g}, {F.edges[G]:.3g}) eV'),
              ('$Q^{MG}_s$', F.Q_s[:, 0], f'group [{F.edges[0]:.3g}, {F.edges[1]:.3g}) eV'),
              ('$Q^{MG}_f$', F.Q_f.sum(axis=1), 'summed over groups'))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, (name, q, label) in zip(axes, panels):
        img = np.where(R >= 0, q[np.maximum(R, 0)], np.nan)
        m = ax.pcolormesh(X, Y, img, shading='auto')
        fig.colorbar(m, ax=ax, shrink=0.8)
        ax.add_patch(plt.Circle((0, 0), problems.R_FUEL, fill=False, color='w', lw=0.8))
        ax.set(aspect='equal', title=f'{name}, {label}', xlabel='x [cm]', ylabel='y [cm]')
    fig.suptitle('Frozen field by source region (graphite pincell, 12 x 12 mesh; '
                 'unnormalized)')
    fig.tight_layout()
    _save(fig, out, 'field_maps.png')


def fig_split_meshes(work, out):
    """Regions of a field with two meshes and an unmeshed cell, and their
    volumes against the closed forms."""
    S = work.split()
    P = work.pincell()
    F = fld.FieldFile.load(S['field'])
    with pipeline.ce_session(S['model'], S['pairs'], S['work'] / 'ce'):
        rmc.field_load(S['field'], P['lifts'])
        lo, hi = F.box
        X, Y, pts = _grid(lo, hi, 500)
        R = rmc.field_locate(pts).reshape(X.shape)
        n_vox, n_box = 4_000, 40_000
        V, counts = rmc.field_region_volumes(n_vox, seed=1, n_box=n_box)
        rmc.field_free()

    # Closed forms (as in tests/rmc/test_multiple_meshes.py)
    (m_fuel, _), (m_outer, _) = S['pairs']
    h = problems.HEIGHT

    def voxel(mesh, b):
        nx, ny, _ = mesh.dimension
        ll = np.asarray(mesh.lower_left)
        w = (np.asarray(mesh.upper_right) - ll) / mesh.dimension
        ix, iy = b % nx, b // nx
        return ll[0] + ix * w[0], ll[0] + (ix + 1) * w[0], ll[1] + iy * w[1], \
            ll[1] + (iy + 1) * w[1], w[0] * w[1] * w[2]

    exact, Vdom, ndom = [], [], []
    box = float(np.prod(np.asarray(hi) - lo))
    for c, _, b in F.keys():
        if c == problems.CELL_RING:
            exact.append(math.pi * (problems.R_RING**2 - problems.R_FUEL**2) * h)
            Vdom.append(box)
            ndom.append(n_box)
            continue
        mesh, r = (m_fuel, problems.R_FUEL) if c == problems.CELL_FUEL else (m_outer, problems.R_RING)
        x0, x1, y0, y1, vv = voxel(mesh, b)
        a = vol.circle_rectangle_area(0, 0, r, x0, x1, y0, y1) * h
        exact.append(a if c == problems.CELL_FUEL else vv - a)
        Vdom.append(vv)
        ndom.append(n_vox)
    exact, Vdom, ndom = map(np.asarray, (exact, Vdom, ndom))
    p = exact / Vdom
    sd = Vdom * np.sqrt(np.clip(p * (1 - p), 0, None) / ndom)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    rng = np.random.default_rng(0)
    colors = rng.permutation(max(R.max() + 1, 1))
    ax = axes[0]
    ax.pcolormesh(X, Y, np.where(R >= 0, colors[np.maximum(R, 0)], np.nan),
                  shading='auto', cmap='tab20')
    for mesh, col in ((m_fuel, 'k'), (m_outer, 'w')):
        ll, ur = np.asarray(mesh.lower_left), np.asarray(mesh.upper_right)
        for xv in np.linspace(ll[0], ur[0], mesh.dimension[0] + 1):
            ax.plot([xv, xv], [ll[1], ur[1]], color=col, lw=0.6)
        for yv in np.linspace(ll[1], ur[1], mesh.dimension[1] + 1):
            ax.plot([ll[0], ur[0]], [yv, yv], color=col, lw=0.6)
    ax.set(aspect='equal', xlabel='x [cm]', ylabel='y [cm]',
           title='Regions (colors); fuel mesh (black), outer mesh (white),\n'
                 'unmeshed ring = one region')
    ax = axes[1]
    order = np.argsort(F.cell_id * 10_000 + F.mesh_bin)
    xs = np.arange(len(order))
    ok = sd[order] > 0
    z = np.where(ok, (V[order] - exact[order]) / np.where(ok, sd[order], 1), 0)
    ax.bar(xs, z, color=[{problems.CELL_FUEL: 'C1', problems.CELL_RING: 'C2',
                          problems.CELL_MOD: 'C0'}[c] for c in F.cell_id[order]])
    ax.axhline(0, color='k', lw=0.5)
    for k in (-3, 3):
        ax.axhline(k, color='0.6', ls='--', lw=0.6)
    ax.set(xlabel='region (fuel orange, ring green, outer blue)',
           ylabel='(sampled - exact) / binomial s.d.',
           title='Point-location volumes against closed forms')
    fig.tight_layout()
    _save(fig, out, 'split_meshes.png')


def fig_ce_vs_mg(work, out):
    """Random ray in the CE process against the multigroup run, batch by batch."""
    P = work.pincell()
    with openmc.StatePoint(P['statepoint']) as sp:
        k_mg = np.asarray(sp.k_generation)
    w = P['work'] / 'ce_batches'
    ce = pipeline.random_ray_ce_model(work.model, P['mg_model'], P['field'])
    w.mkdir(parents=True, exist_ok=True)
    ce.export_to_model_xml(w / 'model.xml')
    cwd = os.getcwd()
    os.chdir(w)
    try:
        openmc.lib.init(output=False)
        rmc.sweep_init()
        k_ce = []
        for _ in range(len(k_mg)):
            rmc.sweep_iterate(1)
            k_ce.append(rmc.sweep_regions()['k'])
        rmc.sweep_finalize()
    finally:
        openmc.lib.finalize()
        os.chdir(cwd)
    k_ce = np.asarray(k_ce)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(k_mg, lw=1.5, label='multigroup process')
    axes[0].plot(k_ce, lw=0.8, ls='--', label='CE process, batch by batch')
    axes[0].axvline(RR['inactive'], color='0.6', ls=':')
    axes[0].set(xlabel='batch', ylabel='k', title='Random ray k per batch')
    axes[0].legend()
    axes[1].semilogy(np.maximum(np.abs(k_ce - k_mg) / k_mg, 1e-17))
    axes[1].set(ylim=(1e-17, 1e-10), xlabel='batch',
                title='|k_CE - k_MG| / k_MG (round-off only on several threads;\n'
                      'bitwise 0 on one thread)')
    fig.tight_layout()
    _save(fig, out, 'ce_vs_mg.png')


def fig_segments(work, out):
    """Segment records: direction and chord-length distributions in a fuel
    region, and the track-length volume ratio of every region."""
    P = work.pincell()
    w = P['work'] / 'ce_segments'
    ce = pipeline.random_ray_ce_model(work.model, P['mg_model'], P['field'])
    w.mkdir(parents=True, exist_ok=True)
    ce.export_to_model_xml(w / 'model.xml')
    cwd = os.getcwd()
    os.chdir(w)
    try:
        openmc.lib.init(output=False)
        rmc.sweep_init()
        rmc.sweep_iterate(20)
        rmc.sweep_set_recording(all_regions=True)
        recs, regs = [], None
        for _ in range(10):
            rmc.sweep_iterate(1)
            recs.append(rmc.sweep_records())
            regs = rmc.sweep_regions()
        rmc.sweep_finalize()
    finally:
        openmc.lib.finalize()
        os.chdir(cwd)
    rec = {k: np.concatenate([r[k] for r in recs]) for k in recs[0]}
    D = ce.settings.particles * ce.settings.random_ray['distance_active'] * len(recs)
    lo, hi = np.asarray(ce.settings.random_ray['ray_source'].space.lower_left), \
        np.asarray(ce.settings.random_ray['ray_source'].space.upper_right)
    rho = D / np.prod(hi - lo)              # active length per unit volume
    exact = {(c, i, b): v for c, i, b, v in zip(*vol.pincell_volumes(work.mesh))}
    V = np.array([exact[(c, i, b)] for c, i, b in
                  zip(regs['cell_id'], regs['instance'], regs['mesh_bin'])])
    L = np.bincount(rec['region'], weights=rec['length'], minlength=len(V))

    fuel = np.flatnonzero(regs['cell_id'] == problems.CELL_FUEL)
    r0 = fuel[np.argmax(V[fuel])]
    s = rec['region'] == r0
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    # Weighted by length, as in omega_j: the length-weighted directions are
    # isotropic. (Counts are not: chords along a voxel's axes are longer, so
    # those directions give fewer segments per unit track length.)
    bins = np.linspace(-1, 1, 21)
    ell = rec['length'][s]
    axes[0].hist(rec['u'][s, 2], bins=bins, weights=ell, density=True, alpha=0.7,
                 label='$\\mu = u_z$')
    axes[0].hist(np.arctan2(rec['u'][s, 1], rec['u'][s, 0]) / np.pi, bins=bins,
                 weights=ell, density=True, alpha=0.7, label='$\\varphi / \\pi$')
    axes[0].axhline(0.5, color='k', lw=0.8)
    axes[0].set(title=f'Length-weighted directions of {s.sum()} segments in one\n'
                      'fuel region (isotropic: both flat at 1/2)', xlabel='value')
    axes[0].legend()
    axes[1].hist(rec['length'][s], bins=40)
    axes[1].set(title='Chord lengths in that region', xlabel='length [cm]')
    ratio = L / (rho * V)
    hit = V > 0
    axes[2].hist(ratio[hit], bins=40)
    axes[2].axvline(1, color='k')
    axes[2].set(title='$\\sum_j \\omega_j / 4\\pi = \\sum_j \\ell_j/(\\rho V_R)$ per region\n'
                      '(exact volumes; mean 1)', xlabel='ratio')
    fig.tight_layout()
    _save(fig, out, 'segments.png')


def fig_tracer(work, out):
    """Backward paths in the slab box, with segments colored by material."""
    S = work.slab()
    P = work.pincell()
    L, Y, Z = problems.SLAB_HALF_X, problems.PITCH / 2, problems.HEIGHT / 2
    rng = np.random.default_rng(3)
    with pipeline.ce_session(S['model'], S['mesh'], S['work'] / 'ce'):
        rmc.field_load(S['field'], P['lifts'])
        fuel_index = openmc.lib.materials[problems.MAT_FUEL]._index
        fig, ax = plt.subplots(figsize=(12, 4))
        for xp in (-L, *problems.SLAB_PLANES, L):
            ax.axvline(xp, color='k', lw=1)
        for i in range(1, problems.SLAB_MESH_NX):
            ax.axvline(-L + 2 * L * i / problems.SLAB_MESH_NX, color='0.6', ls=':')
        n_drawn = 0
        for _ in range(6):
            r = rng.uniform([-L, -Y, -Z], [L, Y, Z])
            u = rng.normal(size=3)
            u /= np.linalg.norm(u)
            if abs(u[0]) < 0.25:
                continue
            got = rmc.trace(r, u)
            if len(got['length']) == 0:
                continue
            # Walk the path, reflecting in y and z, to draw it in x-y
            p, d = r.copy(), u.copy()
            for length, m in zip(got['length'], got['material']):
                q = p + length * d
                ax.plot([p[0], q[0]], [p[1], q[1]],
                        color='C3' if m == fuel_index else 'C0', lw=1.2)
                ax.plot(*q[:2], 'k.', ms=3)
                p = q
                for k, W in ((1, Y), (2, Z)):
                    if abs(abs(p[k]) - W) < 1e-9:
                        d[k] = -d[k]
            ax.plot(*r[:2], 'ko', ms=5)
            n_drawn += 1
        rmc.field_free()
    ax.set(xlabel='x [cm]', ylabel='y [cm]', aspect='equal',
           title='Backward paths from the dots (red: fuel, blue: graphite; black dots: '
                 'segment ends; dotted: mesh planes)'
           + ('' if n_drawn else '\nNo segments: CharacteristicTracer::trace is not '
              'implemented yet'))
    _save(fig, out, 'tracer.png')


FIGURES = {'lifts': fig_lifts, 'field_maps': fig_field_maps,
           'split_meshes': fig_split_meshes, 'ce_vs_mg': fig_ce_vs_mg,
           'segments': fig_segments, 'tracer': fig_tracer}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='rmc_figures')
    ap.add_argument('--work', default='rmc_visual_work')
    ap.add_argument('figures', nargs='*',
                    help=f'figures to make, from {list(FIGURES)} (default: all)')
    args = ap.parse_args()
    unknown = set(args.figures) - set(FIGURES)
    if unknown:
        ap.error(f'unknown figures {sorted(unknown)}')
    out = Path(args.out).resolve()
    work = Work(args.work)
    for name in (args.figures or FIGURES):
        FIGURES[name](work, out)


if __name__ == '__main__':
    main()
