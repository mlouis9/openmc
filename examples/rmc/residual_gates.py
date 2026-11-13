"""Reference quantities for the Stage 4 residual gates, and the residual
histograms compared with them.

- first_collision_reference(): gate G3.5/G4.1's reference. The frozen field
  a_s Q_s + a_f Q_f is the source of analog CE runs (the compiled field
  source in field_source/), and the neutron production of the first
  collision of every source particle is tallied by material and outgoing
  energy (CollisionFilter([1]) x EnergyoutFilter, scores nu-scatter and
  nu-fission). An energy cutoff above every source energy ends each history
  at its first collision, so a run costs one flight per particle.
- field_integral(): the low term, the field integrated over the same bins,
  by exact sums over the fine bins of the lifts (no transport, no noise).
- residual_histogram(): weighted residual records binned the same way, with
  their standard errors.

This module shares no code with the residual sampler beyond the field: the
reference is OpenMC's own transport and tallies.
"""
import math
import os
from pathlib import Path

import h5py
import numpy as np

import openmc

LIB = Path(os.environ.get(
    'RMC_FIELD_SOURCE_LIB',
    Path(__file__).resolve().parent / 'field_source' / 'build' / 'librmc_field_source.so'))


def source_parameters(files, box, a=(1.0, 1.0), E_range=(1e-5, 2e7)):
    """Parameters of the compiled field source; box is (lower_left,
    upper_right) of the field's ray source box."""
    ll, ur = box
    return (f"field={files['field']},lifts={files['lifts']},volumes={files['volumes']},"
            f"a_s={a[0]},a_f={a[1]},E_lo={E_range[0]},E_hi={E_range[1]},"
            + ','.join(f'lo{i}={ll[i]},hi{i}={ur[i]}' for i in range(3)))


def first_collision_reference(model, meshes, files, box, workdir, E_edges,
                              material_ids, particles=100_000, runs=5,
                              a=(1.0, 1.0), k=1.0, threads=None, seed=1):
    """First-collision neutron production of the field source by material and
    outgoing energy bin. meshes are the field's source-region meshes (one
    mesh, or the pairs of mg.source_region_meshes()). Returns {tag: (mean,
    std)} with arrays [material, bin], the fission part divided by k, in the
    field's normalization: OpenMC normalizes tallies per source particle, and
    the compiled source's weights 4 pi V Q / p carry the field's mass.

    Each run is one batch of an eigenvalue calculation whose source is the
    field: fission neutrons go to the fission bank instead of being
    transported (in a fixed-source run they would be, and their own first
    collisions would count, since secondaries from fission start with no
    collisions), (n,xn) secondaries are not created below the energy cutoff,
    and the cutoff ends every history at its first collision. The mean and
    its standard error come from the independent runs.
    """
    import mg
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    results = []
    for i_run in range(runs):
        m = openmc.Model(geometry=model.geometry, materials=model.materials)
        s = openmc.Settings()
        s.run_mode = 'eigenvalue'
        s.particles, s.batches, s.inactive = int(particles), 1, 0
        s.seed = seed + i_run
        s.source = openmc.CompiledSource(library=str(LIB),
                                         parameters=source_parameters(files, box, a))
        s.cutoff = {'energy_neutron': 2.0e7}
        for key in ('free_gas_threshold', 'ptables', 'temperature'):
            val = getattr(model.settings, key)
            if val is not None:
                setattr(s, key, val)
        m.settings = s
        t = openmc.Tally(name='first_collision')
        t.filters = [openmc.MaterialFilter(list(material_ids)),
                     openmc.EnergyoutFilter(list(E_edges)),
                     openmc.CollisionFilter([1])]
        t.scores = ['nu-scatter', 'nu-fission']
        t.estimator = 'analog'
        # The field locates regions on its source-region meshes, which a CE
        # model defines only through a tally
        holders = []
        for i, mesh in enumerate(mg.meshes_of(meshes)):
            h = openmc.Tally(name=f'mesh_holder_{i}')
            h.filters = [openmc.MeshFilter(mesh)]
            h.scores = ['events']
            holders.append(h)
        m.tallies = openmc.Tallies([t] + holders)
        d = workdir / f'run{i_run}'
        sp_path = m.run(cwd=d, output=False, threads=threads)
        with openmc.StatePoint(sp_path) as sp:
            tal = sp.get_tally(name='first_collision')
            results.append(tal.mean.reshape(len(material_ids), len(E_edges) - 1, 2))
    r = np.array(results)
    mean = r.mean(axis=0)
    std = r.std(axis=0, ddof=1) / math.sqrt(runs)
    return {'s': (mean[..., 0], std[..., 0]),
            'f': (mean[..., 1] / k, std[..., 1] / k)}


def total_mass(files, a=(1.0, 1.0)):
    """a_s <1, Q_s> + a_f <1, Q_f> from the field and volume files."""
    with h5py.File(files['field']) as f:
        Q_s, Q_f = f['regions/Q_s'][()], f['regions/Q_f'][()]
        G = int(f.attrs['n_groups'])
    with h5py.File(files['volumes']) as f:
        V, scale = f['V_A'][()], float(f.attrs['scale'])
    return scale * float(V @ (a[0] * Q_s.reshape(-1, G).sum(1) +
                              a[1] * Q_f.reshape(-1, G).sum(1)))


def _fine(files):
    with h5py.File(files['lifts']) as f:
        return f['fine_edges'][()]


def field_integral(rmc, files, regions_by_material, E_edges, tag, emission=None):
    """Integral of a field over (material, energy bin): sum over the regions
    of each material of V_R sum_g Q_{t,R,g} int_bin h_t. emission(region)
    gives Q_{t,R,g} (default: the loaded field's), so the same function
    integrates a sweep's region emission. Exact for bin edges on the fine
    grid; otherwise the fine bin across an edge is split in proportion."""
    fine = _fine(files)
    groups = rmc.field_groups()
    with h5py.File(files['volumes']) as f:
        V = f['V_A'][()]
    mid = 0.5 * (fine[:-1] + fine[1:])
    width = np.diff(fine)
    # Overlap of every fine bin with every output bin
    lo = np.maximum(fine[:-1, None], E_edges[None, :-1])
    hi = np.minimum(fine[1:, None], E_edges[None, 1:])
    overlap = np.clip(hi - lo, 0.0, None)                   # [fine, bins]
    g_of = np.searchsorted(groups, mid, side='right') - 1
    out = np.zeros((len(regions_by_material), len(E_edges) - 1))
    for i, regions in enumerate(regions_by_material):
        for r in regions:
            Q = rmc.field_emission(r, tag) if emission is None else emission(r)
            h = rmc.field_shape(np.full(mid.size, r), mid, tag)
            per_fine = V[r] * Q[np.clip(g_of, 0, len(Q) - 1)] * h      # per eV
            out[i] += per_fine @ overlap
    return out


def residual_histogram(rec, w, material_of, E_edges, n_materials):
    """Sum of weights by (destination material, energy bin) and its standard
    error, treating each sample (a density record with its sampler-route
    particles) as one independent draw."""
    dens = rec['sampler'] == 0
    N = int(dens.sum())
    sample_of = np.cumsum(np.r_[0, dens[:-1]])
    mat = material_of(rec['r'])
    b = np.searchsorted(E_edges, rec['E'], side='right') - 1
    ok = (b >= 0) & (b < len(E_edges) - 1) & (mat >= 0)
    nb = len(E_edges) - 1
    per = np.zeros((N, n_materials * nb))
    np.add.at(per, (sample_of[ok], mat[ok] * nb + b[ok]), w[ok])
    total = per.sum(axis=0)
    sem = per.std(axis=0, ddof=1) * math.sqrt(N)
    return total.reshape(n_materials, nb), sem.reshape(n_materials, nb)
