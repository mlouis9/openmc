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


def bank_reference(model, bank, workdir, E_edges, material_ids, flux_edges=None,
                   particles=100_000, runs=5, k=1.0, threads=None, seed=1):
    """Gate G5.2/G5.4's reference: the first-collision production and the
    uncollided flux of a signed bank, from OpenMC's own transport.

    The positive and negative parts of the bank are written as two source
    files with site weights |w_i|; OpenMC chooses a site uniformly, so the
    tallies per source particle times N_file are the absolute response. Each part runs as
    in first_collision_reference() (one-batch eigenvalue runs, energy cutoff,
    CollisionFilter([1]) for production); the uncollided flux is tallied with
    a track-length estimator and CollisionFilter([0]) by material and energy.
    Returns {'s', 'f', 'flux': (mean, std)}, signed (positive part minus
    negative part), with the fission production divided by k.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    flux_edges = E_edges if flux_edges is None else flux_edges
    parts = {}
    for sign, name in ((1.0, 'positive'), (-1.0, 'negative')):
        idx = np.flatnonzero(np.sign(bank['w']) == sign)
        if idx.size == 0:
            continue
        # Site weights |w_i|; the tallies are scaled by N_file afterwards.
        # (Weights N_file |w_i| would give the same mean, but in an
        # eigenvalue run a collision banks about wgt nu Sigma_f / Sigma_t
        # fission sites, and weights of hundreds overflow the fission bank.)
        sites = [openmc.SourceParticle(r=tuple(bank['r'][i]), u=tuple(bank['u'][i]),
                                       E=float(bank['E'][i]),
                                       wgt=float(abs(bank['w'][i])))
                 for i in idx]
        src = workdir / f'{name}.h5'
        openmc.write_source_file(sites, src)
        results = []
        for i_run in range(runs):
            m = openmc.Model(geometry=model.geometry, materials=model.materials)
            s = openmc.Settings()
            s.run_mode = 'eigenvalue'
            s.particles, s.batches, s.inactive = int(particles), 1, 0
            s.seed = seed + i_run
            s.source = openmc.FileSource(str(src))
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
            f = openmc.Tally(name='uncollided')
            f.filters = [openmc.MaterialFilter(list(material_ids)),
                         openmc.EnergyFilter(list(flux_edges)),
                         openmc.CollisionFilter([0])]
            f.scores = ['flux']
            f.estimator = 'tracklength'
            m.tallies = openmc.Tallies([t, f])
            sp_path = m.run(cwd=workdir / f'{name}_run{i_run}', output=False,
                            threads=threads)
            with openmc.StatePoint(sp_path) as sp:
                a = sp.get_tally(name='first_collision').mean.reshape(
                    len(material_ids), len(E_edges) - 1, 2)
                b = sp.get_tally(name='uncollided').mean.reshape(
                    len(material_ids), len(flux_edges) - 1)
            results.append((a, b))
        a = idx.size * np.array([x[0] for x in results])
        b = idx.size * np.array([x[1] for x in results])
        parts[name] = (sign, a.mean(0), a.std(0, ddof=1) / math.sqrt(runs),
                       b.mean(0), b.std(0, ddof=1) / math.sqrt(runs))
    out = {}
    for key, j in (('s', 0), ('f', 1)):
        mean = sum(p[0] * p[1][..., j] for p in parts.values())
        std = np.sqrt(sum(p[2][..., j]**2 for p in parts.values()))
        if key == 'f':
            mean, std = mean / k, std / k
        out[key] = (mean, std)
    out['flux'] = (sum(p[0] * p[3] for p in parts.values()),
                   np.sqrt(sum(p[4]**2 for p in parts.values())))
    return out


def children_histogram(children, n_flights, material_of, E_edges, n_materials):
    """Children's weights by (collision material, energy bin) and tag, with
    standard errors from the flights (a flight has at most one child; the
    flights that left none contribute zero)."""
    out = {}
    for t, it in (('s', 0), ('f', 1)):
        m = children['tag'] == it
        mat = material_of(children['r'][m])
        b = np.searchsorted(E_edges, children['E'][m], side='right') - 1
        ok = (b >= 0) & (b < len(E_edges) - 1) & (mat >= 0)
        nb = len(E_edges) - 1
        idx = mat[ok] * nb + b[ok]
        w = children['w'][m][ok]
        total = np.bincount(idx, weights=w, minlength=n_materials * nb)
        sq = np.bincount(idx, weights=(w * n_flights)**2, minlength=n_materials * nb)
        # Per flight x_j = N w_child (0 for the others): sum = mean(x)
        var = (sq / n_flights - total**2) / n_flights
        out[t] = (total.reshape(n_materials, nb),
                  np.sqrt(np.maximum(var, 0.0)).reshape(n_materials, nb))
    return out
