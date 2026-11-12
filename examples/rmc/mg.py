"""Multigroup side of residual Monte Carlo (RMC): MGXS generation, the random-ray solve
that writes the frozen field, and the continuous-energy tallies that define
the within-group lifts (docs/source/methods/rmc.rst, methods_rmc_field)."""
import copy
from pathlib import Path

import numpy as np

import openmc


def source_region_meshes(model, meshes):
    """(mesh, domains) pairs for Settings.random_ray['source_region_meshes'].

    meshes is either one mesh, applied to the root universe, or already a
    list of (mesh, domains) pairs, where domains are materials, cells, or
    universes. Random ray applies at most one mesh to each cell instance.
    """
    if isinstance(meshes, openmc.MeshBase):
        return [(meshes, [model.geometry.root_universe])]
    return [(m, list(d)) for m, d in meshes]


def meshes_of(meshes):
    """The distinct meshes in one mesh, an iterable of meshes, or a list of
    (mesh, domains) pairs."""
    if isinstance(meshes, openmc.MeshBase):
        return [meshes]
    out = []
    for item in meshes:
        m = item[0] if isinstance(item, tuple) else item
        if all(m.id != o.id for o in out):
            out.append(m)
    return out


def run_random_ray(ce_model, mesh, workdir, groups='CASMO-8', mgxs_particles=20_000,
                   rays=2_000, batches=150, inactive=50, field_file='rmc_field.h5',
                   mgxs_path=None, random_ray=None, seed=None, threads=None):
    """Convert to multigroup, run random ray, and export the frozen field.

    mesh is one source-region mesh for the whole geometry, or a list of
    (mesh, domains) pairs (see source_region_meshes()).

    mgxs_path names an existing MGXS library to reuse (it is generated there
    if absent); by default it is generated in workdir. random_ray holds extra
    entries for Settings.random_ray, e.g. the prescribed volume estimator.
    threads sets the number of OpenMP threads (default: OpenMC's default).

    Returns (field file path, statepoint path, mgxs.h5 path, MG model).
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    mgxs_path = Path(mgxs_path) if mgxs_path else workdir / 'mgxs.h5'

    mg = copy.deepcopy(ce_model)
    mg.convert_to_multigroup(method='material_wise', groups=groups,
                             mgxs_path=mgxs_path, particles=mgxs_particles)
    mg.convert_to_random_ray()
    mg.settings.random_ray['source_region_meshes'] = source_region_meshes(mg, mesh)
    mg.settings.random_ray['source_shape'] = 'flat'
    mg.settings.random_ray['rmc_field_file'] = field_file
    for key, value in (random_ray or {}).items():
        mg.settings.random_ray[key] = value
    if seed is not None:
        mg.settings.seed = seed
    mg.settings.particles = rays
    mg.settings.batches = batches
    mg.settings.inactive = inactive

    t = openmc.Tally(name='rr_nu_fission')
    t.filters = [openmc.MeshFilter(meshes_of(mesh)[0])]
    t.scores = ['nu-fission']
    mg.tallies = openmc.Tallies([t])

    sp = mg.run(cwd=workdir, output=False, threads=threads)
    return workdir / field_file, Path(sp), mgxs_path, mg


def fine_grid(group_edges, per_group=60):
    """Log-spaced fine energy grid [eV] containing every group edge."""
    edges = np.asarray(group_edges, dtype=float)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        out.append(np.geomspace(max(a, 1e-5), b, per_group + 1)[:-1])
    return np.concatenate(out + [[edges[-1]]])


def run_lift_tallies(ce_model, fine_edges, workdir, particles=20_000, batches=60,
                     inactive=10):
    """CE eigenvalue run tallying per-material emission and flux spectra.

    The emission spectra define the field's within-group shapes h_s and h_f;
    the flux spectrum is used only for proposals.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    m = copy.deepcopy(ce_model)
    mats = list(m.materials)
    t_e = openmc.Tally(name='lift_emission')
    t_e.filters = [openmc.MaterialFilter(mats), openmc.EnergyoutFilter(fine_edges)]
    t_e.scores = ['nu-scatter', 'nu-fission']
    t_p = openmc.Tally(name='lift_flux')
    t_p.filters = [openmc.MaterialFilter(mats), openmc.EnergyFilter(fine_edges)]
    t_p.scores = ['flux']
    m.tallies = openmc.Tallies([t_e, t_p])
    m.settings.particles, m.settings.batches, m.settings.inactive = particles, batches, inactive
    return Path(m.run(cwd=workdir, output=False))
