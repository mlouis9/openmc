"""Multigroup side of multilevel RMC: MGXS generation, the random-ray solve
that writes the frozen field, and the continuous-energy tallies that define
the within-group lifts (docs/source/methods/rmc.rst, methods_rmc_field)."""
import copy
from pathlib import Path

import numpy as np

import openmc


def run_random_ray(ce_model, mesh, workdir, groups='CASMO-8', mgxs_particles=20_000,
                   rays=2_000, batches=150, inactive=50, field_file='rmc_field.h5'):
    """Convert to multigroup, run random ray, and export the frozen field.

    Returns (field file path, statepoint path, mgxs.h5 path, MG model).
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    mgxs_path = workdir / 'mgxs.h5'

    mg = copy.deepcopy(ce_model)
    mg.convert_to_multigroup(method='material_wise', groups=groups,
                             mgxs_path=mgxs_path, particles=mgxs_particles)
    mg.convert_to_random_ray()
    root = mg.geometry.root_universe
    mg.settings.random_ray['source_region_meshes'] = [(mesh, [root])]
    mg.settings.random_ray['source_shape'] = 'flat'
    mg.settings.random_ray['rmc_field_file'] = field_file
    mg.settings.particles = rays
    mg.settings.batches = batches
    mg.settings.inactive = inactive

    t = openmc.Tally(name='rr_nu_fission')
    t.filters = [openmc.MeshFilter(mesh)]
    t.scores = ['nu-fission']
    mg.tallies = openmc.Tallies([t])

    sp = mg.run(cwd=workdir, output=False)
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
