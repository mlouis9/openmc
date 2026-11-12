"""Builds every file of the frozen field for one problem, in order:

1. random ray (multigroup) -> rmc_field.h5, mgxs.h5
2. CE lift tallies         -> rmc_lifts.h5
3. region volumes, twice   -> rmc_volumes.h5 (with the field's normalization)

Steps 1 and 2 run OpenMC as subprocesses. Step 3 runs in this process through
openmc.lib, so the CE model must be initialized first (see ce_session()).
"""
import contextlib
import copy
import os
from pathlib import Path

import numpy as np

import openmc
import openmc.lib
import openmc.lib.rmc  # noqa: F401  (registers the rmc bindings)

import field as fld
import mg


def build_mg_files(ce_model, mesh, workdir, rr_kwargs=None, lift_kwargs=None,
                   per_group=40):
    """Steps 1 and 2. Returns (field path, lift path, statepoint, mgxs path)."""
    workdir = Path(workdir)
    field_path, sp, mgxs, _ = mg.run_random_ray(ce_model, mesh, workdir / 'rr',
                                                **(rr_kwargs or {}))
    F = fld.FieldFile.load(field_path)
    fine = mg.fine_grid(F.edges, per_group)
    lift_sp = mg.run_lift_tallies(ce_model, fine, workdir / 'lift', **(lift_kwargs or {}))
    lift_path = workdir / 'rmc_lifts.h5'
    fld.build_lifts(lift_sp, fine, F.edges, [m.id for m in ce_model.materials], lift_path)
    return field_path, lift_path, sp, mgxs


@contextlib.contextmanager
def ce_session(ce_model, meshes, workdir):
    """Initialize the CE model in this process with openmc.lib, add the
    source-region meshes (the CE model has no tally that would define them),
    and finalize on exit. meshes is one mesh, an iterable of meshes, or the
    (mesh, domains) pairs of mg.source_region_meshes().

    The model is initialized for an ordinary run, not in volume-calculation
    mode (openmc -c): in that mode OpenMC ignores boundary conditions, which
    the backward tracer needs."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    ce_model.export_to_model_xml(workdir / 'model.xml')
    cwd = os.getcwd()
    os.chdir(workdir)
    try:
        openmc.lib.init(output=False)
        for mesh in mg.meshes_of(meshes):
            if mesh.id not in openmc.lib.meshes:
                openmc.lib.RegularMesh._from_python(mesh, None)
        yield
    finally:
        openmc.lib.finalize()
        os.chdir(cwd)


def build_volumes(field_path, lift_path, meshes, out_path, n_per_voxel, seeds=(11, 12),
                  method='cpp', n_box=0):
    """Step 3, inside ce_session(). method='cpp' uses openmc.lib.rmc;
    'python' uses the reference implementation (slow, for testing). n_box
    points in the ray source box sample the regions without a mesh."""
    import openmc.lib.rmc as rmc
    F = fld.FieldFile.load(field_path)
    if method == 'cpp':
        rmc.field_load(field_path, lift_path)          # unnormalized
        V, counts = zip(*(rmc.field_region_volumes(n_per_voxel, s, n_box)
                          for s in seeds))
        rmc.field_free()
    else:
        ref = fld.ReferenceField(F, lift_path, mg.meshes_of(meshes))
        V, counts = zip(*(fld.region_volumes(ref, n_per_voxel, s, n_box)
                          for s in seeds))
    scale = fld.field_scale(F, V[0])
    fld.write_volumes(out_path, V[0], V[1], counts[0], counts[1], scale, n_per_voxel,
                      seeds, n_box)
    return out_path


def random_ray_ce_model(ce_model, mg_model, field_file, export_file='rmc_field_ce.h5'):
    """The CE model, configured to run random ray in the CE process.

    The random ray options, ray count, batches, and seed are copied from the
    multigroup model, so the CE-process solve repeats the multigroup one with
    the field file's tables. The CE-process solve exports its own field file
    (export_file), which is how its results are compared.
    """
    ce = copy.deepcopy(ce_model)
    rr = {k: v for k, v in mg_model.settings.random_ray.items()}
    rr['rmc_field_file'] = export_file
    ce.settings.random_ray = rr
    ce.settings.rmc = {'field_file': str(Path(field_file).resolve())}
    s, m = ce.settings, mg_model.settings
    s.particles, s.batches, s.inactive = m.particles, m.batches, m.inactive
    if m.seed is not None:
        s.seed = m.seed
    ce.tallies = openmc.Tallies()
    return ce


def run_random_ray_in_ce(ce_model, mg_model, field_file, workdir,
                         export_file='rmc_field_ce.h5', threads=None,
                         after=None):
    """Run random ray inside a CE process (openmc.lib) from the field file's
    multigroup tables. Returns the path of the field file it exports.

    threads sets the number of OpenMP threads. after, if given, is called
    with no arguments in the same session once random ray has finished, and
    its return value is returned as well: (path, after()).
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    ce = random_ray_ce_model(ce_model, mg_model, field_file, export_file)
    ce.export_to_model_xml(workdir / 'model.xml')
    cwd = os.getcwd()
    os.chdir(workdir)
    try:
        args = ['-s', str(threads)] if threads else None
        openmc.lib.init(args, output=False)
        openmc.lib.rmc.run_random_ray()
        extra = after() if after is not None else None
    finally:
        openmc.lib.finalize()
        os.chdir(cwd)
    path = workdir / export_file
    return (path, extra) if after is not None else path
