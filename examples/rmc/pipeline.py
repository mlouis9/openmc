"""Builds every file of the frozen field for one problem, in order:

1. random ray (multigroup) -> rmc_field.h5, mgxs.h5
2. CE lift tallies         -> rmc_lifts.h5
3. region volumes, twice   -> rmc_volumes.h5 (with the field's normalization)

Steps 1 and 2 run OpenMC as subprocesses. Step 3 runs in this process through
openmc.lib, so the CE model must be initialized first (see ce_session()).
"""
import contextlib
import os
from pathlib import Path

import numpy as np

import openmc
import openmc.lib

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
def ce_session(ce_model, mesh, workdir):
    """Initialize the CE model in this process with openmc.lib, add the
    source-region mesh (the CE model has no tally that would define it), and
    finalize on exit."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    ce_model.export_to_model_xml(workdir / 'model.xml')
    cwd = os.getcwd()
    os.chdir(workdir)
    try:
        openmc.lib.init(['-c'], output=False)
        if mesh.id not in openmc.lib.meshes:
            openmc.lib.RegularMesh._from_python(mesh, None)
        yield
    finally:
        openmc.lib.finalize()
        os.chdir(cwd)


def build_volumes(field_path, lift_path, mesh, out_path, n_per_voxel, seeds=(11, 12),
                  method='cpp'):
    """Step 3, inside ce_session(). method='cpp' uses openmc.lib.rmc;
    'python' uses the reference implementation (slow, for testing)."""
    import openmc.lib.rmc as rmc
    F = fld.FieldFile.load(field_path)
    if method == 'cpp':
        rmc.field_load(field_path, lift_path)          # unnormalized
        V = [rmc.field_region_volumes(n_per_voxel, s) for s in seeds]
        rmc.field_free()
        w = np.prod((np.asarray(mesh.upper_right) - mesh.lower_left) / mesh.dimension)
        counts = [np.rint(v / w * n_per_voxel) for v in V]
    else:
        ref = fld.ReferenceField(F, lift_path, mesh)
        V, counts = zip(*(fld.region_volumes(ref, n_per_voxel, s) for s in seeds))
    scale = fld.field_scale(F, V[0])
    fld.write_volumes(out_path, V[0], V[1], counts[0], counts[1], scale, n_per_voxel, seeds)
    return out_path
