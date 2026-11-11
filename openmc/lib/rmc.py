"""Bindings to the multilevel residual Monte Carlo (RMC) C API.

Experimental and opt-in: import as ``import openmc.lib.rmc``. Requires a
continuous-energy model initialized with :func:`openmc.lib.init`.
"""
from ctypes import c_char_p, c_double, c_int, c_int64, c_uint64, POINTER

import numpy as np

from . import _dll
from .error import _error_handler

_f64p = np.ctypeslib.ndpointer(np.float64, flags='C_CONTIGUOUS')
_i64p = np.ctypeslib.ndpointer(np.int64, flags='C_CONTIGUOUS')

for _name, _args in [
    ('openmc_rmc_field_load', [c_char_p, c_char_p, c_char_p]),
    ('openmc_rmc_field_free', []),
    ('openmc_rmc_field_n_regions', [POINTER(c_int64)]),
    ('openmc_rmc_field_locate', [c_int64, _f64p, _i64p]),
    ('openmc_rmc_field_evaluate', [c_int64, _i64p, _f64p, c_int, _f64p]),
    ('openmc_rmc_field_mass', [c_int, POINTER(c_double)]),
    ('openmc_rmc_field_region_volumes', [c_int64, c_uint64, _f64p]),
    ('openmc_rmc_run_random_ray', []),
]:
    _fn = getattr(_dll, _name)
    _fn.argtypes, _fn.restype, _fn.errcheck = _args, c_int, _error_handler

TAGS = {'s': 0, 'f': 1}


def field_load(field_file, lift_file, volume_file=None):
    """Load the frozen field. Without a volume file it is unnormalized and its
    masses are NaN; that is how region volumes are computed."""
    vol = '' if volume_file is None else str(volume_file)
    _dll.openmc_rmc_field_load(str(field_file).encode(), str(lift_file).encode(),
                               vol.encode())


def field_free():
    """Release the loaded field."""
    _dll.openmc_rmc_field_free()


def field_n_regions():
    n = c_int64()
    _dll.openmc_rmc_field_n_regions(n)
    return n.value


def field_locate(xyz):
    """Region index of each point in an (n, 3) array; -1 where the field is zero."""
    xyz = np.ascontiguousarray(xyz, dtype=np.float64).reshape(-1, 3)
    out = np.empty(len(xyz), dtype=np.int64)
    _dll.openmc_rmc_field_locate(len(xyz), xyz.ravel(), out)
    return out


def field_evaluate(region, E, tag):
    """Q_t per unit volume, energy, and solid angle for (region, energy) pairs."""
    region = np.ascontiguousarray(region, dtype=np.int64)
    E = np.ascontiguousarray(np.broadcast_to(E, region.shape), dtype=np.float64)
    out = np.empty(region.shape, dtype=np.float64)
    _dll.openmc_rmc_field_evaluate(region.size, region.ravel(), E.ravel(),
                                   TAGS[tag], out.ravel())
    return out


def field_mass(tag):
    """Field mass <1, Q_t>."""
    m = c_double()
    _dll.openmc_rmc_field_mass(TAGS[tag], m)
    return m.value


def field_region_volumes(n_per_voxel, seed):
    """Region volumes [cm^3] from points uniform in each voxel."""
    out = np.empty(field_n_regions(), dtype=np.float64)
    _dll.openmc_rmc_field_region_volumes(int(n_per_voxel), int(seed), out)
    return out


def run_random_ray():
    """Run OpenMC's random ray solver in this continuous-energy process.

    Random ray uses the multigroup tables of ``settings.rmc['field_file']``
    and the ``settings.random_ray`` options of the initialized model. The
    solve is a complete power iteration, as in a multigroup random ray run;
    its results are reported through the usual random ray output and, if
    ``random_ray['rmc_field_file']`` is set, a new field file.
    """
    _dll.openmc_rmc_run_random_ray()
