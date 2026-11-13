"""Bindings to the residual Monte Carlo (RMC) C API.

Experimental and opt-in: import as ``import openmc.lib.rmc``. Requires a
continuous-energy model initialized with :func:`openmc.lib.init`.
"""
import math
from ctypes import c_char_p, c_double, c_int, c_int32, c_int64, c_uint64, POINTER

import numpy as np

from . import _dll
from .error import _error_handler

_f64p = np.ctypeslib.ndpointer(np.float64, flags='C_CONTIGUOUS')
_i64p = np.ctypeslib.ndpointer(np.int64, flags='C_CONTIGUOUS')
_i32p = np.ctypeslib.ndpointer(np.int32, flags='C_CONTIGUOUS')

for _name, _args in [
    ('openmc_rmc_field_load', [c_char_p, c_char_p, c_char_p]),
    ('openmc_rmc_field_free', []),
    ('openmc_rmc_field_n_regions', [POINTER(c_int64)]),
    ('openmc_rmc_field_locate', [c_int64, _f64p, _i64p]),
    ('openmc_rmc_field_evaluate', [c_int64, _i64p, _f64p, c_int, _f64p]),
    ('openmc_rmc_field_mass', [c_int, POINTER(c_double)]),
    ('openmc_rmc_field_region_volumes', [c_int64, c_int64, c_uint64, _f64p, _i64p]),
    ('openmc_rmc_run_random_ray', []),
    ('openmc_rmc_sweep_init', []),
    ('openmc_rmc_sweep_iterate', [c_int]),
    ('openmc_rmc_sweep_set_recording', [c_int64, _i64p]),
    ('openmc_rmc_sweep_n_records', [POINTER(c_int64)]),
    ('openmc_rmc_sweep_records', [_i64p, _f64p, _f64p, _f64p]),
    ('openmc_rmc_sweep_n_regions', [POINTER(c_int64), POINTER(c_int)]),
    ('openmc_rmc_sweep_regions', [_i32p, _i32p, _i32p, _i64p, _f64p, _f64p,
                                  POINTER(c_double)]),
    ('openmc_rmc_sweep_finalize', []),
    ('openmc_rmc_fission_density', [c_int, c_double, c_int64, _f64p, _f64p, _f64p]),
    ('openmc_rmc_sample_fission_energy', [c_int, c_double, c_int64, c_uint64, _f64p]),
    ('openmc_rmc_elastic_image', [c_int, c_double, c_int64, _f64p, _f64p, _f64p, _f64p]),
    ('openmc_rmc_sample_elastic', [c_int, c_double, c_int64, _f64p, c_uint64, _f64p,
                                   _f64p]),
    ('openmc_rmc_tabular_law', [c_char_p, c_double, c_int64, _f64p, _f64p, _f64p,
                                c_int64, c_uint64, _f64p]),
    ('openmc_rmc_xs_total', [c_int32, c_int64, _f64p, c_double, c_double, _f64p]),
    ('openmc_rmc_backward_psi', [_f64p, _f64p, c_int, _f64p, c_double, c_double, c_int,
                                 c_double, c_double, c_int64, c_uint64, _f64p,
                                 POINTER(c_int64)]),
    ('openmc_rmc_trace', [_f64p, _f64p, c_int, POINTER(c_int), _f64p, _i32p, _i64p,
                          POINTER(c_int)]),
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


def field_region_volumes(n_per_voxel, seed, n_box=0):
    """Region volumes [cm^3] by locating uniform points.

    Regions on a source-region mesh use n_per_voxel points in every voxel of
    their mesh; regions without one (or outside it) use n_box points in the
    ray source box. Returns (volumes, hit counts).
    """
    n = field_n_regions()
    V = np.empty(n, dtype=np.float64)
    counts = np.empty(n, dtype=np.int64)
    _dll.openmc_rmc_field_region_volumes(int(n_per_voxel), int(n_box), int(seed),
                                         V, counts)
    return V, counts


def run_random_ray():
    """Run OpenMC's random ray solver in this continuous-energy process.

    Random ray uses the multigroup tables of ``settings.rmc['field_file']``
    and the ``settings.random_ray`` options of the initialized model. The
    solve is a complete power iteration, as in a multigroup random ray run;
    its results are reported through the usual random ray output and, if
    ``random_ray['rmc_field_file']`` is set, a new field file.
    """
    _dll.openmc_rmc_run_random_ray()


# ---------------------------------------------------------------------------
# Random ray one batch at a time
# ---------------------------------------------------------------------------

def sweep_init():
    """Start a random ray solve in this CE process, to be driven batch by
    batch with sweep_iterate(). Random ray is the solver type until
    sweep_finalize()."""
    _dll.openmc_rmc_sweep_init()


def sweep_iterate(n=1):
    """Run n batches of the random ray power iteration."""
    _dll.openmc_rmc_sweep_iterate(int(n))


def sweep_set_recording(regions=None, all_regions=False):
    """Record the active segments crossing the given source regions (indices)
    in later batches. all_regions records every region that exists now;
    neither stops recording."""
    if all_regions:
        _dll.openmc_rmc_sweep_set_recording(-1, np.zeros(1, dtype=np.int64))
        return
    r = np.ascontiguousarray(regions if regions is not None else [], dtype=np.int64)
    _dll.openmc_rmc_sweep_set_recording(len(r), r if len(r) else np.zeros(1, dtype=np.int64))


def sweep_records():
    """Segments recorded in the last batch, ordered by ray and position.

    Returns a dict of arrays: region [n], u [n, 3], length [n] (cm), and
    psi_bar [n, G], the chord-averaged 4 pi psi with groups in descending
    energy (random ray's order).
    """
    n = c_int64()
    _dll.openmc_rmc_sweep_n_records(n)
    n = n.value
    G = sweep_regions()['phi'].shape[1]
    region = np.empty(max(n, 1), dtype=np.int64)
    u = np.empty(max(3 * n, 1))
    length = np.empty(max(n, 1))
    psi = np.empty(max(n * G, 1))
    _dll.openmc_rmc_sweep_records(region, u, length, psi)
    return {'region': region[:n], 'u': u[:3 * n].reshape(n, 3),
            'length': length[:n], 'psi_bar': psi[:n * G].reshape(n, G)}


def sweep_regions():
    """Source regions of the solve: keys (cell_id, instance, mesh_id,
    mesh_bin), the volume [cm^3] the solver used, the latest scalar flux
    iterate phi [R, G] (descending energy), and the latest k."""
    n, G = c_int64(), c_int()
    _dll.openmc_rmc_sweep_n_regions(n, G)
    n, G = n.value, G.value
    out = {k: np.empty(max(n, 1), dtype=np.int32)
           for k in ('cell_id', 'instance', 'mesh_id')}
    out['mesh_bin'] = np.empty(max(n, 1), dtype=np.int64)
    out['volume'] = np.empty(max(n, 1))
    phi = np.empty(max(n * G, 1))
    k = c_double()
    _dll.openmc_rmc_sweep_regions(out['cell_id'], out['instance'], out['mesh_id'],
                                  out['mesh_bin'], out['volume'], phi, k)
    out = {key: v[:n] for key, v in out.items()}
    out['phi'] = phi[:n * G].reshape(n, G)
    out['k'] = k.value
    return out


def sweep_finalize():
    """End the batch-by-batch solve and restore the solver type."""
    _dll.openmc_rmc_sweep_finalize()


def trace(r, u, max_segments=10_000):
    """Trace one path with the C++ CharacteristicTracer on the loaded field.

    Returns a dict: length [n] (cm), material [n] (index into the model's
    materials, -1 for void), region [n] (field region or -1), and
    ends_in_vacuum.
    """
    m = int(max_segments)
    n, vac = c_int(), c_int()
    length = np.empty(m)
    material = np.empty(m, dtype=np.int32)
    region = np.empty(m, dtype=np.int64)
    _dll.openmc_rmc_trace(np.ascontiguousarray(r, dtype=float),
                          np.ascontiguousarray(u, dtype=float), m, n, length,
                          material, region, vac)
    k = n.value
    return {'length': length[:k], 'material': material[:k], 'region': region[:k],
            'ends_in_vacuum': bool(vac.value)}


# ---------------------------------------------------------------------------
# Production evaluators, and OpenMC's samplers for comparison
# ---------------------------------------------------------------------------

def _nuclide_index(nuclide):
    import openmc.lib
    return openmc.lib.nuclides[nuclide]._index if isinstance(nuclide, str) else int(nuclide)


def fission_density(nuclide, E_in, E_out):
    """Outgoing-energy density [1/eV] and CDF of one fission neutron of the
    nuclide's first fission reaction at incident energy E_in [eV]."""
    E_out = np.ascontiguousarray(E_out, dtype=float)
    f, F = np.empty_like(E_out), np.empty_like(E_out)
    _dll.openmc_rmc_fission_density(_nuclide_index(nuclide), float(E_in), E_out.size,
                                    E_out, f, F)
    return f, F


def sample_fission_energy(nuclide, E_in, n, seed=1):
    """n fission neutron energies from OpenMC's sample_fission_neutron()."""
    out = np.empty(int(n))
    _dll.openmc_rmc_sample_fission_energy(_nuclide_index(nuclide), float(E_in), int(n),
                                          int(seed), out)
    return out


def elastic_image(nuclide, kT, E, mu0):
    """Stationary-target elastic: incident energy and density p_lab/gamma for
    each outgoing (E, mu0)."""
    E = np.ascontiguousarray(E, dtype=float)
    mu0 = np.ascontiguousarray(mu0, dtype=float)
    E_in, f = np.empty_like(E), np.empty_like(E)
    _dll.openmc_rmc_elastic_image(_nuclide_index(nuclide), float(kT), E.size, E, mu0,
                                  E_in, f)
    return E_in, f


def sample_elastic(nuclide, kT, E_in, seed=1):
    """Outgoing (E, mu0) from OpenMC's elastic_scatter() for each incident
    energy in E_in."""
    E_in = np.ascontiguousarray(E_in, dtype=float)
    E, mu = np.empty_like(E_in), np.empty_like(E_in)
    _dll.openmc_rmc_sample_elastic(_nuclide_index(nuclide), float(kT), E_in.size, E_in,
                                   int(seed), E, mu)
    return E, mu



def tabular_law(path, E_in, E_out, n_samples=0, seed=1):
    """Density, CDF, and samples of a ContinuousTabular law stored in group
    'law' of an HDF5 file (a test seam for the law's evaluate() and cdf())."""
    E_out = np.ascontiguousarray(E_out, dtype=float)
    f, F = np.empty_like(E_out), np.empty_like(E_out)
    x = np.empty(max(int(n_samples), 1))
    _dll.openmc_rmc_tabular_law(str(path).encode(), float(E_in), E_out.size, E_out,
                                f, F, int(n_samples), int(seed), x)
    return f, F, x[:int(n_samples)]


# ---------------------------------------------------------------------------
# Backward characteristic
# ---------------------------------------------------------------------------

TAILS = {'vacuum': 0, 'roulette': 1, 'fixed_length': 2}


def xs_total(material_id, E, kT=294.0 * 8.617333262e-5, density_mult=1.0):
    """Macroscopic total cross section [1/cm] of a material at energies E [eV]."""
    E = np.ascontiguousarray(E, dtype=float)
    out = np.empty_like(E)
    _dll.openmc_rmc_xs_total(int(material_id), E.size, E, float(np.sqrt(kT)),
                             float(density_mult), out)
    return out


def backward_psi(r, u, E, a=(1.0, 1.0), tail='roulette', cutoff=math.exp(-8.0),
                 length=0.0, max_segments=1_000_000, seed=1):
    """CE angular flux at r for incident direction -u, at energies E, due to
    the loaded field a_s Q_s + a_f Q_f (eq. rmc-backward-psi).

    tail is 'vacuum' (to the vacuum boundary; exact), 'roulette' (unbiased,
    with the given cutoff on exp(-tau)), or 'fixed_length' (stop after
    length cm; biased). Returns (psi, number of segments traced).
    """
    E = np.ascontiguousarray(np.atleast_1d(E), dtype=float)
    out = np.empty_like(E)
    n = c_int64()
    _dll.openmc_rmc_backward_psi(np.ascontiguousarray(r, dtype=float),
                                 np.ascontiguousarray(u, dtype=float), E.size, E,
                                 float(a[0]), float(a[1]), TAILS[tail], float(cutoff),
                                 float(length), int(max_segments), int(seed), out, n)
    return out, n.value
