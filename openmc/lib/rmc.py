"""Bindings to the residual Monte Carlo (RMC) C API.

Experimental and opt-in: import as ``import openmc.lib.rmc``. Requires a
continuous-energy model initialized with :func:`openmc.lib.init`.
"""
import math
from ctypes import (byref, c_char_p, c_double, c_int, c_int32, c_int64, c_uint64,
                    POINTER, Structure)

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
    ('openmc_rmc_field_tally', [c_int64, c_int, c_int, c_double, c_uint64, _f64p,
                                _f64p, _f64p, c_int, _f64p, _f64p]),
    ('openmc_rmc_fission_chi', [c_int32, c_int, c_double, c_int64, _f64p, _f64p]),
    ('openmc_rmc_near_kernel', [c_int64, _f64p, _f64p, _f64p, c_double, _f64p]),
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
    ('openmc_rmc_xs', [c_int32, c_int64, _f64p, c_double, c_double, _f64p, _f64p, _f64p]),
    ('openmc_rmc_channel_production', [c_int32, c_int64, _f64p, c_double, c_double,
                                      _f64p, _f64p]),
    ('openmc_rmc_sample_sampler_route', [c_int32, c_double, _f64p, c_double, c_double,
                                         c_int64, c_uint64, _f64p, _f64p, _f64p, _i32p]),
    ('openmc_rmc_field_emission', [c_int64, c_int, _f64p]),
    ('openmc_rmc_field_shape', [c_int64, _i64p, _f64p, c_int, _f64p]),
    ('openmc_rmc_mg_control', [_f64p, _f64p, c_int64, _f64p, c_int, c_int, c_double,
                               c_double, c_int64, c_uint64, _f64p, _f64p, _f64p,
                               _f64p]),
    ('openmc_rmc_trace', [_f64p, _f64p, c_int, POINTER(c_int), _f64p, _i32p, _i64p,
                          POINTER(c_int)]),
]:
    _fn = getattr(_dll, _name)
    _fn.argtypes, _fn.restype, _fn.errcheck = _args, c_int, _error_handler

# Functions called with explicit ctypes arguments (pointers, structures): no
# argument types, but errors are still raised
for _name in ('openmc_rmc_field_groups', 'openmc_rmc_residual_sample',
              'openmc_rmc_residual_records', 'openmc_rmc_residual_directions',
              'openmc_rmc_residual_weights', 'openmc_rmc_sweep_field',
              'openmc_rmc_sweep_field_region', 'openmc_rmc_bank_create',
              'openmc_rmc_bank_size', 'openmc_rmc_bank_get', 'openmc_rmc_bank_free',
              'openmc_rmc_propagate', 'openmc_rmc_residual_records_aux',
              'openmc_rmc_flight_children', 'openmc_rmc_propagate_aux',
              'openmc_rmc_field_production', 'openmc_rmc_flight_children_incident',
              'openmc_rmc_field_first_flight_rates', 'openmc_rmc_forward_residual_bins',
              'openmc_rmc_grouped_fission_weights'):
    _fn = getattr(_dll, _name)
    _fn.restype, _fn.errcheck = c_int, _error_handler

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


# ---------------------------------------------------------------------------
# Residual samples (Stage 4)
# ---------------------------------------------------------------------------

def xs(material_id, E, kT=294.0 * 8.617333262e-5, density_mult=1.0):
    """Total, elastic, and nu-fission macroscopic cross sections [1/cm]."""
    E = np.ascontiguousarray(E, dtype=float)
    t, el, nf = np.empty_like(E), np.empty_like(E), np.empty_like(E)
    _dll.openmc_rmc_xs(int(material_id), E.size, E, float(np.sqrt(kT)),
                       float(density_mult), t, el, nf)
    return t, el, nf


class ResidualConfig(Structure):
    """Mirror of openmc_rmc_residual_config (include/openmc/rmc/capi.h)."""
    _fields_ = [('directions', c_int), ('n_directions', c_int), ('control', c_int),
                ('n_control_directions', c_int), ('control_energy', c_int),
                ('sampler_route', c_int), ('tail', c_int), ('cutoff', c_double),
                ('length', c_double), ('max_segments', c_int64),
                ('voxel_mode', c_int), ('voxel_weights', POINTER(c_double)),
                ('n_voxel_weights', c_int64), ('weights_per_volume', c_int),
                ('beta', c_double), ('E_min', c_double), ('E_max', c_double),
                ('incident_mode', c_int), ('beta_in', c_double),
                ('Ein_min', c_double), ('Ein_max', c_double),
                ('all_sampler', c_int), ('n_energies', c_int),
                ('common_numbers', c_int), ('sampler_control', c_int),
                ('region_samples', c_int), ('net', c_int),
                ('E_table_edges', POINTER(c_double)), ('n_E_table_edges', c_int64),
                ('E_table', POINTER(c_double)), ('n_E_table', c_int64),
                ('cone_fraction', c_double), ('cone_kappa', c_double),
                ('cone_E0', c_double), ('cone_E1', c_double),
                ('skin_fraction', c_double), ('skin_x', c_double), ('skin_y', c_double),
                ('skin_radius', c_double), ('skin_dmin', c_double),
                ('fixed_r', POINTER(c_double)), ('fixed_E', POINTER(c_double)),
                ('fixed_u', POINTER(c_double)), ('n_fixed', c_int64),
                ('mu_fraction', c_double), ('mu_E_edges', POINTER(c_double)),
                ('n_mu_E_edges', c_int64), ('mu_n_bins', c_int),
                ('mu_table', POINTER(c_double)), ('n_mu_table', c_int64),
                ('flux_control', c_int), ('flux_quadrature', c_int),
                ('flux_control_Emin', c_double), ('flux_control_local_Emax', c_double),
                ('n_outgoing', c_int), ('image_roulette', c_double),
                ('Ein_table_edges', POINTER(c_double)), ('n_Ein_table_edges', c_int64),
                ('Ein_table', POINTER(c_double)), ('n_Ein_table', c_int64)]


DIRECTIONS = {'isotropic': 0, 'sweep': 1}
CONTROL_ENERGY = {'exact': 0, 'same_energy': 1}
VOXELS = {'box': 0, 'volume': 1, 'fissionable': 2, 'tabulated': 3}
INCIDENT = {'log_uniform': 0, 'flux': 1}
TERMS = ('C_s', 'C_f', 'M_s', 'M_f', 'R_s', 'R_f', 'U', 'Z')
SAMPLER_CONTROL = {'none': 0, 'region': 1}
RECORD_KINDS = ('density', 'sampler', 'region')


def residual_sample(tag, n, seed=1, directions='isotropic', n_directions=8,
                    control=False, n_control_directions=0, control_energy='exact',
                    sampler_route=True, tail='roulette', cutoff=math.exp(-8.0),
                    length=0.0, max_segments=10_000_000, voxel='box',
                    voxel_weights=None, weights_per_volume=False, beta=1.0,
                    E_range=(1e-5, 2e7), incident='log_uniform', beta_in=0.1,
                    Ein_range=(1e-5, 2e7), keep_directions=False,
                    all_sampler=False, n_energies=1, common_numbers=False,
                    sampler_control='none', region_samples=1, net=False,
                    E_table=None, cone=None, skin=None, fixed=None, mu_table=None,
                    flux_control=False, flux_quadrature=32, flux_control_Emin=0.0,
                    flux_control_local_Emax=0.0, n_outgoing=1,
                    image_roulette=0.0, Ein_table=None):
    """Residual records of n samples of tag 's' or 'f' with the loaded field.

    The options are those of rmc::ResidualOptions (include/openmc/rmc/
    residual.h). Returns a dict of arrays: r, u [m, 3]; E, density, tag,
    sampler, kind (index into RECORD_KINDS), channel (index into
    CHANNEL_KINDS, -1 for density records), and the terms C_s, C_f, M_s,
    M_f, R_s, R_f, U, Z [m], where m counts the n density-branch records and
    the sampler-route and region-term particles.

    Sampler route options (rmc::SamplerOptions): all_sampler puts every
    channel on the sampler route; n_energies is K, the incident energies per
    direction; common_numbers makes the K draws share a stream;
    sampler_control 'region' is the flux-difference control with
    region_samples forward samples of its region term; net (biased, for
    study) gives one outgoing state the net of the K.
    With keep_directions, also 'ce' and 'mg': the per-direction CE and MG
    estimates (G3.7). E_table = (edges, table[R, n_bins]) replaces the
    emission spectrum of the (1 - beta) part of the outgoing-energy proposal
    by per-region bin probabilities (log-uniform within a bin). cone =
    (fraction, kappa, E0, E1) draws that fraction of the directions from a
    forward cone about the outgoing direction (rmc::ResidualOptions). skin =
    (fraction, x, y, radius, d_min) draws that fraction of box destinations
    at a log-uniform depth inside a z-cylinder (rmc::VoxelOptions). fixed =
    (r[n, 3], E[n], u[n, 3]) evaluates the residual at given destinations
    and outgoing states (density 1; diagnostics). mu_table = (fraction,
    E_edges, table[n_materials, n_E, n_mu]) draws that fraction of the
    directions with learned cosines (rmc::ResidualOptions::mu_*); with
    keep_directions, 'mu' holds every kept direction's cosine.
    flux_control adds the isotropic-flux control of the density route to
    M_s and R_s (use c = 1 in residual_weights).
    """
    w = None
    if voxel_weights is not None:
        w = np.ascontiguousarray(voxel_weights, dtype=float)
    cfg = ResidualConfig(
        DIRECTIONS[directions], int(n_directions), int(bool(control)),
        int(n_control_directions), CONTROL_ENERGY[control_energy],
        int(bool(sampler_route)), TAILS[tail], float(cutoff), float(length),
        int(max_segments), VOXELS[voxel],
        w.ctypes.data_as(POINTER(c_double)) if w is not None else None,
        0 if w is None else w.size, int(bool(weights_per_volume)), float(beta),
        float(E_range[0]), float(E_range[1]), INCIDENT[incident], float(beta_in),
        float(Ein_range[0]), float(Ein_range[1]), int(bool(all_sampler)),
        int(n_energies), int(bool(common_numbers)),
        SAMPLER_CONTROL[sampler_control], int(region_samples), int(bool(net)),
        None, 0, None, 0, *(cone if cone is not None else (0.0, 0.0, 1e4, 1e7)),
        *(skin if skin is not None else (0.0, 0.0, 0.0, 0.0, 1e-5)), None, None, None, 0,
        0.0, None, 0, 0, None, 0, int(bool(flux_control)), int(flux_quadrature),
        float(flux_control_Emin), float(flux_control_local_Emax), int(n_outgoing),
        float(image_roulette), None, 0, None, 0)
    if Ein_table is not None:
        ee = np.ascontiguousarray(Ein_table[0], dtype=float)
        et = np.ascontiguousarray(Ein_table[1], dtype=float).ravel()
        cfg.Ein_table_edges = ee.ctypes.data_as(POINTER(c_double))
        cfg.n_Ein_table_edges = ee.size
        cfg.Ein_table = et.ctypes.data_as(POINTER(c_double))
        cfg.n_Ein_table = et.size
    if mu_table is not None:
        frac, mE, mt = mu_table
        mE = np.ascontiguousarray(mE, dtype=float)
        mt = np.ascontiguousarray(mt, dtype=float)
        cfg.mu_fraction = float(frac)
        cfg.mu_E_edges = mE.ctypes.data_as(POINTER(c_double))
        cfg.n_mu_E_edges = mE.size
        cfg.mu_n_bins = mt.shape[-1]
        flat = mt.ravel()
        cfg.mu_table = flat.ctypes.data_as(POINTER(c_double))
        cfg.n_mu_table = flat.size
    if fixed is not None:
        fr = np.ascontiguousarray(fixed[0], dtype=float).ravel()
        fE = np.ascontiguousarray(fixed[1], dtype=float).ravel()
        fu = np.ascontiguousarray(fixed[2], dtype=float).ravel()
        cfg.fixed_r = fr.ctypes.data_as(POINTER(c_double))
        cfg.fixed_E = fE.ctypes.data_as(POINTER(c_double))
        cfg.fixed_u = fu.ctypes.data_as(POINTER(c_double))
        cfg.n_fixed = fE.size
    if E_table is not None:
        edges = np.ascontiguousarray(E_table[0], dtype=float)
        table = np.ascontiguousarray(E_table[1], dtype=float).ravel()
        cfg.E_table_edges = edges.ctypes.data_as(POINTER(c_double))
        cfg.n_E_table_edges = edges.size
        cfg.E_table = table.ctypes.data_as(POINTER(c_double))
        cfg.n_E_table = table.size
    m = c_int64()
    _dll.openmc_rmc_residual_sample(byref(cfg), TAGS[tag], int(n), int(seed),
                                    int(bool(keep_directions)), byref(m))
    m = m.value
    out = dict(r=np.empty(3 * m), u=np.empty(3 * m), E=np.empty(m),
               density=np.empty(m), tag=np.empty(m, dtype=np.int32),
               sampler=np.empty(m, dtype=np.int32),
               kind=np.empty(m, dtype=np.int32), channel=np.empty(m, dtype=np.int32))
    terms = np.empty(len(TERMS) * max(m, 1))
    p = lambda a: a.ctypes.data_as(POINTER(c_double))
    _dll.openmc_rmc_residual_records(p(out['r']), p(out['u']), p(out['E']),
                                     p(out['density']),
                                     out['tag'].ctypes.data_as(POINTER(c_int)),
                                     out['sampler'].ctypes.data_as(POINTER(c_int)),
                                     out['kind'].ctypes.data_as(POINTER(c_int)),
                                     out['channel'].ctypes.data_as(POINTER(c_int)),
                                     p(terms))
    out['r'] = out['r'].reshape(m, 3)
    out['u'] = out['u'].reshape(m, 3)
    out['E_in'] = np.empty(m)
    out['nuclide'] = np.empty(m, dtype=np.int32)
    _dll.openmc_rmc_residual_records_aux(p(out['E_in']),
                                         out['nuclide'].ctypes.data_as(POINTER(c_int)))
    for i, name in enumerate(TERMS):
        out[name] = terms[i * m:(i + 1) * m].copy()
    if keep_directions:
        k = c_int64()
        _dll.openmc_rmc_residual_directions(byref(k), None, None)
        ce, mg = np.empty(k.value), np.empty(k.value)
        _dll.openmc_rmc_residual_directions(byref(k), p(ce), p(mg))
        out['ce'], out['mg'] = ce, mg
        mu = np.empty(k.value)
        _dll.openmc_rmc_residual_direction_mu(byref(k), _dp(mu) if k.value else None)
        out['mu'] = mu
    return out


def residual_weights(rec, N, a=(1.0, 1.0), b=(1.0, 1.0), k=1.0, c=0.0, lam=0.0):
    """Weights of residual records for amplitudes a, b, eigenvalue k, control
    coefficient c, and mixing lambda (rmc::weight_records)."""
    n = len(rec['E'])
    terms = np.ascontiguousarray(np.concatenate(
        [rec[t] if t in rec else np.zeros(len(rec['E'])) for t in TERMS]))
    if n == 0:
        return np.empty(0)
    w = np.empty(n)
    scalars = np.array([a[0], a[1], b[0], b[1], k, c, lam], dtype=float)
    p = lambda x: x.ctypes.data_as(POINTER(c_double))
    tag = np.ascontiguousarray(rec['tag'], dtype=np.int32)
    smp = np.ascontiguousarray(rec['sampler'], dtype=np.int32)
    _dll.openmc_rmc_residual_weights(
        c_int64(n), tag.ctypes.data_as(POINTER(c_int)),
        smp.ctypes.data_as(POINTER(c_int)), p(terms),
        p(np.ascontiguousarray(rec['density'], dtype=float)), c_int64(int(N)),
        p(scalars), p(w))
    return w


def sweep_field(batch):
    """One random ray transport sweep from Q_s and one from Q_f of the
    loaded field, with the rays of the given batch (needs sweep_init(), a
    few sweep_iterate() batches so that every region exists, and a field
    loaded with volumes). Returns the number of recorded segments."""
    n = c_int64()
    _dll.openmc_rmc_sweep_field(c_int64(int(batch)), byref(n))
    return n.value


def sweep_field_region(region):
    """Segments of the last field sweep in one field region: dict with u
    [n, 3], length [n], psi_s, psi_f [n, G] (ascending groups, per unit solid
    angle), and rho, the sweep's track length per unit volume."""
    rho, n = c_double(), c_int64()
    _dll.openmc_rmc_sweep_field_region(c_int64(int(region)), byref(rho), byref(n),
                                       None, None, None, None)
    m = n.value
    G = len(field_groups()) - 1
    out = dict(u=np.empty(3 * max(m, 1)), length=np.empty(max(m, 1)),
               psi_s=np.empty(max(m, 1) * G), psi_f=np.empty(max(m, 1) * G))
    p = lambda a: a.ctypes.data_as(POINTER(c_double))
    _dll.openmc_rmc_sweep_field_region(c_int64(int(region)), byref(rho), byref(n),
                                       p(out['u']), p(out['length']),
                                       p(out['psi_s']), p(out['psi_f']))
    return dict(u=out['u'][:3 * m].reshape(m, 3), length=out['length'][:m],
                psi_s=out['psi_s'][:m * G].reshape(m, G),
                psi_f=out['psi_f'][:m * G].reshape(m, G), rho=rho.value)


# ---------------------------------------------------------------------------
# Channel registry and sampler route
# ---------------------------------------------------------------------------

CHANNEL_KINDS = ('elastic', 'thermal_elastic', 'inelastic', 'fission')


def channel_production(material_id, E, kT=294.0 * 8.617333262e-5, density_mult=1.0):
    """Production nu_bar Sigma [1/cm] of a material's channels.

    Returns (by_kind, by_route): dicts of arrays over E, keyed by channel
    kind, and by (tag, route) with tags 's', 'f' and routes 'evaluator',
    'sampler'.
    """
    E = np.ascontiguousarray(E, dtype=float)
    k = np.empty(4 * E.size)
    r = np.empty(4 * E.size)
    _dll.openmc_rmc_channel_production(int(material_id), E.size, E, float(np.sqrt(kT)),
                                       float(density_mult), k, r)
    k = k.reshape(-1, 4)
    r = r.reshape(-1, 4)
    by_kind = {name: k[:, i] for i, name in enumerate(CHANNEL_KINDS)}
    by_route = {(t, route): r[:, 2 * it + ir] for it, t in enumerate('sf')
                for ir, route in enumerate(('evaluator', 'sampler'))}
    return by_kind, by_route


def sample_sampler_route(material_id, E_in, u_in, n, seed=1,
                         kT=294.0 * 8.617333262e-5, density_mult=1.0):
    """n draws of the scattering sampler route at (E_in, u_in). Returns a dict:
    production [n], E [n], u [n, 3], kind [n] (index into CHANNEL_KINDS)."""
    u_in = np.ascontiguousarray(u_in, dtype=float)
    out = dict(production=np.empty(n), E=np.empty(n), u=np.empty(3 * n),
               kind=np.empty(n, dtype=np.int32))
    _dll.openmc_rmc_sample_sampler_route(int(material_id), float(E_in), u_in,
                                         float(np.sqrt(kT)), float(density_mult),
                                         int(n), int(seed), out['production'],
                                         out['E'], out['u'], out['kind'])
    out['u'] = out['u'].reshape(n, 3)
    return out


# ---------------------------------------------------------------------------
# Field groups and shapes; multigroup control
# ---------------------------------------------------------------------------

SHAPES = {'s': 0, 'f': 1, 'phi': 2}


def field_groups():
    """Group edges [eV] of the loaded field, ascending."""
    from ctypes import byref
    G = c_int()
    _dll.openmc_rmc_field_groups(byref(G), None)
    edges = np.empty(G.value + 1)
    _dll.openmc_rmc_field_groups(byref(G), edges.ctypes.data_as(POINTER(c_double)))
    return edges


def field_emission(region, tag):
    """Group-integrated emission Q_{t,R,g} per unit volume (ascending g)."""
    Q = np.empty(len(field_groups()) - 1)
    _dll.openmc_rmc_field_emission(int(region), TAGS[tag], Q)
    return Q


def field_shape(region, E, which):
    """Within-group shape h [1/eV] of 's', 'f', or 'phi' at (region, E)."""
    region = np.ascontiguousarray(region, dtype=np.int64)
    E = np.ascontiguousarray(np.broadcast_to(E, region.shape), dtype=float)
    h = np.empty(region.shape)
    _dll.openmc_rmc_field_shape(region.size, region.ravel(), E.ravel(),
                                SHAPES[which], h.ravel())
    return h


def mg_control(r, u_in, E, tag, tail='roulette', cutoff=math.exp(-8.0),
               length=0.0, max_segments=10_000_000, seed=1):
    """MG production q^MG of tag at outgoing energies E, on one MG-only
    backward path for incident direction u_in at r. Returns (q_s, q_f,
    psi_s, psi_f): production from the MG flux of Q_s and of Q_f, and those
    group-integrated angular fluxes."""
    E = np.ascontiguousarray(np.atleast_1d(E), dtype=float)
    G = len(field_groups()) - 1
    q_s, q_f = np.empty(E.size), np.empty(E.size)
    m_s, m_f = np.empty(G), np.empty(G)
    _dll.openmc_rmc_mg_control(np.ascontiguousarray(r, dtype=float),
                               np.ascontiguousarray(u_in, dtype=float), E.size, E,
                               TAGS[tag], TAILS[tail], float(cutoff), float(length),
                               int(max_segments), int(seed), q_s, q_f, m_s, m_f)
    return q_s, q_f, m_s, m_f


# ---------------------------------------------------------------------------
# Signed banks and forward flights
# ---------------------------------------------------------------------------

def _dp(a):
    return a.ctypes.data_as(POINTER(c_double))


class Bank:
    """A signed bank held in C++ memory (rmc::SignedBank), by handle."""

    def __init__(self, r=None, u=None, E=None, w=None, tag=None):
        n = 0 if w is None else len(w)
        r = np.ascontiguousarray(np.zeros((0, 3)) if r is None else r, dtype=float)
        u = np.ascontiguousarray(np.zeros((0, 3)) if u is None else u, dtype=float)
        E = np.ascontiguousarray([] if E is None else E, dtype=float)
        w = np.ascontiguousarray([] if w is None else w, dtype=float)
        t = np.ascontiguousarray([] if tag is None else
                                 [TAGS[x] if isinstance(x, str) else int(x) for x in tag],
                                 dtype=np.int32)
        h = c_int32()
        _dll.openmc_rmc_bank_create(c_int64(n), _dp(r), _dp(u), _dp(E), _dp(w),
                                    t.ctypes.data_as(POINTER(c_int)), byref(h))
        self.handle = h.value

    def __len__(self):
        n = c_int64()
        _dll.openmc_rmc_bank_size(c_int32(self.handle), byref(n))
        return n.value

    def arrays(self):
        """Copies of the bank: dict of r, u [n, 3], E, w, tag [n]."""
        n = len(self)
        out = dict(r=np.empty((n, 3)), u=np.empty((n, 3)), E=np.empty(n),
                   w=np.empty(n), tag=np.empty(n, dtype=np.int32))
        if n:
            _dll.openmc_rmc_bank_get(c_int32(self.handle), _dp(out['r']), _dp(out['u']),
                                     _dp(out['E']), _dp(out['w']),
                                     out['tag'].ctypes.data_as(POINTER(c_int)))
        return out

    def free(self):
        if self.handle is not None:
            _dll.openmc_rmc_bank_free(c_int32(self.handle))
            self.handle = None


def propagate(bank, n, k, seed=1, out=None, tally_materials=None, tally_edges=None):
    """n forward flights from bank; children are appended to out (a new Bank
    by default). Returns (out, stats, flux): stats is a dict of the flight
    counts and W_abs; flux, with tally_materials (IDs) and tally_edges, the
    signed track length [material, energy bin]."""
    if out is None:
        out = Bank()
    mats = np.ascontiguousarray(tally_materials or [], dtype=np.int32)
    edges = np.ascontiguousarray([] if tally_edges is None else tally_edges, dtype=float)
    flux = np.zeros(max(len(mats) * max(len(edges) - 1, 0), 1))
    stats = np.zeros(6)
    _dll.openmc_rmc_propagate(c_int32(bank.handle), c_int64(int(n)), c_double(k),
                              c_uint64(int(seed)), c_int32(out.handle),
                              c_int(len(mats)), mats.ctypes.data_as(POINTER(c_int32)),
                              c_int(len(edges)), _dp(edges), _dp(flux), _dp(stats))
    names = ('flights', 'leaked', 'captured', 'scattered', 'fissions', 'W_abs')
    st = dict(zip(names, stats))
    if len(mats) and len(edges) > 1:
        return out, st, flux[:len(mats) * (len(edges) - 1)].reshape(len(mats), -1)
    return out, st, None


# ---------------------------------------------------------------------------
# Correction tallies and exact cancellation
# ---------------------------------------------------------------------------

def field_tally(n_points, n_E=4, tail='roulette', cutoff=math.exp(-8.0), seed=1, mg=False):
    """The field's CE fission and nu-fission rates by field region and
    source tag ([R, 2]; columns s, f; a = 1, no 1/k), from backward traces
    at n_points uniform points (only those in fissionable material score).
    Returns (fission, nu_fission, fission_sq), fission_sq the sum of squared
    per-point scores (variance of the mean: (sq / n - mean^2) / n). With mg,
    also (nu_fission_mg, diff_sq): the MG nu-fission rate of the MG flux on
    the same paths, and the sum of squares of the per-point difference
    (needs the MG tables, i.e. a field with exported tables)."""
    R = field_n_regions()
    f, nf, sq = np.empty(2 * R), np.empty(2 * R), np.empty(2 * R)
    nm, dq = np.empty(2 * R), np.empty(2 * R)
    _dll.openmc_rmc_field_tally(int(n_points), int(n_E), TAILS[tail], float(cutoff),
                                int(seed), f, nf, sq, int(bool(mg)), nm, dq)
    out = (f.reshape(R, 2), nf.reshape(R, 2), sq.reshape(R, 2))
    if mg:
        out = out + (nm.reshape(R, 2), dq.reshape(R, 2))
    return out


def fission_chi(material_id, nuclide, E_in, E_out):
    """chi(E_out | E_in) [1/eV] of one fission neutron of a nuclide (name or
    data::nuclides index), partial reactions weighted by cross section."""
    E_out = np.ascontiguousarray(E_out, dtype=float)
    out = np.empty_like(E_out)
    _dll.openmc_rmc_fission_chi(int(material_id), _nuclide_index(nuclide), float(E_in),
                                E_out.size, E_out, out)
    return out


def near_kernel(x, xp, E, d_max):
    """Near point kernel of an isotropic parent at xp [n, 3] with energy E
    [n] at x [n, 3], paths shorter than d_max (rmc::near_kernel)."""
    x = np.ascontiguousarray(x, dtype=float).reshape(-1, 3)
    xp = np.ascontiguousarray(xp, dtype=float).reshape(-1, 3)
    E = np.ascontiguousarray(np.broadcast_to(E, len(x)), dtype=float)
    out = np.empty(len(x))
    _dll.openmc_rmc_near_kernel(len(x), x, xp, E, float(d_max), out)
    return out


def field_production(n_points, n_E=4, tail='roulette', cutoff=math.exp(-4.0), seed=1,
                     score='nu-fission', hyperfine=0.0, sigma_e=1.0, classes=16):
    """CE nu-fission production of the field's first-flight flux per source
    tag, X[2], and the variance of each estimate, from backward points drawn
    from the field's own lifted MG fission rate (needs the MG tables). With
    score='loss', the net loss rate instead: absorption less the gain of
    multiplying scatterings, in every material. With hyperfine > 0, energies
    in the groups inside the resonance range (0.5 eV to 200 keV) are drawn
    from bins of that lethargy width instead of the lifts' fine bins: the
    bins are sorted into classes of Sigma_t, a class is chosen with the
    weight W_c [(1 - exp(-S_c d)) / S_c + exp(-S_c d) / sigma_e] for a path
    with d cm of the point's material behind it, and the energy within the
    class by the score's cross section. (On the graphite pincell this
    proposal gave no gain over the lifts' bins.)"""
    X, var = np.zeros(2), np.zeros(2)
    _dll.openmc_rmc_field_production(c_int64(int(n_points)), c_int(int(n_E)),
                                     c_int(TAILS[tail]), c_double(cutoff),
                                     c_uint64(int(seed)), _dp(X), _dp(var),
                                     c_int({'nu-fission': 0, 'loss': 1}[score]),
                                     c_double(hyperfine), c_double(sigma_e), c_int(int(classes)))
    return X, var


def field_first_flight_rates(n_rays, n_E=8, cutoff=math.exp(-4.0), seed=1, du=1e-4,
                             sigma_e=1.0, control=False, eta=None, eta_f=0.0, n_out=1):
    """First-collision rates of the field's emission per source tag, from
    forward rays integrated analytically (no noise from where on the ray
    the collision falls): the nu-fission production X[2] and the net loss
    L[2], and the variance of each. Source points are drawn from the field's
    emission times a rational approximation of the loss probability with
    escape cross section sigma_e, on sub-bins of lethargy width du; n_E
    energies share a ray. Returns (X, L, var_X, var_L)."""
    rate, var, mg, dvar = np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4)
    worth = np.zeros(4)
    eta_a = None if eta is None else np.ascontiguousarray(eta, dtype=np.float64).ravel()
    control = control or eta is not None
    _dll.openmc_rmc_field_first_flight_rates(c_int64(int(n_rays)), c_int(int(n_E)),
                                             c_double(cutoff), c_uint64(int(seed)),
                                             c_double(du), c_double(sigma_e),
                                             _dp(rate), _dp(var), c_int(int(control)),
                                             _dp(mg), _dp(dvar),
                                             None if eta is None else _dp(eta_a),
                                             c_double(eta_f), c_int(int(n_out)), _dp(worth))
    if eta is not None:
        # The importance of the first-collision emission: CE mean, multigroup
        # mean, variance of the CE mean, variance of the mean difference
        return worth
    if control:
        # With control: also the multigroup rates on the same rays and the
        # variance of the mean difference, as (X, L) pairs
        return (rate[[0, 2]], rate[[1, 3]], var[[0, 2]], var[[1, 3]],
                mg[[0, 2]], mg[[1, 3]], dvar[[0, 2]], dvar[[1, 3]])
    return rate[[0, 2]], rate[[1, 3]], var[[0, 2]], var[[1, 3]]


def forward_residual_bins(n_rays, zone, n_zone, n_mu=2, n_phi=4, n_E=8, n_out=1,
                          cutoff=math.exp(-4.0), seed=1):
    """The field's residual in a bin basis from forward rays: (diff, low),
    each [2 source tags, n_bins], n_bins = 2 * n_zone * G * n_mu * n_phi with
    bin = ((tag * n_zone + zone) * G + group) * D + direction. diff is the
    CE deposit of the first-collision emission minus the multigroup deposit
    on the same rays, low the multigroup deposit. Tag f deposits are
    productions (without 1 / k)."""
    zone = np.ascontiguousarray(zone, dtype=np.int32)
    nb = 2 * int(n_zone) * (len(field_groups()) - 1) * n_mu * n_phi
    diff, low = np.zeros(2 * nb), np.zeros(2 * nb)
    _dll.openmc_rmc_forward_residual_bins(c_int64(int(n_rays)), c_int(int(n_E)),
                                          c_int(int(n_out)), c_double(cutoff),
                                          c_uint64(int(seed)),
                                          zone.ctypes.data_as(POINTER(c_int32)),
                                          c_int(int(n_zone)), c_int(n_mu), c_int(n_phi),
                                          _dp(diff), _dp(low))
    return diff.reshape(2, nb), low.reshape(2, nb)


def grouped_fission_weights(x, nuclide, E_f, offset, member, records, d_max, k):
    """Grouped cancellation at fission births (rmc::grouped_fission_weight):
    for each candidate (site x[n, 3], fissioning nuclide, emitted energy) and
    its group member[offset[c]:offset[c+1]] of emission records, the weight
    sum J_j / sum m_j q_j. records: dict of r [M, 3], kind, E, u [M, 3],
    nuclide, w, m. Returns (W, ok)."""
    x = np.ascontiguousarray(x, dtype=float)
    n = len(E_f)
    nuclide = np.ascontiguousarray(nuclide, dtype=np.int32)
    E_f = np.ascontiguousarray(E_f, dtype=float)
    offset = np.ascontiguousarray(offset, dtype=np.int64)
    member = np.ascontiguousarray(member, dtype=np.int64)
    rr = np.ascontiguousarray(records['r'], dtype=float)
    rk = np.ascontiguousarray(records['kind'], dtype=np.int32)
    rE = np.ascontiguousarray(records['E'], dtype=float)
    ru = np.ascontiguousarray(records['u'], dtype=float)
    rn = np.ascontiguousarray(records['nuclide'], dtype=np.int32)
    rw = np.ascontiguousarray(records['w'], dtype=float)
    rm = np.ascontiguousarray(records['m'], dtype=float)
    W = np.zeros(n)
    ok = np.zeros(n, dtype=np.int32)
    ip = lambda a: a.ctypes.data_as(POINTER(c_int))
    lp = lambda a: a.ctypes.data_as(POINTER(c_int64))
    _dll.openmc_rmc_grouped_fission_weights(
        c_int64(n), _dp(x), ip(nuclide), _dp(E_f), lp(offset), lp(member),
        c_int64(len(rw)), _dp(rr), ip(rk), _dp(rE), _dp(ru), ip(rn), _dp(rw), _dp(rm),
        c_double(d_max), c_double(k), _dp(W), ip(ok))
    return W, ok.astype(bool)


def propagate_aux(bank, n, k, seed=1, out=None, children=True, region_tally=True,
                  history=False):
    """Flights as propagate(), also returning a dict with the children's
    parent index, path length, and nuclide (if children), and the collision
    estimates of the fission and nu-fission rates by field region [R]. With
    history, each flight continues through its scatterings until capture,
    fission, or leakage, and only fission children are returned."""
    if out is None:
        out = Bank()
    R = field_n_regions()
    rf, rnf = np.zeros(max(R, 1)), np.zeros(max(R, 1))
    stats = np.zeros(10)
    nc = c_int64()
    _dll.openmc_rmc_propagate_aux(c_int32(bank.handle), c_int64(int(n)), c_double(k),
                                  c_uint64(int(seed)), c_int32(out.handle),
                                  int(bool(children)), int(bool(region_tally)),
                                  int(bool(history)), byref(nc), _dp(rf), _dp(rnf), _dp(stats))
    m = nc.value
    aux = dict(parent=np.empty(m, dtype=np.int64), path=np.empty(m),
               nuclide=np.empty(m, dtype=np.int32))
    if children:
        _dll.openmc_rmc_flight_children(aux['parent'].ctypes.data_as(POINTER(c_int64)),
                                        _dp(aux['path']),
                                        aux['nuclide'].ctypes.data_as(POINTER(c_int)))
        aux['E_in'] = np.empty(m)
        aux['u_in'] = np.empty((m, 3))
        aux['mt'] = np.empty(m, dtype=np.int32)
        _dll.openmc_rmc_flight_children_incident(
            _dp(aux['E_in']), _dp(aux['u_in']), aux['mt'].ctypes.data_as(POINTER(c_int)))
    aux['region_fission'] = rf[:R]
    aux['region_nu_fission'] = rnf[:R]
    names = ('flights', 'leaked', 'captured', 'scattered', 'fissions', 'W_abs',
             'collisions', 'absorption', 'multiplied', 'nu_fission')
    return out, dict(zip(names, stats)), aux
