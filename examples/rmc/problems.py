"""Verification problems for residual Monte Carlo (RMC).

Two problems are defined, both 2 cm thick in z with reflective boundaries on
every face, so that random ray, mesh tallies, and mesh sources all see a
bounded box:

infinite_medium
    The graphite pincell's materials homogenized by volume fraction. The
    solution has no spatial structure, so any residual is purely in energy
    and angle.
graphite_pincell
    A UO2 pin in graphite. Adds spatial structure.

Graphite is used without S(alpha,beta) data, and matched_physics() makes
OpenMC treat every target nucleus as stationary. These are consistent CE
physics models, applied identically by RMC and by the reference
Monte Carlo runs, under which stationary-target elastic kinematics are exact.

Every geometry, material, and mesh ID is explicit, because files exchanged
between the multigroup and continuous-energy runs identify objects by ID.
"""
import math

import openmc

# Explicit IDs, in separate ranges per object type.
MAT_FUEL, MAT_MOD = 1, 2
CELL_FUEL, CELL_MOD = 11, 12
SURF_PIN = 31
UNIV_ROOT = 1
MESH_SR = 21

PITCH = 1.26     # cm
R_FUEL = 0.4096  # cm
HEIGHT = 2.0     # cm, reflective in z


def matched_physics(settings):
    """Apply the physics settings shared by every run of these problems.

    OpenMC treats the target as stationary when E >= free_gas_threshold * kT
    and A > 1. With kT ~ 0.025 eV and a data floor of 1e-5 eV, any threshold
    below ~4e-4 makes that true at every tabulated energy; the Python API
    requires a positive value. Unresolved-resonance probability tables are
    disabled so that cross sections are deterministic functions of energy.
    """
    settings.free_gas_threshold = 1e-6
    settings.ptables = False
    settings.temperature = {'default': 294.0, 'method': 'nearest'}


def materials():
    """UO2 fuel (3.1 wt%) and graphite without S(alpha,beta)."""
    fuel = openmc.Material(material_id=MAT_FUEL, name='fuel')
    fuel.add_element('U', 1.0, enrichment=3.1)
    fuel.add_element('O', 2.0)
    fuel.set_density('g/cm3', 10.4)

    mod = openmc.Material(material_id=MAT_MOD, name='graphite')
    mod.add_nuclide('C0', 1.0)  # ENDF/B-VII.1 name; C12/C13 in later libraries
    mod.set_density('g/cm3', 1.7)
    return fuel, mod


def box_region(half_x, half_y, half_z):
    """Region inside a box centred at the origin, reflective on every face."""
    planes = [
        openmc.XPlane(-half_x, boundary_type='reflective'),
        openmc.XPlane(+half_x, boundary_type='reflective'),
        openmc.YPlane(-half_y, boundary_type='reflective'),
        openmc.YPlane(+half_y, boundary_type='reflective'),
        openmc.ZPlane(-half_z, boundary_type='reflective'),
        openmc.ZPlane(+half_z, boundary_type='reflective'),
    ]
    return +planes[0] & -planes[1] & +planes[2] & -planes[3] & +planes[4] & -planes[5]


def source_region_mesh(n_per_side):
    """Regular mesh over the box, n_per_side x n_per_side x 1 voxels."""
    mesh = openmc.RegularMesh(mesh_id=MESH_SR)
    mesh.lower_left = (-PITCH / 2, -PITCH / 2, -HEIGHT / 2)
    mesh.upper_right = (+PITCH / 2, +PITCH / 2, +HEIGHT / 2)
    mesh.dimension = (n_per_side, n_per_side, 1)
    return mesh


def _settings(particles, batches, inactive, fissionable_only):
    s = openmc.Settings()
    s.run_mode = 'eigenvalue'
    s.particles, s.batches, s.inactive = particles, batches, inactive
    kwargs = {'constraints': {'fissionable': True}} if fissionable_only else {}
    s.source = openmc.IndependentSource(
        space=openmc.stats.Box((-PITCH / 2, -PITCH / 2, -HEIGHT / 2),
                               (PITCH / 2, PITCH / 2, HEIGHT / 2)),
        **kwargs)
    matched_physics(s)
    return s


def graphite_pincell(n_per_side=12, particles=10_000, batches=110, inactive=10):
    """UO2 pin in graphite, all faces reflective.

    Returns the CE model and the source-region mesh.
    """
    fuel, mod = materials()
    box = box_region(PITCH / 2, PITCH / 2, HEIGHT / 2)
    pin = openmc.ZCylinder(surface_id=SURF_PIN, r=R_FUEL)
    c_fuel = openmc.Cell(cell_id=CELL_FUEL, fill=fuel, region=-pin & box)
    c_mod = openmc.Cell(cell_id=CELL_MOD, fill=mod, region=+pin & box)
    geom = openmc.Geometry(openmc.Universe(universe_id=UNIV_ROOT, cells=[c_fuel, c_mod]))
    model = openmc.Model(geometry=geom, materials=openmc.Materials([fuel, mod]),
                         settings=_settings(particles, batches, inactive, True))
    return model, source_region_mesh(n_per_side)


def infinite_medium(particles=10_000, batches=110, inactive=10):
    """Graphite-pincell materials homogenized by volume fraction, in a reflective box.

    Returns the CE model and a one-voxel source-region mesh.
    """
    fuel, mod = materials()
    f_fuel = math.pi * R_FUEL**2 / PITCH**2
    mix = openmc.Material.mix_materials([fuel, mod], [f_fuel, 1 - f_fuel], 'vo')
    mix.id = MAT_FUEL
    box = box_region(PITCH / 2, PITCH / 2, HEIGHT / 2)
    cell = openmc.Cell(cell_id=CELL_FUEL, fill=mix, region=box)
    geom = openmc.Geometry(openmc.Universe(universe_id=UNIV_ROOT, cells=[cell]))
    model = openmc.Model(geometry=geom, materials=openmc.Materials([mix]),
                         settings=_settings(particles, batches, inactive, False))
    return model, source_region_mesh(1)
