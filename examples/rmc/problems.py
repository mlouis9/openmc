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
slab_box
    Three planar slabs (fuel, graphite, fuel) between vacuum faces in x,
    reflective in y and z, with a source-region mesh in x whose planes do
    not coincide with the slab planes. Every straight path through it has a
    closed form, which tests the backward tracer exactly.
graphite_pincell_split_meshes
    The graphite pincell with the moderator split by a cylinder into an
    inner ring and an outer region, and a different source-region treatment
    in each cell: a fine mesh on the fuel, none on the ring, and a coarse mesh
    on the outer moderator. Exercises fields with several meshes.

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
CELL_FUEL, CELL_MOD, CELL_RING = 11, 12, 13
SURF_PIN, SURF_RING = 31, 32
UNIV_ROOT = 1
MESH_SR, MESH_FUEL, MESH_OUTER, MESH_SLAB = 21, 22, 23, 24
CELL_SLABS = (41, 42, 43)
SURF_SLABS = (51, 52, 53, 54)
SLAB_HALF_X = 2.0                # cm, vacuum faces at x = +/-2
SLAB_PLANES = (-0.7, 0.3)        # interior slab planes
SLAB_MESH_NX = 4                 # mesh planes at -1, 0, 1

PITCH = 1.26     # cm
R_FUEL = 0.4096  # cm
R_RING = 0.5     # cm, outer radius of the moderator ring (split-mesh problem)
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


def source_region_mesh(n_per_side, mesh_id=MESH_SR):
    """Regular mesh over the box, n_per_side x n_per_side x 1 voxels."""
    mesh = openmc.RegularMesh(mesh_id=mesh_id)
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


def graphite_pincell_split_meshes(n_fuel=4, n_outer=3, particles=10_000,
                                  batches=110, inactive=10):
    """The graphite pincell with three cells and two source-region meshes.

    Cells: fuel (r < R_FUEL), a graphite ring (R_FUEL < r < R_RING), and the
    outer graphite. The fuel gets an n_fuel x n_fuel mesh over the pin's
    bounding square, the outer graphite an n_outer x n_outer mesh over the
    whole box, and the ring no mesh, so it is a single source region.

    Returns the CE model and the (mesh, domains) pairs for random ray.
    """
    fuel, mod = materials()
    box = box_region(PITCH / 2, PITCH / 2, HEIGHT / 2)
    pin = openmc.ZCylinder(surface_id=SURF_PIN, r=R_FUEL)
    ring = openmc.ZCylinder(surface_id=SURF_RING, r=R_RING)
    c_fuel = openmc.Cell(cell_id=CELL_FUEL, fill=fuel, region=-pin & box)
    c_ring = openmc.Cell(cell_id=CELL_RING, fill=mod, region=+pin & -ring & box)
    c_mod = openmc.Cell(cell_id=CELL_MOD, fill=mod, region=+ring & box)
    geom = openmc.Geometry(openmc.Universe(universe_id=UNIV_ROOT,
                                           cells=[c_fuel, c_ring, c_mod]))
    model = openmc.Model(geometry=geom, materials=openmc.Materials([fuel, mod]),
                         settings=_settings(particles, batches, inactive, True))

    m_fuel = openmc.RegularMesh(mesh_id=MESH_FUEL)
    m_fuel.lower_left = (-R_FUEL, -R_FUEL, -HEIGHT / 2)
    m_fuel.upper_right = (R_FUEL, R_FUEL, HEIGHT / 2)
    m_fuel.dimension = (n_fuel, n_fuel, 1)
    m_outer = source_region_mesh(n_outer, MESH_OUTER)
    return model, [(m_fuel, [c_fuel]), (m_outer, [c_mod])]


def slab_box(particles=10_000, batches=110, inactive=10):
    """Fuel | graphite | fuel slabs in x, vacuum at x = +/-SLAB_HALF_X,
    reflective in y and z.

    Returns the CE model and a mesh of SLAB_MESH_NX bins in x over the box.
    """
    fuel, mod = materials()
    xs = [openmc.XPlane(-SLAB_HALF_X, boundary_type='vacuum', surface_id=SURF_SLABS[0]),
          openmc.XPlane(SLAB_PLANES[0], surface_id=SURF_SLABS[1]),
          openmc.XPlane(SLAB_PLANES[1], surface_id=SURF_SLABS[2]),
          openmc.XPlane(SLAB_HALF_X, boundary_type='vacuum', surface_id=SURF_SLABS[3])]
    yz = (+openmc.YPlane(-PITCH / 2, boundary_type='reflective')
          & -openmc.YPlane(PITCH / 2, boundary_type='reflective')
          & +openmc.ZPlane(-HEIGHT / 2, boundary_type='reflective')
          & -openmc.ZPlane(HEIGHT / 2, boundary_type='reflective'))
    fills = (fuel, mod, fuel)
    cells = [openmc.Cell(cell_id=CELL_SLABS[i], fill=fills[i],
                         region=+xs[i] & -xs[i + 1] & yz) for i in range(3)]
    geom = openmc.Geometry(openmc.Universe(universe_id=UNIV_ROOT, cells=cells))
    s = _settings(particles, batches, inactive, True)
    s.source = openmc.IndependentSource(
        space=openmc.stats.Box((-SLAB_HALF_X, -PITCH / 2, -HEIGHT / 2),
                               (SLAB_HALF_X, PITCH / 2, HEIGHT / 2)),
        constraints={'fissionable': True})
    model = openmc.Model(geometry=geom, materials=openmc.Materials([fuel, mod]),
                         settings=s)
    mesh = openmc.RegularMesh(mesh_id=MESH_SLAB)
    mesh.lower_left = (-SLAB_HALF_X, -PITCH / 2, -HEIGHT / 2)
    mesh.upper_right = (SLAB_HALF_X, PITCH / 2, HEIGHT / 2)
    mesh.dimension = (SLAB_MESH_NX, 1, 1)
    return model, mesh


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
