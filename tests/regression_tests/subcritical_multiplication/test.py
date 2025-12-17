import openmc
import pytest

from tests.testing_harness import PyAPITestHarness

def create_universe():
    # Define materials
    heu = openmc.Material(name='HEU')
    heu.add_element('U', 1, enrichment=93.0, enrichment_type='wo')
    heu.set_density('g/cm3', 19.1)

    dep_uranium = openmc.Material(name='Depleted Uranium')
    dep_uranium.add_nuclide('U238', 1.0)
    dep_uranium.set_density('g/cm3', 19.1)

    mats = openmc.Materials([heu, dep_uranium])
    mats.export_to_xml()

    # Geometry
    fuel_radius = 4.8
    dep_uranium_radius = 12.2

    fuel_sphere = openmc.Sphere(r=fuel_radius)
    dep_uranium_sphere = openmc.Sphere(r=dep_uranium_radius, boundary_type='vacuum')

    fuel_region = -fuel_sphere
    dep_uranium_region = +fuel_sphere & -dep_uranium_sphere

    fuel_cell = openmc.Cell(name='Fuel', region=fuel_region)
    fuel_cell.fill = heu

    dep_uranium_cell = openmc.Cell(name='Depleted Uranium', region=dep_uranium_region)
    dep_uranium_cell.fill = dep_uranium

    universe = openmc.Universe(cells=[fuel_cell, dep_uranium_cell])
    return universe

@pytest.fixture
def model_random_ray():
    """Test error for random ray subcritical multiplication."""
    model = openmc.Model()

    universe = create_universe()
    model.geometry = openmc.Geometry(universe)
    model.settings.run_mode = 'subcritical multiplication'
    n = 100
    mesh = openmc.RegularMesh()
    mesh.dimension = (n, n, n)
    mesh.lower_left = model.geometry.bounding_box.lower_left
    mesh.upper_right = model.geometry.bounding_box.upper_right
    model.settings.random_ray['source_region_meshes'] = [(mesh, [model.geometry.root_universe])]

    model.settings.batches = 10
    model.settings.inactive = 5
    model.settings.particles = 1000

    return model

@pytest.fixture
def model_sphere_subcritical():
    model = openmc.Model()

    universe = create_universe()
    model.geometry = openmc.Geometry(universe)
    model.settings.run_mode = 'subcritical multiplication'

    model.settings.batches = 10
    model.settings.inactive = 5
    model.settings.particles = 1000

    return model

@pytest.fixture
def model_sphere_eigenvalue():
    model = openmc.Model()

    universe = create_universe()
    model.geometry = openmc.Geometry(universe)
    model.settings.run_mode = 'eigenvalue'

    model.settings.batches = 10
    model.settings.inactive = 5
    model.settings.particles = 1000

    return model

def test_random_ray_fail(model_random_ray):
    harness = PyAPITestHarness("statepoint.10.h5", model_random_ray, inputs_true='inputs_true_1.dat')
    with pytest.raises(RuntimeError, match="random ray solver not currently supported in subcritical multiplication mode"):
        harness.main()

def test_sphere_run(model_sphere_subcritical, model_sphere_eigenvalue):
    """Test that subcritical multiplication and eigenvalue runs give the same digested results."""
    harness_subcritical = PyAPITestHarness("statepoint.10.h5", model_sphere_subcritical, inputs_true='inputs_true_2.dat', 
                               results_true='results_true_sphere.dat')
    harness_subcritical.main()
    harness_eigenvalue = PyAPITestHarness("statepoint.10.h5", model_sphere_eigenvalue, inputs_true='inputs_true_3.dat', 
                               results_true='results_true_sphere.dat')
    harness_eigenvalue.main()