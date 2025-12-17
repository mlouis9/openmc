import openmc
import os
import sys

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

def model_sphere():
    """Test input creation for subcritical multiplication run mode."""
    model = openmc.Model()

    universe = create_universe()
    model.geometry = openmc.Geometry(universe)
    model.settings.run_mode = 'subcritical multiplication'

    model.settings.batches = 10
    model.settings.inactive = 5
    model.settings.particles = 1000

    return model

if __name__ == '__main__':
    # Check if we want to just clean
    
    if len(sys.argv) > 1 and sys.argv[1] == 'clean':
        os.remove('model.xml') if os.path.exists('model.xml') else None
        os.remove('materials.xml') if os.path.exists('materials.xml') else None
        os.remove('summary.h5') if os.path.exists('summary.h5') else None
        os.remove('statepoint.10.h5') if os.path.exists('statepoint.10.h5') else None
    else:
        model_sphere().run()
        with openmc.StatePoint('statepoint.10.h5') as sp:
            k_combined = sp.k_combined
            print(sp.__dir__())
            print(sp.keff, sp.k_generation, sp.seed)
            print(sp.run_mode, sp.summary)
            print(f'k-combined: {k_combined}')