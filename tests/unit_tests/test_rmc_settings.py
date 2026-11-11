import pytest

import openmc


def test_rmc_field_file_roundtrip(run_in_tmpdir):
    """The option survives export to settings.xml and reading it back."""
    s = openmc.Settings()
    s.random_ray['rmc_field_file'] = 'field.h5'
    s.export_to_xml()
    assert openmc.Settings.from_xml().random_ray['rmc_field_file'] == 'field.h5'


def test_rmc_field_file_type():
    """A non-path value is rejected when the random_ray dictionary is assigned."""
    s = openmc.Settings()
    with pytest.raises(TypeError):
        s.random_ray = {'rmc_field_file': 3}


def test_prescribed_volumes_roundtrip(run_in_tmpdir):
    """The prescribed estimator and its volume file survive export and import."""
    s = openmc.Settings()
    s.random_ray['volume_estimator'] = 'prescribed'
    s.random_ray['source_region_volumes'] = 'volumes.h5'
    s.export_to_xml()
    rr = openmc.Settings.from_xml().random_ray
    assert rr['volume_estimator'] == 'prescribed'
    assert rr['source_region_volumes'] == 'volumes.h5'


def test_source_region_volumes_type():
    s = openmc.Settings()
    with pytest.raises(TypeError):
        s.random_ray = {'source_region_volumes': 3}


def test_rmc_inputs_roundtrip(run_in_tmpdir):
    """The multilevel RMC input files survive export and import."""
    s = openmc.Settings()
    s.rmc = {'field_file': 'field.h5', 'lift_file': 'lifts.h5',
             'volume_file': 'volumes.h5'}
    s.export_to_xml()
    assert openmc.Settings.from_xml().rmc == s.rmc


def test_rmc_inputs_validation():
    s = openmc.Settings()
    with pytest.raises(ValueError):
        s.rmc = {'unknown': 'x.h5'}
    with pytest.raises(TypeError):
        s.rmc = {'field_file': 1}
