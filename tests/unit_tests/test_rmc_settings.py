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
