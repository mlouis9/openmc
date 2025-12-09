import openmc
import os

def test_subcritical_multiplication_run_mode():
    """Test input creation for subcritical multiplication run mode."""
    settings = openmc.Settings()
    settings.run_mode = 'subcritical multiplication'
    settings.export_to_xml()
    
    with open('settings.xml', 'r') as f:
        xml_content = f.read()

    with open('inputs_true.dat', 'r') as f:
        expected_content = f.read()
        
    os.remove('settings.xml')

    assert xml_content == expected_content