"""Tests for the frozen-field export and its continuous-energy evaluation."""
import copy

import pytest

import problems


def _ids(model):
    g = model.geometry
    cells = {c.id: (c.fill.id if c.fill is not None else None)
             for c in g.get_all_cells().values()}
    mats = {m.id for m in g.get_all_materials().values()}
    return cells, mats


@pytest.mark.slow
def test_ids_stable_through_mg_conversion(run_in_tmpdir):
    """Cell and material IDs, and each cell's fill, are unchanged by conversion
    to multigroup. The field file identifies regions by these IDs, so the MG
    and CE models must agree on them."""
    ce, _ = problems.graphite_pincell(particles=1000)
    mg = copy.deepcopy(ce)
    mg.convert_to_multigroup(method='material_wise', groups='CASMO-8',
                             mgxs_path='mgxs.h5', particles=1000)
    assert _ids(ce) == _ids(mg)
