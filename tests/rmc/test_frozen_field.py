"""Tests of the frozen field: the export from random ray, the lifts, and the
continuous-energy implementations (Python reference and C++).

Definitions: docs/source/methods/rmc.rst, methods_rmc_field.
"""
import copy

import h5py
import numpy as np
import pytest

import openmc

import field as fld
import pipeline
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


# ---------------------------------------------------------------------------
# The export
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_export_fission_production(pincell, field_files):
    """The exported flux, volumes, and MGXS reproduce random ray's own
    nu-fission tally, and the exported k is the statepoint's. Checks the flux
    normalization and the group order of the export."""
    model, _ = pincell
    F = fld.FieldFile.load(field_files['field'])
    total = sum(F.volume_rr[i] * F.density_mult[i]
                * F.nu_sigma_f[list(F.mat_ids).index(F.material_id[i])] @ F.phi[i]
                for i in range(F.n_regions) if F.material_id[i] >= 0)
    with openmc.StatePoint(field_files['statepoint']) as sp:
        tally = sp.get_tally(name='rr_nu_fission').mean.sum()
        k = sp.keff.nominal_value
    assert total == pytest.approx(tally, rel=1e-6)
    assert F.k_mg == pytest.approx(k, rel=1e-12)


@pytest.mark.slow
def test_export_emission_independent(pincell, field_files):
    """Q_s and Q_f recomputed from mgxs.h5 with numpy equal the C++ export."""
    model, _ = pincell
    F = fld.FieldFile.load(field_files['field'])
    Q_s, Q_f = fld.recompute_emission(F, field_files['mgxs'],
                                      {m.id: m for m in model.materials})
    np.testing.assert_allclose(F.Q_s, Q_s, rtol=1e-12, atol=0)
    np.testing.assert_allclose(F.Q_f, Q_f, rtol=1e-12, atol=0)


@pytest.mark.slow
def test_export_sanity(pincell, field_files):
    model, mesh = pincell
    F = fld.FieldFile.load(field_files['field'])
    assert np.all(np.isfinite(F.Q_s)) and np.all(F.Q_s >= 0) and np.all(F.Q_f >= 0)
    assert set(F.material_id[F.material_id >= 0]) <= {m.id for m in model.materials}
    assert np.all(np.diff(F.edges) > 0)
    assert set(F.mesh_id) == {mesh.id}
    assert len(set(F.keys())) == F.n_regions


# ---------------------------------------------------------------------------
# Lifts
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_lift_integrals(field_files):
    """Every lift integrates to one over each group (on the part of the group
    covered by the fine grid), and every interior group edge is a fine edge."""
    with h5py.File(field_files['lifts'], 'r') as f:
        fe, ge = f['fine_edges'][()], f['group_edges'][()]
        M = len(f['material_ids'])
        dE = np.diff(fe)
        F = len(dE)
        g_of = np.searchsorted(ge, fe[:-1], side='right') - 1
        assert set(ge[1:]) <= set(fe)
        for key in ('h_s', 'h_f', 'h_phi'):
            h = f[key][()].reshape(M, F)
            assert np.all(h >= 0)
            for m in range(M):
                for g in range(len(ge) - 1):
                    s = g_of == g
                    assert np.sum(h[m, s] * dE[s]) == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# In a CE process: reference field, volumes, and the C++ field
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def ce(pincell, field_files):
    """The CE model initialized in this process with openmc.lib."""
    model, mesh = pincell
    with pipeline.ce_session(model, mesh, field_files['work'] / 'ce_tests'):
        yield


@pytest.fixture(scope='module')
def reference(pincell, field_files, ce):
    _, mesh = pincell
    F = fld.FieldFile.load(field_files['field'])
    with h5py.File(field_files['volumes'], 'r') as f:
        scale = float(f.attrs['scale'])
    return fld.ReferenceField(F, field_files['lifts'], mesh, scale=scale)


def _points(mesh, n, seed):
    rng = np.random.default_rng(seed)
    ll, ur = np.asarray(mesh.lower_left), np.asarray(mesh.upper_right)
    return ll + (ur - ll) * rng.random((n, 3))


@pytest.mark.slow
def test_reference_locator(pincell, reference):
    """Every located region has the material OpenMC finds at the point, and no
    point falls in an undiscovered region."""
    _, mesh = pincell
    F = reference.F
    for x in _points(mesh, 20_000, 1):
        R = reference.locate(x)
        assert R >= 0, f'point {x} is in no exported region'
        cell, _ = openmc.lib.find_cell(x)
        assert F.material_id[R] == cell.fill.id


@pytest.mark.slow
def test_reference_volumes(pincell, field_files):
    """Volumes A and B agree within binomial error; fuel volume is pi r^2 h."""
    _, mesh = pincell
    with h5py.File(field_files['volumes'], 'r') as f:
        VA, VB, n = f['V_A'][()], f['V_B'][()], int(f.attrs['n_per_voxel'])
    F = fld.FieldFile.load(field_files['field'])
    V_vox = np.prod((np.asarray(mesh.upper_right) - mesh.lower_left) / mesh.dimension)
    p = np.clip(VA / V_vox, 1e-12, 1)
    sd = V_vox * np.sqrt(p * (1 - p) * 2 / n)          # sd of VA - VB, one voxel per region
    assert np.all(np.abs(VA - VB) <= 5 * sd + 1e-15)
    fuel = np.sum(VA[F.material_id == problems.MAT_FUEL])
    exact = np.pi * problems.R_FUEL**2 * problems.HEIGHT
    assert fuel == pytest.approx(exact, rel=5 * np.sqrt(exact * V_vox / (n * fuel)))
    assert np.sum(VA) == pytest.approx(np.prod(np.asarray(mesh.upper_right) - mesh.lower_left))


@pytest.fixture(scope='module')
def cpp_field(field_files, ce):
    import openmc.lib.rmc as rmc
    rmc.field_load(field_files['field'], field_files['lifts'], field_files['volumes'])
    yield rmc
    rmc.field_free()


@pytest.mark.slow
def test_cpp_masses(field_files, cpp_field):
    """The C++ field's fission mass is 1/k_mg: unit MG fission production."""
    F = fld.FieldFile.load(field_files['field'])
    assert cpp_field.field_mass('f') == pytest.approx(1.0 / F.k_mg, rel=1e-12)
    assert cpp_field.field_mass('s') > 0


@pytest.mark.slow
def test_cpp_locate_matches_reference(pincell, reference, cpp_field):
    """FrozenField::locate agrees exactly with the Python reference."""
    _, mesh = pincell
    x = _points(mesh, 20_000, 3)
    got = cpp_field.field_locate(x)
    want = np.array([reference.locate(xi) for xi in x])
    bad = np.flatnonzero(got != want)
    assert bad.size == 0, (f'{bad.size} of {len(x)} points disagree; first: '
                           f'x={x[bad[0]]}, C++ {got[bad[0]]}, Python {want[bad[0]]}')


@pytest.mark.slow
def test_cpp_evaluate_matches_reference(pincell, reference, cpp_field):
    """FrozenField::evaluate agrees with the Python reference to round-off,
    including energies on and on either side of every group and fine edge."""
    _, mesh = pincell
    edges = np.unique(np.concatenate([reference.fine, reference.F.edges]))
    E = np.concatenate([edges, np.nextafter(edges, 0), np.nextafter(edges, np.inf),
                        np.geomspace(1e-6, 3e7, 400)])
    regions = np.unique(np.concatenate([[-1], np.arange(reference.F.n_regions)]))
    for tag in ('s', 'f'):
        for R in regions:
            got = cpp_field.field_evaluate(np.full(E.size, R), E, tag)
            want = np.array([reference.evaluate(R, tag, e) for e in E])
            np.testing.assert_allclose(got, want, rtol=1e-14, atol=0,
                                       err_msg=f'tag {tag}, region {R}')


@pytest.mark.slow
def test_cpp_volumes_match_reference(pincell, field_files, cpp_field):
    """openmc_rmc_field_region_volumes agrees with the Python volumes within
    binomial statistics, and its total is the mesh volume."""
    _, mesh = pincell
    n_cpp = 20_000
    V = cpp_field.field_region_volumes(n_cpp, seed=7)
    with h5py.File(field_files['volumes'], 'r') as f:
        VA, n_py = f['V_A'][()], int(f.attrs['n_per_voxel'])
    V_vox = np.prod((np.asarray(mesh.upper_right) - mesh.lower_left) / mesh.dimension)
    p = np.clip(VA / V_vox, 1e-12, 1)
    sd = V_vox * np.sqrt(p * (1 - p) * (1 / n_py + 1 / n_cpp))
    assert np.all(np.abs(V - VA) <= 5 * sd + 1e-15)
    assert np.sum(V) == pytest.approx(np.prod(np.asarray(mesh.upper_right) - mesh.lower_left))
    # Same seed, same answer
    np.testing.assert_array_equal(V, cpp_field.field_region_volumes(n_cpp, seed=7))
