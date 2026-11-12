"""Frozen fields whose source regions use several meshes, or none.

Random ray applies at most one source-region mesh to each cell instance, and
different instances may use different meshes or none. The frozen field
follows it: a point is located by its cell instance, then the mesh of that
instance (if any), then the bin in that mesh. Region volumes are sampled per
mesh, voxel by voxel, and in the ray source box for regions without a mesh.

The problem is graphite_pincell_split_meshes (examples/rmc/problems.py):
fuel on a fine mesh over the pin's bounding square, an unmeshed graphite
ring, and outer graphite on a coarse mesh over the box. Every region volume
has a closed form, so the volumes are checked against exact values.
"""
import math

import numpy as np
import pytest

import openmc.lib

import field as fld
import mg
import pipeline
import problems
import volumes as vol

# A short random ray run: the tests are geometric, so the field need not be
# converged, only every region discovered.
SPLIT_RR = dict(rays=400, batches=30, inactive=10)


@pytest.fixture(scope='module')
def split(field_files, tmp_path_factory):
    """Field file of the split-mesh problem, reusing the session's MGXS and
    lifts (both depend only on the materials, which are the same)."""
    model, pairs = problems.graphite_pincell_split_meshes()
    work = tmp_path_factory.mktemp('split_meshes')
    field_path, _, _, _ = mg.run_random_ray(
        model, pairs, work / 'rr', mgxs_path=field_files['mgxs'], **SPLIT_RR)
    return dict(model=model, pairs=pairs, field=field_path,
                lifts=field_files['lifts'], work=work)


@pytest.fixture(scope='module')
def split_ce(split):
    """CE session with both meshes, the C++ field (unnormalized), and the
    Python reference."""
    import openmc.lib.rmc as rmc
    with pipeline.ce_session(split['model'], split['pairs'], split['work'] / 'ce'):
        rmc.field_load(split['field'], split['lifts'])
        F = fld.FieldFile.load(split['field'])
        ref = fld.ReferenceField(F, split['lifts'], mg.meshes_of(split['pairs']))
        yield rmc, ref
        rmc.field_free()


def _exact_volumes(F, pairs):
    """Closed-form volume of every region in the field file."""
    (m_fuel, _), (m_outer, _) = pairs
    h = problems.HEIGHT

    def voxel(mesh, b):
        nx, ny, _ = mesh.dimension
        ll = np.asarray(mesh.lower_left, dtype=float)
        w = (np.asarray(mesh.upper_right, dtype=float) - ll) / mesh.dimension
        ix, iy = b % nx, (b // nx) % ny
        return ll[0] + ix * w[0], ll[0] + (ix + 1) * w[0], \
            ll[1] + iy * w[1], ll[1] + (iy + 1) * w[1]

    V = np.empty(F.n_regions)
    for r, (c, _, b) in enumerate(F.keys()):
        if c == problems.CELL_FUEL:
            x0, x1, y0, y1 = voxel(m_fuel, b)
            V[r] = vol.circle_rectangle_area(0, 0, problems.R_FUEL, x0, x1, y0, y1) * h
        elif c == problems.CELL_RING:
            V[r] = math.pi * (problems.R_RING**2 - problems.R_FUEL**2) * h
        else:
            x0, x1, y0, y1 = voxel(m_outer, b)
            V[r] = ((x1 - x0) * (y1 - y0)
                    - vol.circle_rectangle_area(0, 0, problems.R_RING, x0, x1, y0, y1)) * h
    return V


def _domain_volumes(F, pairs):
    """Volume of the sampling domain of every region: its mesh voxel, or the
    ray source box."""
    sizes = {m.id: float(np.prod((np.asarray(m.upper_right) - m.lower_left)
                                 / m.dimension)) for m, _ in pairs}
    box = float(np.prod(np.asarray(F.box[1]) - F.box[0]))
    return np.array([sizes[m] if (m >= 0 and b >= 0) else box
                     for m, b in zip(F.mesh_id.tolist(), F.mesh_bin.tolist())])


@pytest.mark.slow
def test_export_mesh_assignment(split):
    """Each cell's regions carry its own mesh: fuel on the fine mesh, the ring
    on none (one region, bin 0), the outer graphite on the coarse mesh."""
    F = fld.FieldFile.load(split['field'])
    (m_fuel, _), (m_outer, _) = split['pairs']
    for cell, mesh in ((problems.CELL_FUEL, m_fuel), (problems.CELL_MOD, m_outer)):
        sel = F.cell_id == cell
        assert set(F.mesh_id[sel]) == {mesh.id}
        assert np.all((F.mesh_bin[sel] >= 0) & (F.mesh_bin[sel] < np.prod(mesh.dimension)))
    ring = F.cell_id == problems.CELL_RING
    assert ring.sum() == 1
    assert F.mesh_id[ring][0] == -1 and F.mesh_bin[ring][0] == 0
    np.testing.assert_allclose(F.box[0], [-problems.PITCH / 2, -problems.PITCH / 2,
                                          -problems.HEIGHT / 2])


@pytest.mark.slow
def test_cpp_locate_matches_reference_split_meshes(split, split_ce):
    """FrozenField::locate agrees exactly with the Python reference, and every
    region is found by some point."""
    rmc, ref = split_ce
    lo, hi = (np.asarray(v) for v in ref.F.box)
    x = lo + (hi - lo) * np.random.default_rng(5).random((20_000, 3))
    got = rmc.field_locate(x)
    want = np.array([ref.locate(xi) for xi in x])
    bad = np.flatnonzero(got != want)
    assert bad.size == 0, (f'{bad.size} points disagree; first x={x[bad[0]]}, '
                           f'C++ {got[bad[0]]}, Python {want[bad[0]]}')
    assert np.all(got >= 0)
    # Regions too small to be hit by 20k points are allowed to be missed
    tiny = _exact_volumes(ref.F, split['pairs']) < 5 * np.prod(hi - lo) / len(x)
    assert set(np.flatnonzero(~tiny)) <= set(got.tolist())


def _z_scores(V, exact, V_dom, n_dom):
    p = exact / V_dom
    var = V_dom**2 * p * (1 - p) / n_dom
    ok = var > 1e-12 * V_dom**2 / n_dom
    # Regions with p = 0 or 1 (a voxel wholly inside or outside a cell) are
    # estimated exactly, up to round-off in the closed form
    np.testing.assert_allclose(V[~ok], exact[~ok], rtol=1e-12, atol=1e-15)
    return (V[ok] - exact[ok]) / np.sqrt(var[ok])


@pytest.mark.slow
def test_cpp_volumes_exact_split_meshes(split, split_ce):
    """C++ volumes agree with the closed forms region by region, within
    binomial statistics, for both meshes and the box-sampled ring."""
    rmc, ref = split_ce
    n_vox, n_box = 4_000, 40_000
    V, counts = rmc.field_region_volumes(n_vox, seed=3, n_box=n_box)
    F = ref.F
    exact = _exact_volumes(F, split['pairs'])
    V_dom = _domain_volumes(F, split['pairs'])
    n_dom = np.where(V_dom == float(np.prod(np.asarray(F.box[1]) - F.box[0])),
                     n_box, n_vox)
    z = _z_scores(V, exact, V_dom, n_dom)
    assert np.max(np.abs(z)) < 5
    assert np.sum(z**2) < len(z) + 4 * math.sqrt(2 * len(z))
    np.testing.assert_array_equal(counts, np.rint(V / V_dom * n_dom))
    # Without box points the ring cannot be sampled
    with pytest.raises(Exception, match='n_box'):
        rmc.field_region_volumes(n_vox, seed=3, n_box=0)


@pytest.mark.slow
def test_reference_volumes_exact_split_meshes(split, split_ce):
    """The Python reference volumes agree with the closed forms too."""
    _, ref = split_ce
    n_vox, n_box = 300, 3_000
    V, _ = fld.region_volumes(ref, n_vox, seed=4, n_box=n_box)
    F = ref.F
    exact = _exact_volumes(F, split['pairs'])
    V_dom = _domain_volumes(F, split['pairs'])
    n_dom = np.where(V_dom == float(np.prod(np.asarray(F.box[1]) - F.box[0])),
                     n_box, n_vox)
    z = _z_scores(V, exact, V_dom, n_dom)
    assert np.max(np.abs(z)) < 5


@pytest.mark.slow
def test_tracer_regions_match_materials_split_meshes(split_ce):
    """The backward tracer through the fuel mesh, the unmeshed ring, and the
    outer mesh (all faces reflective): each segment's region, where the field
    has one, is a region of the segment's material, and a path never reaches
    vacuum. (The tracer's exact geometry is tested in test_tracer.py.)"""
    rmc, ref = split_ce
    F = ref.F
    mat_index = {m.id: m._index for m in openmc.lib.materials.values()}
    rng = np.random.default_rng(11)
    lo, hi = (np.asarray(v) for v in F.box)
    n_found = 0
    for _ in range(20):
        u = rng.normal(size=3)
        got = rmc.trace(rng.uniform(lo, hi), u / np.linalg.norm(u), max_segments=300)
        assert not got['ends_in_vacuum']
        assert len(got['length']) == 300
        assert np.all(got['length'] >= 0)
        for m, reg in zip(got['material'], got['region']):
            if reg >= 0:
                n_found += 1
                assert m == mat_index[int(F.material_id[reg])]
    assert n_found > 0.9 * 20 * 300
