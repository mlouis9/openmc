"""The backward characteristic tracer (gate G2.1), rmc::CharacteristicTracer.

The tracer follows a straight path from a destination through the CE
geometry, applying boundary conditions, and splits it into segments by cell
instance and source-region mesh bin, as random ray does. These tests check
it without any statistics:

- In the slab box (problems.slab_box), every path has a closed form: the
  x motion is monotone until a vacuum face, the y and z motions reflect
  between walls, and the segment breaks are the union of the slab planes,
  the mesh planes, and the y and z reflection times. The tracer must return
  the same segments, materials, and field regions, to 1e-9 cm.
- A trace stops after max_segments segments without reaching vacuum.

The tracer on several meshes is tested in test_multiple_meshes.py.
"""
import os

import numpy as np
import pytest

import openmc
import openmc.lib

import field as fld
import mg
import pipeline
import problems

SLAB_RR = dict(rays=200, batches=12, inactive=4)
TOL = 1e-9   # cm


@pytest.fixture(scope='module')
def slab(field_files, tmp_path_factory):
    """Slab box field file (short random ray run; only its regions matter),
    with the C++ field loaded in a CE session."""
    import openmc.lib.rmc as rmc
    model, mesh = problems.slab_box()
    work = tmp_path_factory.mktemp('slab')
    field_path, _, _, _ = mg.run_random_ray(
        model, mesh, work / 'rr', mgxs_path=field_files['mgxs'], **SLAB_RR)
    with pipeline.ce_session(model, mesh, work / 'ce'):
        rmc.field_load(field_path, field_files['lifts'])
        F = fld.FieldFile.load(field_path)
        mat_index = {m.id: m._index for m in openmc.lib.materials.values()}
        yield dict(rmc=rmc, F=F, model=model, mesh=mesh, mat_index=mat_index)
        rmc.field_free()


def exact_segments(r, u):
    """Closed-form segments of the path from r along u in the slab box:
    (lengths, cell IDs, mesh bins)."""
    L, Y, Z = problems.SLAB_HALF_X, problems.PITCH / 2, problems.HEIGHT / 2
    x0, y0, z0 = r
    ux, uy, uz = u
    t_exit = ((L if ux > 0 else -L) - x0) / ux

    breaks = {0.0, t_exit}
    nx = problems.SLAB_MESH_NX
    mesh_planes = [-L + 2 * L * i / nx for i in range(1, nx)]
    for p in list(problems.SLAB_PLANES) + mesh_planes:
        t = (p - x0) / ux
        if 0 < t < t_exit:
            breaks.add(t)
    # Reflections between walls at +/-W: in the unfolded coordinate, walls are
    # at odd multiples of W in the direction of travel
    for c0, uc, W in ((y0, uy, Y), (z0, uz, Z)):
        if uc == 0:
            continue
        s = 1 if uc > 0 else -1
        k = 0
        while True:
            t = (s * W * (2 * k + 1) - c0) / uc
            if t >= t_exit:
                break
            if t > 0:
                breaks.add(t)
            k += 1
    t = np.array(sorted(breaks))
    lengths = np.diff(t)
    xm = x0 + ux * 0.5 * (t[:-1] + t[1:])
    cell = np.array(problems.CELL_SLABS)[np.searchsorted(problems.SLAB_PLANES, xm)]
    bins = np.floor((xm + L) / (2 * L / nx)).astype(int)
    return lengths, cell, bins


def _random_path(rng):
    L, Y, Z = problems.SLAB_HALF_X, problems.PITCH / 2, problems.HEIGHT / 2
    r = rng.uniform([-L, -Y, -Z], [L, Y, Z])
    while True:
        mu = rng.uniform(-1, 1)
        phi = rng.uniform(0, 2 * np.pi)
        s = np.sqrt(1 - mu * mu)
        u = np.array([mu, s * np.cos(phi), s * np.sin(phi)])
        if abs(u[0]) > 0.2:       # keep the number of reflections modest
            return r, u


@pytest.mark.slow
def test_slab_paths_match_closed_form(slab):
    rmc, F = slab['rmc'], slab['F']
    key_to_region = {k: i for i, k in enumerate(F.keys())}
    cell_material = {c.id: c.fill.id for c in slab['model'].geometry.get_all_cells().values()}
    rng = np.random.default_rng(2026)
    for case in range(40):
        r, u = _random_path(rng)
        got = rmc.trace(r, u)
        lengths, cells, bins = exact_segments(r, u)
        msg = f'case {case}: r={r}, u={u}'
        assert got['ends_in_vacuum'], msg
        assert len(got['length']) == len(lengths), msg
        np.testing.assert_allclose(got['length'], lengths, rtol=0, atol=TOL, err_msg=msg)
        want_mat = [slab['mat_index'][cell_material[c]] for c in cells]
        np.testing.assert_array_equal(got['material'], want_mat, err_msg=msg)
        want_region = [key_to_region.get((int(c), 0, int(b)), -1)
                       for c, b in zip(cells, bins)]
        np.testing.assert_array_equal(got['region'], want_region, err_msg=msg)


@pytest.mark.slow
def test_trace_stops_at_max_segments(slab):
    rmc = slab['rmc']
    r, u = _random_path(np.random.default_rng(7))
    full = rmc.trace(r, u)
    assert len(full['length']) > 3
    part = rmc.trace(r, u, max_segments=3)
    assert not part['ends_in_vacuum']
    assert len(part['length']) == 3
    np.testing.assert_array_equal(part['length'], full['length'][:3])


@pytest.mark.slow
def test_start_outside_geometry_is_an_error(slab):
    with pytest.raises(Exception, match='outside the geometry'):
        slab['rmc'].trace([10.0, 0.0, 0.0], [1.0, 0.0, 0.0])
