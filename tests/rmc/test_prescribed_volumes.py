"""Prescribed source-region volumes in random ray.

Random ray estimates each source region's volume from the track length the
rays deposit in it. With ``volume_estimator = 'prescribed'`` it instead uses
volumes read from a file (docs/source/usersguide/random_ray.rst, "Prescribed
volumes"). For the graphite pincell those volumes have closed forms
(examples/rmc/volumes.py). The gates:

- the closed forms are right (limits, and agreement with sampling);
- they agree with point-location volumes computed through OpenMC's geometry;
- random ray uses exactly the prescribed volumes (the exported volumes equal
  them to round-off);
- the eigenvalue agrees with a run that estimates the volumes;
- a region that rays reach but the file omits stops the run.
"""
import math

import h5py
import numpy as np
import pytest

import openmc

import field as fld
import mg
import problems
import volumes as vol


# ---------------------------------------------------------------------------
# Closed forms (no OpenMC)
# ---------------------------------------------------------------------------

def test_circle_rectangle_area_limits():
    r = problems.R_FUEL
    full = math.pi * r * r
    assert vol.circle_rectangle_area(0, 0, r, -1, 1, -1, 1) == pytest.approx(full, rel=1e-14)
    assert vol.circle_rectangle_area(0, 0, r, 0, 1, 0, 1) == pytest.approx(full / 4, rel=1e-14)
    assert vol.circle_rectangle_area(0, 0, r, -0.1, 0.1, -0.1, 0.1) == pytest.approx(0.04, rel=1e-14)
    assert vol.circle_rectangle_area(0, 0, r, 0.5, 1, 0.5, 1) == 0.0
    # Off-centre disk: a half-plane cut through its centre
    assert vol.circle_rectangle_area(0.3, -0.2, r, 0.3, 5, -5, 5) == pytest.approx(full / 2, rel=1e-14)


def test_circle_rectangle_area_matches_sampling():
    """Random rectangles straddling the circle, against hit-or-miss sampling."""
    rng = np.random.default_rng(20261111)
    r, n = problems.R_FUEL, 200_000
    z = []
    for _ in range(20):
        x0, y0 = rng.uniform(-0.6, 0.4, 2)
        x1, y1 = x0 + rng.uniform(0.05, 0.4), y0 + rng.uniform(0.05, 0.4)
        A = (x1 - x0) * (y1 - y0)
        exact = vol.circle_rectangle_area(0, 0, r, x0, x1, y0, y1)
        p = exact / A
        if p * (1 - p) < 1e-6:
            continue
        pts = rng.uniform([x0, y0], [x1, y1], (n, 2))
        est = np.mean(np.sum(pts**2, axis=1) < r * r)
        z.append((est - p) / math.sqrt(p * (1 - p) / n))
    z = np.array(z)
    assert len(z) >= 10
    assert np.max(np.abs(z)) < 5
    assert abs(np.mean(z)) < 4 / math.sqrt(len(z))


def test_pincell_volumes_partition_box(pincell):
    """Fuel volumes sum to the cylinder, all volumes to the box."""
    _, mesh = pincell
    cid, inst, b, V = vol.pincell_volumes(mesh)
    h = problems.HEIGHT
    assert np.all(V >= 0) and np.all(inst == 0)
    assert V.sum() == pytest.approx(problems.PITCH**2 * h, rel=1e-13)
    assert V[cid == problems.CELL_FUEL].sum() == pytest.approx(
        math.pi * problems.R_FUEL**2 * h, rel=1e-13)
    n_vox = np.prod(mesh.dimension)
    assert len(V) == 2 * n_vox
    assert len(set(zip(cid.tolist(), b.tolist()))) == len(V)


# ---------------------------------------------------------------------------
# Against OpenMC's geometry
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_pincell_volumes_match_point_location(pincell, field_files):
    """The closed forms agree with the point-location volumes of the frozen
    field (two independent seeds), region by region, within binomial
    statistics. This checks the closed forms, the mesh bin numbering, and
    OpenMC's cell lookup against each other."""
    _, mesh = pincell
    F = fld.FieldFile.load(field_files['field'])
    exact = {(c, i, b): v for c, i, b, v in zip(*vol.pincell_volumes(mesh))}
    V_vox = np.prod(np.asarray(mesh.width))
    with h5py.File(field_files['volumes'], 'r') as f:
        n = int(f.attrs['n_per_voxel'])
        samples = [f['V_A'][()], f['V_B'][()]]
    for V in samples:
        z = []
        for key, v_hat in zip(F.keys(), V):
            p = exact[key] / V_vox
            if p * (1 - p) * n < 5:
                # Nearly empty or nearly full: at most a few points disagree
                assert abs(v_hat - exact[key]) <= 5 * V_vox / n
                continue
            z.append((v_hat - exact[key]) / (V_vox * math.sqrt(p * (1 - p) / n)))
        z = np.array(z)
        assert np.max(np.abs(z)) < 5
        chi2 = np.sum(z**2)
        assert chi2 < len(z) + 4 * math.sqrt(2 * len(z))


# ---------------------------------------------------------------------------
# Random ray with prescribed volumes
# ---------------------------------------------------------------------------

def _volume_file(path, mesh, drop=None):
    cid, inst, b, V = vol.pincell_volumes(mesh)
    if drop is not None:
        keep = ~((cid == drop[0]) & (b == drop[1]))
        cid, inst, b, V = cid[keep], inst[keep], b[keep], V[keep]
    vol.write_source_region_volumes(path, cid, inst, b, V)
    return path


@pytest.fixture(scope='session')
def prescribed_run(pincell, field_files, tmp_path_factory):
    """Random ray with prescribed volumes, same MGXS and settings as the
    field_files run, which estimates its volumes."""
    model, mesh = pincell
    work = tmp_path_factory.mktemp('prescribed')
    vpath = _volume_file(work / 'sr_volumes.h5', mesh)
    rr = {'volume_estimator': 'prescribed', 'source_region_volumes': str(vpath)}
    field_path, sp, _, _ = mg.run_random_ray(
        model, mesh, work / 'rr', mgxs_path=field_files['mgxs'], random_ray=rr,
        **{k: v for k, v in field_files['rr_kwargs'].items()
           if k != 'mgxs_particles'})
    return dict(field=field_path, statepoint=sp, volumes=vpath)


@pytest.mark.slow
def test_prescribed_volumes_are_used_exactly(pincell, prescribed_run):
    """The volumes random ray exports equal the prescribed ones to round-off.
    The export multiplies the solver's volume fraction by the ray source box
    volume, so this checks the whole path from file to flux update."""
    _, mesh = pincell
    F = fld.FieldFile.load(prescribed_run['field'])
    exact = {(c, i, b): v for c, i, b, v in zip(*vol.pincell_volumes(mesh))}
    expected = np.array([exact[k] for k in F.keys()])
    np.testing.assert_allclose(F.volume_rr, expected, rtol=1e-12, atol=0)


def _batch_mean_and_std(sp_path, block=25):
    """Mean active k and its standard error from block means, which absorbs
    the batch-to-batch correlation of random ray's k."""
    with openmc.StatePoint(sp_path) as sp:
        k = np.asarray(sp.k_generation)[sp.n_inactive:]
    nb = len(k) // block
    means = k[:nb * block].reshape(nb, block).mean(axis=1)
    return k.mean(), means.std(ddof=1) / math.sqrt(nb)


@pytest.mark.slow
def test_prescribed_k_agrees_with_estimated_volumes(field_files, prescribed_run):
    """Exact volumes remove the volume estimator's error but should not move
    k beyond statistics in a problem where the estimator is accurate."""
    k_est, s_est = _batch_mean_and_std(field_files['statepoint'])
    k_pre, s_pre = _batch_mean_and_std(prescribed_run['statepoint'])
    assert abs(k_pre - k_est) < 4 * math.hypot(s_est, s_pre), (k_est, k_pre)


@pytest.mark.slow
def test_missing_prescribed_volume_stops_run(pincell, field_files, tmp_path):
    """A region that rays reach must have a prescribed volume. The fuel part
    of the voxel containing the pin centre is always reached."""
    model, mesh = pincell
    nx, ny, _ = mesh.dimension
    centre_bin = nx // 2 + nx * (ny // 2)
    vpath = _volume_file(tmp_path / 'sr_volumes.h5', mesh,
                         drop=(problems.CELL_FUEL, centre_bin))
    rr = {'volume_estimator': 'prescribed', 'source_region_volumes': str(vpath)}
    with pytest.raises(RuntimeError, match='No prescribed volume'):
        mg.run_random_ray(model, mesh, tmp_path / 'rr', mgxs_path=field_files['mgxs'],
                          random_ray=rr, rays=50, batches=3, inactive=1)
