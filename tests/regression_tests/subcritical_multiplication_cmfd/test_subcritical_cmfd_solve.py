"""Verification tests for the subcritical-multiplication CMFD fixed-source solve.

Scope, after several iterations:
  * The SOLVER is validated structurally (A = M - F, phi matches a direct
    solve, source projection orientation) on a single clean in-memory run.
  * ACCELERATION CORRECTNESS is validated by the property that actually
    matters: feedback must not move the answer. That tally comparison is the
    real acceptance test; the reweight is driven by the flux *shape* only.

Two things learned the hard way, encoded here:

  1. Multiple in-memory `openmc.lib` inits in ONE process corrupt cross-section
     temperature state (2nd+ run loads 0 K -> supercritical). The fork's
     finalize() does not reset the temperature globals. So every run that needs
     its own init is launched in a SEPARATE PROCESS (spawn). The structural
     fixture is the first/only in-process init and stays clean.

  2. The CMFD "k" = 1 - 1/M is the *source*-multiplication factor, not the
     fundamental eigenvalue sp.keff, and it is fragile on coarse near-critical
     matrices. It does NOT drive the reweight (shape only), so it is not
     asserted against transport here. Validate the physics via tallies.

Slow tests: `pytest -m slow`. Requires scipy and nuclear data.
"""

import glob
import multiprocessing as mp
import os

import numpy as np
import pytest
import scipy.sparse as sparse
import scipy.sparse.linalg as spla

import openmc
import openmc.cmfd as cmfd


RADIUS = 6.5                                 # subcritical at 294 K (k ~ 0.966)
N = 5
TEMPERATURE = 294.0

FAST = dict(particles=30000, batches=12, inactive=6)
MED = dict(particles=40000, batches=100, inactive=60)
EGRID = [0.0, 5.0e5, 20.0e6]                 # two-group


class CapturingCMFDRun(cmfd.CMFDRun):
    """CMFDRun that records the subcritical solve internals each batch."""

    def __init__(self):
        super().__init__()
        self.captured = []

    def _calc_fission_source(self):
        super()._calc_fission_source()
        dbg = getattr(self, "_subcritical_debug", None)
        if self._subcritical and dbg is not None:
            self.captured.append(dict(
                loss=dbg["loss"].copy(),
                prod=dbg["prod"].copy(),
                A=dbg["A"].copy(),
                b=np.array(dbg["b"], copy=True),
                phi_raw=np.array(dbg["phi_raw"], copy=True),
                M=dbg["M"],
                keff=dbg["keff"],
                cmfd_src=np.array(self._cmfd_src, copy=True),
                indices=tuple(self._indices),
            ))


def build_model(source_xyz=(0.0, 0.0, 0.0), **counts):
    fissile = openmc.Material(name="fissile_cube_mat")
    fissile.add_nuclide("U235", 1.0)
    fissile.set_density("g/cm3", 19.1)
    fissile.temperature = TEMPERATURE
    materials = openmc.Materials([fissile])

    cube = openmc.model.RectangularParallelepiped(
        -RADIUS, RADIUS, -RADIUS, RADIUS, -RADIUS, RADIUS,
        boundary_type="vacuum")
    cell = openmc.Cell(fill=fissile, region=-cube)
    geometry = openmc.Geometry([cell])

    settings = openmc.Settings()
    settings.run_mode = "subcritical multiplication"
    settings.particles = counts["particles"]
    settings.batches = counts["batches"]
    settings.inactive = counts["inactive"]
    settings.embedded_tally_scaling = True
    settings.verbosity = 4
    settings.temperature = {"default": TEMPERATURE, "method": "nearest",
                            "tolerance": 1000.0}
    settings.source = openmc.IndependentSource(
        space=openmc.stats.Point(source_xyz))

    tally = openmc.Tally(name="absorption")
    tally.scores = ["absorption"]
    tallies = openmc.Tallies([tally])

    return openmc.model.Model(geometry, materials, settings, tallies)


def build_cmfd_run(feedback, capturing=True):
    mesh = cmfd.CMFDMesh()
    mesh.lower_left = [-RADIUS, -RADIUS, -RADIUS]
    mesh.upper_right = [RADIUS, RADIUS, RADIUS]
    mesh.dimension = (N, N, N)
    mesh.energy = EGRID

    run = CapturingCMFDRun() if capturing else cmfd.CMFDRun()
    run.mesh = mesh
    run.tally_begin = 1
    run.solver_begin = 1
    run.feedback = feedback
    run.window_type = "expanding"
    run.display = {"entropy": False, "balance": False}
    if capturing:
        run._store_subcritical_debug = True
    return run


def _run_in(dirpath, model, cmfd_run):
    cwd = os.getcwd()
    os.chdir(dirpath)
    try:
        model.export_to_xml()
        cmfd_run.run()
        return os.path.abspath(sorted(glob.glob("statepoint.*.h5"))[-1])
    finally:
        os.chdir(cwd)


def _absorption(sp_path):
    with openmc.StatePoint(sp_path) as sp:
        t = sp.get_tally(name="absorption")
        v = t.get_values(scores=["absorption"]).ravel()
        s = t.get_values(scores=["absorption"], value="std_dev").ravel()
    return float(v.sum()), float(np.sqrt((s ** 2).sum()))


def _physical(caps):
    return [c for c in caps if np.isfinite(c["M"]) and np.isfinite(c["keff"])]


# --------------------------------------------------------------------------
# Process isolation: each in-memory run gets a fresh interpreter so the fork's
# un-reset temperature globals can't leak from one run to the next.
# --------------------------------------------------------------------------

def _absorption_worker(conn, dirpath, source_xyz, counts, feedback):
    try:
        model = build_model(source_xyz=source_xyz, **counts)
        run = build_cmfd_run(feedback=feedback, capturing=False)
        sp = _run_in(dirpath, model, run)
        conn.send(("ok",) + _absorption(sp))
    except Exception as exc:               # pragma: no cover - reported to parent
        import traceback
        conn.send(("err", repr(exc), traceback.format_exc()))
    finally:
        conn.close()


def _absorption_isolated(dirpath, source_xyz, counts, feedback):
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe()
    p = ctx.Process(target=_absorption_worker,
                    args=(child, str(dirpath), source_xyz, counts, feedback))
    p.start()
    msg = parent.recv()
    p.join()
    if msg[0] != "ok":
        raise RuntimeError(f"isolated run failed: {msg[1]}\n{msg[2]}")
    return msg[1], msg[2]


# --------------------------------------------------------------------------
# Structural tests: single clean in-memory run (first/only init in-process)
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def offcenter(tmp_path_factory):
    d = tmp_path_factory.mktemp("cmfd_offcenter")
    model = build_model(source_xyz=(3.0, 0.0, 0.0), **FAST)
    run = build_cmfd_run(feedback=False)      # diagnostic-only: tests the solve
    _run_in(d, model, run)
    phys = _physical(run.captured)
    assert phys, ("no physically-valid subcritical solves captured; check that "
                  "cross sections loaded at 294 K (not 0 K) and the cube is "
                  "subcritical")
    return run


def test_two_group_solve_runs(offcenter):
    """2-group, off-center case that broke Gauss-Seidel completes via the
    direct solve and yields a physical M, k."""
    cap = _physical(offcenter.captured)[-1]
    assert cap["indices"][3] == 2, "expected a two-group solve"
    assert cap["M"] > 1.0 and 0.0 < cap["keff"] < 1.0


def test_source_projection_orientation(offcenter):
    cap = _physical(offcenter.captured)[-1]
    nx, ny, nz, ng = cap["indices"]
    fs_x = cap["cmfd_src"].sum(axis=(1, 2, 3))
    x_centers = -RADIUS + (np.arange(nx) + 0.5) * (2 * RADIUS / nx)
    centroid = float(np.sum(fs_x * x_centers) / np.sum(fs_x))
    assert np.argmax(fs_x) >= nx // 2, (
        f"peak x-cell {np.argmax(fs_x)} not in +x half -- RHS mirrored/transposed")
    assert centroid > 0.5, f"fission-source x-centroid {centroid:.3f} cm not in +x"


def test_A_is_loss_minus_prod(offcenter):
    cap = _physical(offcenter.captured)[-1]
    A = cap["A"].tocsr(); A.sort_indices()
    expected = (cap["loss"] - cap["prod"]).tocsr(); expected.sort_indices()
    assert np.array_equal(A.indptr, expected.indptr)
    assert np.array_equal(A.indices, expected.indices)
    assert np.allclose(A.data, expected.data), "A != (loss - prod)"
    assert cap["prod"].nnz > 0, "no fission coupling present in F"


def test_fixed_source_solver_matches_direct(offcenter):
    n_physical = 0
    for i, cap in enumerate(offcenter.captured):
        A = cap["A"].tocsc()
        b = cap["b"]
        phi_direct = spla.spsolve(A, b)
        rel = np.linalg.norm(cap["phi_raw"] - phi_direct) / np.linalg.norm(phi_direct)
        assert rel < 1e-6, f"batch {i}: solved phi deviates from direct, rel={rel:.2e}"
        if np.isfinite(cap["M"]):
            n_physical += 1
            M_direct = 1.0 + cap["prod"].dot(phi_direct).sum() / b.sum()
            assert abs(cap["M"] - M_direct) / M_direct < 1e-6
            assert cap["M"] > 1.0 and 0.0 < cap["keff"] < 1.0
    assert n_physical > 0, "no physical subcritical solves captured"


# --------------------------------------------------------------------------
# Physics acceptance test: feedback must not move the answer.
# Each in-memory run is isolated in its own process (temperature-state safety).
# --------------------------------------------------------------------------

@pytest.mark.slow
def test_feedback_does_not_bias(tmp_path):
    d_on = tmp_path / "on"; d_on.mkdir()
    v_on, s_on = _absorption_isolated(d_on, (0.0, 0.0, 0.0), MED, feedback=True)

    d_off = tmp_path / "off"; d_off.mkdir()
    v_off, s_off = _absorption_isolated(d_off, (0.0, 0.0, 0.0), MED, feedback=False)

    z = abs(v_on - v_off) / np.sqrt(s_on ** 2 + s_off ** 2)
    assert z < 4.0, (
        f"feedback-on {v_on:.4e}+/-{s_on:.1e} vs feedback-off "
        f"{v_off:.4e}+/-{s_off:.1e} at z={z:.2f}")