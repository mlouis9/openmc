import sys
from pathlib import Path

import pytest

# Make examples/rmc importable as plain modules: `import problems`
EXAMPLES = Path(__file__).resolve().parents[2] / 'examples' / 'rmc'
sys.path.insert(0, str(EXAMPLES))

import pipeline  # noqa: E402
import problems  # noqa: E402

# Small but converged settings for the graphite pincell. Random ray in graphite
# converges slowly (many scatters per fission generation), so it needs a few
# hundred inactive batches even with few rays.
RR_KWARGS = dict(mgxs_particles=5_000, rays=500, batches=500, inactive=300)
LIFT_KWARGS = dict(particles=5_000, batches=30, inactive=10)


@pytest.fixture
def run_in_tmpdir(monkeypatch, tmp_path):
    """Run each test from its own temporary working directory."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture(scope='session')
def pincell():
    """CE model and source-region mesh of the graphite pincell."""
    return problems.graphite_pincell()


@pytest.fixture(scope='session')
def field_files(pincell, tmp_path_factory):
    """Random ray, lift tallies, and reference volumes, built once per session.

    The volumes come from the Python reference implementation, so that tests
    of the C++ field do not depend on the C++ volume function.
    """
    model, mesh = pincell
    work = tmp_path_factory.mktemp('graphite_pincell')
    field_path, lift_path, sp, mgxs = pipeline.build_mg_files(
        model, mesh, work, rr_kwargs=RR_KWARGS, lift_kwargs=LIFT_KWARGS)
    volume_path = work / 'rmc_volumes.h5'
    with pipeline.ce_session(model, mesh, work / 'ce'):
        pipeline.build_volumes(field_path, lift_path, mesh, volume_path,
                               n_per_voxel=2_000, method='python')
    return dict(work=work, field=field_path, lifts=lift_path, volumes=volume_path,
                statepoint=sp, mgxs=mgxs, rr_kwargs=RR_KWARGS)
