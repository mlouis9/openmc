"""The compiled field source: its weighted samples reproduce the field's mass
per voxel and group (docs/source/methods/rmc.rst, methods_rmc_field_source)."""
import os
from pathlib import Path

import h5py
import numpy as np
import pytest

import openmc
import openmc.lib

import field as fld
import pipeline

LIB = Path(os.environ.get(
    'RMC_FIELD_SOURCE_LIB',
    Path(__file__).resolve().parents[2] / 'examples' / 'rmc' / 'field_source'
    / 'build' / 'librmc_field_source.so'))


def source_parameters(field_files, mesh, a_s=1.0, a_f=1.0, E_lo=1e-5, E_hi=2e7):
    ll, ur = mesh.lower_left, mesh.upper_right
    return (f"field={field_files['field']},lifts={field_files['lifts']},"
            f"volumes={field_files['volumes']},a_s={a_s},a_f={a_f},"
            f"E_lo={E_lo},E_hi={E_hi},"
            + ','.join(f'lo{i}={ll[i]},hi{i}={ur[i]}' for i in range(3)))


@pytest.mark.slow
@pytest.mark.skipif(not LIB.exists(), reason=f'field source plugin not built ({LIB})')
def test_field_source_mass(pincell, field_files):
    model, mesh = pincell
    a_s, a_f = 0.7, 1.3
    m = model.__class__(geometry=model.geometry, materials=model.materials,
                        settings=model.settings)
    m.settings = model.settings.__class__()
    m.settings.run_mode = 'fixed source'
    m.settings.particles, m.settings.batches = 1000, 1
    src = openmc.CompiledSource(library=str(LIB),
                                parameters=source_parameters(field_files, mesh, a_s, a_f))
    m.settings.source = src

    with pipeline.ce_session(m, mesh, field_files['work'] / 'source_tests'):
        sites = openmc.lib.sample_external_source(400_000, prn_seed=17)
    x = np.array([s.r for s in sites])
    E = np.array([s.E for s in sites])
    w = np.array([s.wgt for s in sites])

    # Expected mass per (voxel, group) from the frozen volumes
    F = fld.FieldFile.load(field_files['field'])
    with h5py.File(field_files['volumes'], 'r') as f:
        V, scale = f['V_A'][()], float(f.attrs['scale'])
    ref = fld.ReferenceField(F, field_files['lifts'], mesh)
    n_vox, G = int(np.prod(mesh.dimension)), F.n_groups
    expected = np.zeros((n_vox, G))
    lift_mass = _lift_mass_in_groups(field_files['lifts'], F)   # [M_row, tag, G]
    rows = {int(mid): i for i, mid in enumerate(lift_mass['ids'])}
    for R in range(F.n_regions):
        if F.material_id[R] < 0:
            continue
        b = int(F.mesh_bin[R])
        r = rows[int(F.material_id[R])]
        expected[b] += scale * V[R] * (a_s * F.Q_s[R] * lift_mass['s'][r]
                                       + a_f * F.Q_f[R] * lift_mass['f'][r])

    vox = np.array([ref.mesh_bin(xi) for xi in x])
    grp = np.searchsorted(F.edges, E, side='right') - 1
    n = len(w)
    est = np.zeros((n_vox, G))
    sq = np.zeros((n_vox, G))
    np.add.at(est, (vox, grp), w)
    np.add.at(sq, (vox, grp), w * w)
    mean = est / n
    sem = np.sqrt(np.maximum(sq / n - mean**2, 0) / n)

    # Total mass, then per bin
    total_sem = np.sqrt(np.var(w, ddof=1) / n)
    assert abs(w.mean() - expected.sum()) < 4 * total_sem
    live = sem > 0
    z = (mean[live] - expected[live]) / sem[live]
    assert np.mean(np.abs(z) > 3) < 0.01, f'max |z| = {np.max(np.abs(z)):.1f}'


def _lift_mass_in_groups(lift_path, F):
    """Fraction of each group's lift inside [E_lo, E_hi]; 1 for the defaults."""
    with h5py.File(lift_path, 'r') as f:
        ids = f['material_ids'][()]
    ones = np.ones((len(ids), F.n_groups))
    return {'ids': ids, 's': ones, 'f': ones}
