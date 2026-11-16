"""Importance options of the residual sampler added in the speedup study
(gate G8.6): the skin destination proposal, the tabulated outgoing-energy
proposal, and forward-cone directions.

Each option changes only where samples go and how they are weighted, so the
estimates must keep their means. On the graphite pincell, for tag s with N
samples per run:

- the low term's integral, mean of U / density over the density records,
  is the same with and without the skin and tabulated proposals (it depends
  on the destination and energy densities only);
- the CE production's integral, mean of C_s / density, is the same with and
  without forward-cone directions (it depends on the direction weights).

Means are compared with their standard errors (5 sigma).
"""
import math

import numpy as np
import pytest

import pipeline
import problems

TAIL = dict(tail='roulette', cutoff=math.exp(-3))


@pytest.fixture(scope='module')
def lib(pincell, field_files):
    import openmc.lib.rmc as rmc
    model, mesh = pincell
    with pipeline.ce_session(model, mesh, field_files['work'] / 'importance_options'):
        rmc.field_load(field_files['field'], field_files['lifts'], field_files['volumes'])
        yield rmc
        rmc.field_free()


def _integral(rmc, term, n, seed, **opts):
    rec = rmc.residual_sample('s', n, seed=seed, n_directions=4, **TAIL, **opts)
    d = rec['sampler'] == 0
    x = rec[term][d] / rec['density'][d]
    return x.mean(), x.std(ddof=1) / math.sqrt(len(x))


def _same(a, b):
    return abs(a[0] - b[0]) < 5 * math.hypot(a[1], b[1])


@pytest.mark.slow
def test_skin_destinations_keep_the_low_term(lib):
    rmc = lib
    base = _integral(rmc, 'U', 20_000, 1, beta=0.1)
    skin = _integral(rmc, 'U', 20_000, 2, beta=0.1,
                     skin=(0.4, 0.0, 0.0, problems.R_FUEL, 1e-4))
    assert _same(base, skin), (base, skin)


@pytest.mark.slow
def test_tabulated_energy_proposal_keeps_the_low_term(lib):
    rmc = lib
    R = rmc.field_n_regions()
    edges = np.logspace(-5, np.log10(2e7), 41)
    rng = np.random.default_rng(3)
    table = rng.random((R, len(edges) - 1)) ** 3     # uneven, nonzero rows
    base = _integral(rmc, 'U', 20_000, 4, beta=0.1)
    tab = _integral(rmc, 'U', 20_000, 5, beta=0.3, E_table=(edges, table))
    assert _same(base, tab), (base, tab)


@pytest.mark.slow
def test_forward_cone_keeps_the_production(lib):
    rmc = lib
    base = _integral(rmc, 'C_s', 20_000, 6, beta=0.1)
    cone = _integral(rmc, 'C_s', 20_000, 7, beta=0.1, cone=(0.5, 10.0, 1e3, 1e6))
    assert _same(base, cone), (base, cone)


def _total(rmc, term, n, seed, **opts):
    """Estimate of the term's integral and its standard error over
    destinations: records of one destination (n_outgoing of them, each with
    n_outgoing times its density) are summed first."""
    rec = rmc.residual_sample('s', n, seed=seed, n_directions=4, **TAIL, **opts)
    d = rec['sampler'] == 0
    ko = opts.get('n_outgoing', 1)
    x = (rec[term][d] / rec['density'][d]).reshape(n, ko).sum(axis=1)
    return x.mean(), x.std(ddof=1) / math.sqrt(n)


@pytest.mark.slow
def test_several_outgoing_states_keep_the_integrals(lib):
    """G8.7: n_outgoing stratified states per destination, sharing its
    traces, with the image roulette: the low term's and the CE production's
    integrals are unchanged."""
    rmc = lib
    for term, seeds in (('U', (11, 12)), ('C_s', (13, 14))):
        base = _total(rmc, term, 20_000, seeds[0], beta=0.1)
        multi = _total(rmc, term, 5_000, seeds[1], beta=0.1, n_outgoing=4,
                       image_roulette=0.25)
        assert _same(base, multi), (term, base, multi)


@pytest.mark.slow
def test_tabulated_incident_proposal_keeps_the_sampler_production(lib):
    """G8.8: the sampler-route particles' total production is the same with
    the log-uniform and a tabulated incident-energy proposal."""
    import openmc.lib
    rmc = lib
    edges = np.logspace(-5, np.log10(2e7), 61)
    n_mat = len(openmc.lib.materials)
    table = np.zeros((n_mat, 60))
    table[:, 40:] = np.linspace(1.0, 3.0, 20)       # above about 10 keV

    def production(seed, **opts):
        n = 20_000
        rec = rmc.residual_sample('s', n, seed=seed, n_directions=4, **TAIL, **opts)
        smp = rec['sampler'] == 1
        dest = np.cumsum(np.r_[0, (~smp)[:-1]].astype(np.int64))
        x = np.bincount(dest[smp], weights=(rec['C_s'][smp] + rec['C_f'][smp]) /
                        rec['density'][smp], minlength=n)
        return x.mean(), x.std(ddof=1) / math.sqrt(n)

    base = production(21)
    tab = production(22, Ein_table=(edges, table), beta_in=0.2)
    assert _same(base, tab), (base, tab)
