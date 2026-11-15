"""Efficiency of residual estimators for the eigenvalue: the variance, per
unit of sampling time, of the fission yield of one generation's residual.

    python residual_efficiency.py --work rmc_visual_work --configs base,ctrl --reps 16

Why this measure. In the iteration, a generation's residual records enter
the bank and reach fission over the following generations. To first order
their effect on k is their total fission yield Y = <1, F (I - S)^{-1} r_hat>,
the fission weight of their histories. A history-based flight of each
record (rmc.propagate_aux with history) estimates it. The variance of Y per
generation, times the time to sample the records, measures how much
eigenvalue noise an estimator injects per second; the stage8 runs confirm
the ranking on the full iteration.

Each replicate samples both tags with N records each at a = b = (1, 1) and
k = k_MG, applies the configured fission treatment (pooling or not),
transports the records with n_hist histories, and records Y_s and Y_f. The
transport's own noise is measured by transporting the same records twice
with different seeds and subtracted.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

import cancellation as canc
import controlled_residual_study as cs
import pipeline

TAIL = dict(tail='roulette', cutoff=math.exp(-4))

# name: per-tag residual_sample options, the control coefficient c, pooling
# of the fission residual, and N per tag (default the command line's)
CONFIGS = {
    'base': dict(s=dict(directions='isotropic', n_directions=8, **TAIL),
                 f=dict(directions='isotropic', n_directions=8, all_sampler=True, **TAIL),
                 pool=True),
}


def config(name):
    """Configurations by name: 'base' and variations written key=value with
    '+' between them, e.g. 'nd=2+ctrl=1+beta=0.1'. Keys:
      nd   directions per destination (both tags)
      ctrl 1: the MG control with c = 1 on the density branch (both tags);
           L = nL MG-only directions restore its mean
      nL   MG-only directions of the control's mean (default 8)
      beta weight of the log-uniform part of the outgoing-energy proposal
      fdens 1: the fission residual on the density route (no pooling)
      inc  'flux': incident energies from the path's MG flux
      vox  destination proposal: box, volume, fissionable
      K    incident energies per direction on the sampler route
      Ns, Nf  records per tag
    """
    c = json.loads(json.dumps(CONFIGS['base']))
    c['c'] = 0.0
    if name == 'base':
        return c
    for item in name.split('+'):
        key, val = item.split('=')
        if key == 'nd':
            for t in 'sf':
                c[t]['n_directions'] = int(val)
        elif key == 'ctrl':
            c['c'] = float(val)
            for t in 'sf':
                c[t]['control'] = c['c'] != 0.0
                c[t].setdefault('n_control_directions', 8)
        elif key == 'nL':
            for t in 'sf':
                c[t]['n_control_directions'] = int(val)
        elif key == 'beta':
            for t in 'sf':
                c[t]['beta'] = float(val)
        elif key == 'fdens':
            if int(val):
                c['f'].pop('all_sampler', None)
                c['pool'] = False
        elif key == 'inc':
            for t in 'sf':
                c[t]['incident'] = val
        elif key == 'vox':
            for t in 'sf':
                c[t]['voxel'] = val
        elif key == 'cone':
            # fraction:kappa[:E0:E1]
            v = [float(x) for x in val.split(':')]
            v += [1e4, 1e7][len(v) - 2:]
            for t in 'sf':
                c[t]['cone'] = tuple(v)
        elif key == 'skin':
            # fraction[:dmin], about the pin (problems.R_FUEL at the origin)
            import problems
            v = [float(x) for x in val.split(':')]
            c['s']['skin'] = (v[0], 0.0, 0.0, problems.R_FUEL, v[1] if len(v) > 1 else 1e-5)
        elif key == 'sall':
            c['s']['all_sampler'] = bool(int(val))
        elif key == 'cut':
            for t in 'sf':
                c[t]['cutoff'] = math.exp(-float(val))
        elif key == 'voxf':
            c['f']['voxel'] = val
        elif key == 'nds':
            c['s']['n_directions'] = int(val)
        elif key == 'ndf':
            c['f']['n_directions'] = int(val)
        elif key == 'K':
            for t in 'sf':
                c[t]['n_energies'] = int(val)
        elif key in ('Ns', 'Nf'):
            c[key] = int(val)
        elif key == 'opt':
            c['opt'] = val          # tags with the optimized proposal, e.g. 'sf'
        elif key == 'ctrls':
            c['c'] = float(val)
            c['s']['control'] = True
            c['s'].setdefault('n_control_directions', 8)
        else:
            raise ValueError(f'unknown key {key}')
    return c


def contributions(rmc, rec, w, n_per, k, seed):
    """Per-destination-sample contributions to Y: each record's weight times
    its fission yield per unit weight, estimated twice independently by
    n_per history flights per record (so that the product of the two
    estimates is unbiased for the squared yield)."""
    M = len(w)
    dens = rec['sampler'] == 0
    # records of one destination: its sampler particles, then its density
    # record (as DestinationPooling assumes)
    sample = np.cumsum(np.r_[0, dens[:-1]].astype(np.int64))
    etas = []
    for j in range(2):
        b = rmc.Bank(r=rec['r'], u=rec['u'], E=rec['E'], w=np.ones(M),
                     tag=rec['tag'].astype(np.int32))
        o, st, aux = rmc.propagate_aux(b, n_per * M, k, seed=seed + j, children=True,
                                       region_tally=False, history=True)
        ch = o.arrays()
        o.free()
        b.free()
        # n_per M flights over M unit parents: each flight carries 1/n_per,
        # so a parent's children's total weight is its yield per unit weight
        etas.append(np.bincount(aux['parent'], weights=ch['w'], minlength=M))
    S = sample.max() + 1
    c1 = np.bincount(sample, weights=w * etas[0], minlength=S)
    c2 = np.bincount(sample, weights=w * etas[1], minlength=S)
    return c1, c2, sample, dens


E_EDGES = np.logspace(-5, np.log10(2e7), 61)


def importance_table(rmc, files, k, n_states=200_000, n_per=4, seed=99):
    """Fission yield per unit weight of a neutron by field region and energy
    bin (60 log bins), from history flights of states uniform in the box,
    log-uniform in energy, and isotropic: the k-importance of an emission,
    averaged over direction and within the bin."""
    rng = np.random.default_rng(seed)
    import h5py
    with h5py.File(files['field']) as f:
        lo, hi = f['domain_lower_left'][()], f['domain_upper_right'][()]
    r = lo + (hi - lo) * rng.random((n_states, 3))
    reg = rmc.field_locate(r)
    ok = reg >= 0
    r, reg = r[ok], reg[ok]
    M = len(reg)
    E = 10**rng.uniform(-5, np.log10(2e7), M)
    mu = 2 * rng.random(M) - 1
    ph = 2 * np.pi * rng.random(M)
    st = np.sqrt(1 - mu**2)
    u = np.column_stack([st * np.cos(ph), st * np.sin(ph), mu])
    b = rmc.Bank(r=r, u=u, E=E, w=np.ones(M), tag=np.zeros(M, dtype=np.int32))
    o, st_, aux = rmc.propagate_aux(b, n_per * M, k, seed=seed, children=True,
                                    region_tally=False, history=True)
    ch = o.arrays()
    o.free()
    b.free()
    eta = np.bincount(aux['parent'], weights=ch['w'], minlength=M)
    R = rmc.field_n_regions()
    e = np.clip(np.searchsorted(E_EDGES, E, side='right') - 1, 0, len(E_EDGES) - 2)
    key = reg * (len(E_EDGES) - 1) + e
    nb = R * (len(E_EDGES) - 1)
    tot = np.bincount(key, weights=eta, minlength=nb)
    cnt = np.bincount(key, minlength=nb)
    table = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan).reshape(R, -1)
    # Empty bins: the material-wide mean of the energy bin
    with np.errstate(invalid='ignore'):
        fill = np.nanmean(table, axis=0)
    table = np.where(np.isnan(table), fill[None, :], table)
    return np.nan_to_num(table)


def lookup(table, rmc, r, E):
    reg = np.maximum(rmc.field_locate(r), 0)
    e = np.clip(np.searchsorted(E_EDGES, E, side='right') - 1, 0, len(E_EDGES) - 2)
    return table[reg, e]


P_EDGES = np.logspace(-5, np.log10(2e7), 101)


def build_proposal(rmc, files, k, table, base='base', N_big=50_000, floor=0.1, seed0=7):
    """Destination and outgoing-energy proposal for each tag, near the
    variance-optimal one for Y: probability of (region R, energy bin e)
    proportional to sqrt(n_b m2_b), with n_b the pilot's samples in the bin
    and m2_b the sum of their squared contributions (so that the density is
    the pilot's times sqrt(E[c^2 | bin])). A fraction floor of each tag's
    region probability goes to volume (support everywhere)."""
    c = config(base)
    with __import__('h5py').File(files['volumes']) as f:
        V = f['V_A'][()]
    R = len(V)
    nE = len(P_EDGES) - 1
    out = {}
    for t in 'sf':
        seed = 1_000_003 * (seed0 + 1) + (17 if t == 's' else 29)
        rec = rmc.residual_sample(t, N_big, seed=seed, **c[t])
        w = rmc.residual_weights(rec, N_big, (1.0, 1.0), (1.0, 1.0), k, c['c'], 0.0)
        dens = rec['sampler'] == 0
        sample = np.cumsum(np.r_[0, dens[:-1]].astype(np.int64))
        cs_ = np.bincount(sample, weights=w * lookup(table, rmc, rec['r'], rec['E']))
        reg = np.maximum(rmc.field_locate(rec['r'][dens]), 0)
        e = np.clip(np.searchsorted(P_EDGES, rec['E'][dens], side='right') - 1, 0, nE - 1)
        key = reg * nE + e
        n_b = np.bincount(key, minlength=R * nE).astype(float)
        m2_b = np.bincount(key, weights=cs_**2, minlength=R * nE)
        p = np.sqrt(n_b * m2_b).reshape(R, nE)
        piR = p.sum(axis=1)
        piR = (1 - floor) * piR / piR.sum() + floor * V / V.sum()
        out[t] = dict(voxel_weights=piR, E_table=(P_EDGES, p))
    return out


def run_config_table(rmc, files, name, N, k, table, N_big=100_000, seed0=0, proposal=None):
    """Variance of Y per generation at N records per tag, with the yield of
    each record from the importance table: Var(Y_N) = (S m2 - Y^2) / N over
    S = N_big destinations (weights carry 1/N_big), per-destination sums."""
    c = config(name)
    pool = canc.DestinationPooling(rmc, files) if c.get('pool') else None
    res = dict(name=name, N=N, N_big=N_big)
    for t in 'sf':
        n_t = c.get('N' + t, N)
        seed = 1_000_003 * (seed0 + 1) + (17 if t == 's' else 29)
        opts = dict(c[t])
        if c.get('opt') and proposal is not None and t in c['opt']:
            opts.update(voxel='tabulated', voxel_weights=proposal[t]['voxel_weights'],
                        E_table=proposal[t]['E_table'])
            opts.setdefault('beta', 0.1)
        t0 = time.perf_counter()
        rec = rmc.residual_sample(t, N_big, seed=seed, **opts)
        w = rmc.residual_weights(rec, N_big, (1.0, 1.0), (1.0, 1.0), k, c['c'], 0.0)
        dens = rec['sampler'] == 0
        sample = np.cumsum(np.r_[0, dens[:-1]].astype(np.int64))
        if t == 'f' and pool is not None:
            bank, _ = pool(rec, w, N_big, 1.0, np.random.default_rng(seed))
            T = time.perf_counter() - t0
            # pooled particles are attributed to samples in order: kept
            # particles keep their record's sample; one pooled particle per
            # destination follows (DestinationPooling's output order)
            contrib = bank['w'] * lookup(table, rmc, bank['r'], bank['E'])
            cs_ = contrib  # each particle as its own unit (destinations of the
            # pool are independent; kept particles of one destination are few)
            W = float(np.abs(bank['w']).sum())
        else:
            T = time.perf_counter() - t0
            contrib = w * lookup(table, rmc, rec['r'], rec['E'])
            cs_ = np.bincount(sample, weights=contrib)
            W = float(np.abs(w).sum())
        S = N_big
        m2 = float(np.sum(cs_**2))
        Y = float(cs_.sum())
        var_N = (S * m2 - Y**2) / n_t
        top = float(np.sort(cs_**2)[-10:].sum() / m2) if m2 > 0 else 0.0
        T_N = T * n_t / N_big
        res[t] = dict(mean=Y, var=var_N, T=T_N, W=W * N_big / n_t / (N_big / n_t),
                      cost_var=var_N * T_N, N=n_t, top10=top,
                      mass_var=float((S * np.sum(np.bincount(sample, weights=w)**2)
                                      - w.sum()**2) / n_t))
    res['cost_var'] = res['s']['cost_var'] + res['f']['cost_var']
    return res


def run_config(rmc, files, name, N, reps, n_hist, k, seed0=0, N_big=20_000, n_per=8):
    """Variance of Y per generation at N records per tag, from one large
    sample of N_big destinations per tag: Var(Y_N) = N_big^2 Var(c) / N, with
    c the per-destination contributions (weights carry 1/N_big)."""
    c = config(name)
    pool = canc.DestinationPooling(rmc, files) if c.get('pool') else None
    res = dict(name=name, N=N, N_big=N_big)
    for t in 'sf':
        n_t = c.get('N' + t, N)
        seed = 1_000_003 * (seed0 + 1) + (17 if t == 's' else 29)
        opts = dict(c[t])
        if c.get('opt') and proposal is not None and t in c['opt']:
            opts.update(voxel='tabulated', voxel_weights=proposal[t]['voxel_weights'],
                        E_table=proposal[t]['E_table'])
            opts.setdefault('beta', 0.1)
        t0 = time.perf_counter()
        rec = rmc.residual_sample(t, N_big, seed=seed, **opts)
        w = rmc.residual_weights(rec, N_big, (1.0, 1.0), (1.0, 1.0), k, c['c'], 0.0)
        if t == 'f' and pool is not None:
            # Pooling replaces each destination's records by kept particles
            # and one pooled particle; keep the per-destination structure by
            # pooling, then attributing each output to its destination
            bank, _ = pool(rec, w, N_big, 1.0, np.random.default_rng(seed))
            T = time.perf_counter() - t0
            # Pooled banks lose the record order; treat each particle as its
            # own sample (pooling is per destination, so particles of one
            # destination are nearly independent after pooling)
            rec2 = dict(r=bank['r'], u=bank['u'], E=bank['E'], tag=bank['tag'],
                        sampler=np.zeros(len(bank['w']), dtype=np.int32))
            c1, c2, _, _ = contributions(rmc, rec2, bank['w'], n_per, k, 7919 * seed)
            W = float(np.abs(bank['w']).sum())
        else:
            T = time.perf_counter() - t0
            c1, c2, _, _ = contributions(rmc, rec, w, n_per, k, 7919 * seed)
            W = float(np.abs(w).sum())
        S = N_big
        m2 = float(np.sum(c1 * c2))          # sum of squared contributions
        mean = float(c1.sum())
        var_N = (S * m2 - mean**2) / n_t
        T_N = T * n_t / N_big
        res[t] = dict(mean=mean, var=var_N, T=T_N, W=W, cost_var=var_N * T_N, N=n_t)
    res['cost_var'] = res['s']['cost_var'] + res['f']['cost_var']
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--work', default='rmc_visual_work')
    ap.add_argument('--configs', default='base')
    ap.add_argument('--N', type=int, default=2000)
    ap.add_argument('--reps', type=int, default=1)
    ap.add_argument('--n-hist', type=int, default=0)
    ap.add_argument('--N-big', type=int, default=100_000)
    ap.add_argument('--field-groups', default=None,
                    help='use field_variants.field_files with this group structure')
    ap.add_argument('--mgxs-particles', type=int, default=20_000)
    ap.add_argument('--field-mesh', type=int, default=12,
                    help='source-region mesh per side of the field variant')
    ap.add_argument('--pilot', default='base',
                    help='configuration of the pilot run of the optimized proposal')
    ap.add_argument('--n-per', type=int, default=8)
    ap.add_argument('--out', default='residual_efficiency.json')
    args = ap.parse_args()

    import openmc.lib.rmc as rmc
    if args.field_groups:
        import field_variants as fv
        w, files, k_mg = fv.field_files(args.work, args.field_groups, args.mgxs_particles,
                                        n_per_side=args.field_mesh)
    else:
        w, files, k_mg = cs.exact_volume_files(args.work)
    results = []
    p = Path(args.out)
    if p.exists():
        results = json.loads(p.read_text())
    with pipeline.ce_session(w.model, w.mesh, files['work'] / ('residual_efficiency_' + Path(args.out).stem)):
        rmc.field_load(files['field'], files['lifts'], files['volumes'])
        tpath = files['work'] / 'importance_table.npy'
        if tpath.exists():
            table = np.load(tpath)
        else:
            table = importance_table(rmc, files, k_mg)
            np.save(tpath, table)
        proposal = None
        if any('opt=' in n for n in args.configs.split(',')):
            proposal = build_proposal(rmc, files, k_mg, table, base=args.pilot)
        for name in args.configs.split(','):
            r = run_config_table(rmc, files, name, args.N, k_mg, table, N_big=args.N_big,
                                 proposal=proposal)
            results.append(r)
            print(f"{name:32s} Ys {r['s']['mean']:+.3f} Yf {r['f']['mean']:+.3f} "
                  f"var_s {r['s']['var']:.3e} T_s {r['s']['T']:.2f}  "
                  f"var_f {r['f']['var']:.3e} T_f {r['f']['T']:.2f}  "
                  f"top10 {r['s']['top10']:.2f}/{r['f']['top10']:.2f}  var*T {r['cost_var']:.3e}",
                  flush=True)
            p.write_text(json.dumps(results, indent=1))
        rmc.field_free()


if __name__ == '__main__':
    main()
