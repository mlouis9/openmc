"""Analysis of the Stage 8 runs (stage8.py): bias and figures of merit
against the CE reference, W_abs histories, and a projection of the figure
of merit after optimization.

    python stage8_analysis.py RESULTS_DIR OUTDIR [FIELD]

RESULTS_DIR holds reference.json and <variant>.json; OUTDIR receives the
figures (stage8_*.pdf) and stage8_summary.json, and a LaTeX file of numbers
(stage8_numbers.tex) for the report. FIELD (the field file) adds the radial
profile of the shape deviation.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 9, 'legend.fontsize': 7.5, 'font.family': 'serif'})
ORDER = ['none', 'binned', 'binned_exact_low', 'exact', 'combined', 'combined_fine']
LABEL = {'none': 'no cancellation', 'binned': 'binned netting',
         'binned_exact_low': 'binned, exact low term (f)',
         'exact': 'exact only (pooling + regional)',
         'combined': 'exact + binned netting',
         'combined_fine': 'exact + finer netting'}
COL = {'none': '#555555', 'binned': '#1f5f8b', 'binned_exact_low': '#2e7d32',
       'exact': '#d68910', 'combined': '#b03a2e', 'combined_fine': '#6c3483'}

# Projected speedup factors per kernel after optimization (report section on
# performance): cross sections on backward segments dominate the residual
# Source-region mesh: MESH_N x MESH_N over the cell (problems.pincell)
MESH_N = 12

PROJECTION = {'time_residual': 4.0, 'time_flights': 1.5, 'time_cancel': 5.0,
              'time_tally': 4.0, 'time_other': 5.0}


def load(d):
    ref = json.loads((d / 'reference.json').read_text())
    runs = {}
    for v in ORDER:
        p = d / f'{v}.json'
        if p.exists():
            runs[v] = json.loads(p.read_text())
    return ref, runs


def ref_stats(ref, key='rates'):
    k = np.array([r['k'] for r in ref['replicas']])
    shape = np.array([r.get(key, r['rates']) for r in ref['replicas']])
    T = np.array([r['time_active'] for r in ref['replicas']])
    return dict(k=k.mean(), k_sd_rep=k.std(ddof=1),
                k_sd_batch=float(np.mean([r['k_sd'] for r in ref['replicas']])),
                shape=shape.mean(axis=0), shape_sd=shape.std(axis=0, ddof=1),
                T=T.mean(), histories=ref['particles'] * (ref['batches'] - ref['inactive']))


def rmc_stats(run, regions, inactive):
    reps = []
    for rep in run['replicas']:
        if rep.get('failure') or 'tally' not in rep['rows'][-1]:
            continue
        rows = rep['rows'][inactive:]
        k = np.array([r['k_next'] for r in rows])
        tal = np.array([np.array(r['tally'])[regions] for r in rows])
        shapes = tal / tal.sum(axis=1, keepdims=True)
        T = sum(r['time'] for r in rows)
        times = {key: sum(r.get(key, 0.0) for r in rows) for key in
                 ('time_residual', 'time_flights', 'time_cancel')}
        times['time_total'] = T
        reps.append(dict(k=k, shape=shapes, T=T, times=times,
                         W=np.array([r['W_abs'] for r in rep['rows']])))
    k_mean = np.array([r['k'].mean() for r in reps])
    shape_mean = np.array([r['shape'].mean(axis=0) for r in reps])
    # Batch means (blocks of 5 generations) within each replica, pooled
    def bm(x, b=5):
        n = len(x) // b
        m = x[:n * b].reshape(n, b, *x.shape[1:]).mean(axis=1)
        return m.var(axis=0, ddof=1) / n
    k_var_bm = np.mean([bm(r['k']) for r in reps])
    sh_var_bm = np.mean([bm(r['shape']) for r in reps], axis=0)
    return dict(k=k_mean.mean(), k_sd_rep=k_mean.std(ddof=1) if len(k_mean) > 1 else np.nan,
                k_sd_bm=math.sqrt(k_var_bm), shape=shape_mean.mean(axis=0),
                shape_sd_rep=shape_mean.std(axis=0, ddof=1),
                shape_sd_bm=np.sqrt(sh_var_bm), T=np.mean([r['T'] for r in reps]),
                times={key: np.mean([r['times'][key] for r in reps]) for key in reps[0]['times']},
                W=[r['W'] for r in reps], k_gen=[r['k'] for r in reps],
                n_active=len(reps[0]['k']))


def main():
    d, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    ref, runs = load(d)
    def is_nu(run):
        return (run['args'].get('tally_mg_points') or 0) > 0
    R_by = {key: ref_stats(ref, key) for key in ('rates', 'nu_rates')}
    R = R_by['rates']
    regions = np.array(ref['regions'])
    summary = dict(reference=dict(k=R['k'], k_sd_rep=R['k_sd_rep'], k_sd_batch=R['k_sd_batch'],
                                  T=R['T'], histories=R['histories']), variants={})
    stats = {}
    histories = {}
    for v, run in runs.items():
        inactive = run['args']['inactive']
        histories[v] = [dict(W=np.array([r['W_abs'] for r in rep['rows']]),
                             k=np.array([r['k_next'] for r in rep['rows']]),
                             failure=rep.get('failure'))
                        for rep in run['replicas']]
        if not any((not rep.get('failure')) and 'tally' in rep['rows'][-1]
                   for rep in run['replicas']):
            summary['variants'][v] = dict(failed=[rep.get('failure') for rep in run['replicas']],
                                          W_last=[float(h['W'][-1]) for h in histories[v]],
                                          generations=[len(h['W']) for h in histories[v]])
            continue
        S = rmc_stats(run, np.array(run['fuel_regions']), inactive)
        stats[v] = S
        # the low + correction tally scores nu-fission when it has the MG control
        R = R_by['nu_rates' if is_nu(run) else 'rates']
        S['R'] = R
        n_rep = len(S['W'])
        # Bias: k and shape against the reference, with the combined error
        dk = S['k'] - R['k']
        sk = math.hypot(S['k_sd_bm'] / math.sqrt(len(S['W'])), R['k_sd_rep'] / math.sqrt(3))
        z_shape = (S['shape'] - R['shape']) / np.hypot(
            S['shape_sd_bm'] / math.sqrt(len(S['W'])), R['shape_sd'] / math.sqrt(3))
        # Replica estimate: the spread of replica means, which sees the slow
        # scattering modes that 5-generation batch means do not
        sk_rep = math.hypot(S['k_sd_rep'] / math.sqrt(n_rep), R['k_sd_rep'] / math.sqrt(len(ref['replicas'])))
        # Figures of merit (single replica: sigma of its mean, its active time)
        fom_k_ref = 1.0 / (R['k_sd_batch']**2 * R['T'])
        fom_k = 1.0 / (S['k_sd_bm']**2 * S['T'])
        fom_k_ref_rep = 1.0 / (R['k_sd_rep']**2 * R['T'])
        fom_k_rep = 1.0 / (S['k_sd_rep']**2 * S['T'])
        rel_ref = np.mean((R['shape_sd'] / R['shape'])**2)
        rel = np.mean((S['shape_sd_bm'] / S['shape'])**2)
        fom_sh_ref = 1.0 / (rel_ref * R['T'])
        fom_sh = 1.0 / (rel * S['T'])
        # Projection: per-kernel speedups
        t = S['times']
        other = t['time_total'] - t['time_residual'] - t['time_flights'] - t['time_cancel']
        t_proj = (t['time_residual'] / PROJECTION['time_residual'] +
                  t['time_flights'] / PROJECTION['time_flights'] +
                  t['time_cancel'] / PROJECTION['time_cancel'] +
                  other / PROJECTION['time_other'])
        speed = t['time_total'] / t_proj
        summary['variants'][v] = dict(
            k=S['k'], k_sd_bm=S['k_sd_bm'], k_sd_rep=S['k_sd_rep'], dk_pcm=1e5 * dk,
            dk_sigma=dk / sk if sk > 0 else float('nan'), dk_sigma_rep=dk / sk_rep,
            fom_k_rep=fom_k_rep, fom_k_ratio_rep=fom_k_rep / fom_k_ref_rep,
            fom_k_ratio_rep_projected=fom_k_rep / fom_k_ref_rep * speed,
            n_replicas=n_rep, n_active=S['n_active'],
            shape_chi2=float(np.sum(z_shape**2)), shape_dof=int(len(z_shape)),
            shape_max_z=float(np.max(np.abs(z_shape))),
            T=S['T'], fom_k=fom_k, fom_k_ratio=fom_k / fom_k_ref,
            fom_shape=fom_sh, fom_shape_ratio=fom_sh / fom_sh_ref,
            W_final=float(np.mean([w[-1] for w in S['W']])),
            W_mean_active=float(np.mean([w[inactive:].mean() for w in S['W']])),
            times={k_: float(v_) for k_, v_ in t.items()}, time_other=float(other),
            projected_speedup=float(speed),
            fom_k_ratio_projected=fom_k / fom_k_ref * speed,
            fom_shape_ratio_projected=fom_sh / fom_sh_ref * speed)
    (out / 'stage8_summary.json').write_text(json.dumps(summary, indent=1, default=float))

    # Figures
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for v in ORDER:
        for i, h in enumerate(histories.get(v, [])):
            axes[0].plot(h['W'], color=COL[v], lw=0.8, alpha=0.8,
                         label=LABEL[v] if i == 0 else None)
            if h['failure']:
                axes[0].plot(len(h['W']) - 1, h['W'][-1], 'x', color=COL[v], ms=6)
        if v not in stats:
            continue
        S = stats[v]
        for i, k in enumerate(S['k_gen']):
            axes[1].plot(np.arange(len(k)), k, color=COL[v], lw=0.6, alpha=0.7,
                         label=LABEL[v] if i == 0 else None)
    axes[0].set_yscale('log')
    axes[0].set_xlabel('generation')
    axes[0].set_ylabel(r'$W_{\rm abs}$ of the candidate bank')
    handles, labels = axes[0].get_legend_handles_labels()
    axes[1].axhline(R['k'], color='k', lw=1.2, label='CE reference')
    axes[1].set_xlabel('active generation')
    axes[1].set_ylabel('$k_n$')
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(out / 'stage8_wabs_k.pdf', bbox_inches='tight')
    plt.close(fig)

    # Shape: relative deviation from the reference by fuel region against the
    # region's distance from the pin centre (mesh-cell units), and the z
    # histogram. The field file (optional third argument) gives the mesh bins.
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    mesh_bin = None
    if len(sys.argv) > 3:
        import h5py
        with h5py.File(sys.argv[3]) as f:
            mesh_bin = f['regions/mesh_bin'][()][np.array(ref['regions'])]
    for v in ORDER:
        if v not in stats:
            continue
        Sv = stats[v]
        Rv = Sv['R']
        dev = Sv['shape'] / Rv['shape'] - 1.0
        if mesh_bin is not None:
            n = MESH_N
            r = np.round(np.hypot(mesh_bin % n - (n - 1) / 2, mesh_bin // n - (n - 1) / 2), 1)
            rr = np.unique(r)
            axes[0].plot(rr, [100 * dev[r == q].mean() for q in rr], 'o-', ms=3, lw=0.9,
                         color=COL[v], label=LABEL[v])
        z = (Sv['shape'] - Rv['shape']) / np.hypot(
            Sv['shape_sd_bm'] / math.sqrt(len(Sv['W'])), Rv['shape_sd'] / math.sqrt(3))
        axes[1].hist(np.clip(z, -6, 6), bins=np.linspace(-6, 6, 25), histtype='step',
                     color=COL[v], label=LABEL[v], density=True)
    axes[0].axhline(0, color='k', lw=0.8)
    axes[0].set_xlabel('distance of the region from the pin centre (mesh cells)')
    axes[0].set_ylabel('deviation from reference (%)')
    axes[0].legend(fontsize=6, loc='lower left')
    g = np.linspace(-6, 6, 200)
    axes[1].plot(g, np.exp(-g**2 / 2) / math.sqrt(2 * math.pi), 'k', lw=1.2)
    axes[1].set_xlabel('z of fuel-region fission fraction (batch means)')
    fig.tight_layout()
    fig.savefig(out / 'stage8_shape.pdf', bbox_inches='tight')
    plt.close(fig)

    # Per-generation diagnostics of one replica of the best variant: what each
    # canceller removes, and the noise of the production and amplitudes
    best = 'combined' if 'combined' in runs else None
    if best is not None:
        rows = runs[best]['replicas'][0]['rows']
        g = np.array([r['n'] for r in rows])
        get = lambda f: np.array([f(r) for r in rows], dtype=float)
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
        axes[0].plot(g, get(lambda r: r['cancel']['netting']['W_before']), color='#1f5f8b',
                     lw=0.9, label='before netting')
        axes[0].plot(g, get(lambda r: r['cancel']['netting']['W_after']), color='#1f5f8b',
                     lw=0.9, ls='--', label='after netting')
        axes[0].plot(g, get(lambda r: r['cancel']['pooling']['W_in']), color='#b03a2e',
                     lw=0.9, label='fission residual, before pooling')
        axes[0].plot(g, get(lambda r: r['cancel']['pooling']['W_out']), color='#b03a2e',
                     lw=0.9, ls='--', label='fission residual, after pooling')
        axes[0].set_yscale('log')
        axes[0].set_xlabel('generation')
        axes[0].set_ylabel('absolute weight')
        axes[0].legend(fontsize=6)
        for key, col, lab in (('f', 'k', '$f_n$'), ('a_s', '#1f5f8b', '$a_s$'),
                              ('a_f', '#b03a2e', '$a_f$')):
            axes[1].plot(g, get(lambda r: r[key]), color=col, lw=0.8, label=lab)
        axes[1].axhline(1.0, color='k', lw=0.5, ls=':')
        axes[1].set_xlabel('generation')
        axes[1].set_ylabel('production and amplitudes')
        axes[1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / 'stage8_diagnostics.pdf', bbox_inches='tight')
        plt.close(fig)

    # LaTeX numbers
    lines = [f"\\newcommand{{\\refk}}{{{R['k']:.5f}}}",
             f"\\newcommand{{\\refksd}}{{{R['k_sd_rep']:.5f}}}",
             f"\\newcommand{{\\refT}}{{{R['T']:.0f}}}"]
    rows = []
    for v in ORDER:
        if v not in summary['variants'] or 'failed' in summary['variants'][v]:
            continue
        s = summary['variants'][v]
        rows.append(f"{LABEL[v]} & {s['k']:.5f} & {s['dk_pcm']:+.0f} & {s['dk_sigma']:+.1f} & "
                    f"{s['dk_sigma_rep']:+.1f} & {s['W_mean_active']:.3g} & {s['T']:.0f} & "
                    f"{s['fom_k_ratio']:.2g} & {s['fom_k_ratio_rep']:.2g} & {s['fom_shape_ratio']:.2g} & "
                    f"{s['projected_speedup']:.1f}\\\\")
    (out / 'stage8_table.tex').write_text('\n'.join(rows) + '\n')
    (out / 'stage8_numbers.tex').write_text('\n'.join(lines) + '\n')
    print(json.dumps(summary, indent=1, default=float))


if __name__ == '__main__':
    main()
