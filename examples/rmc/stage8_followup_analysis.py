"""Analysis of the Stage 8 follow-up runs: the eigenvalue bias against the
number of residual samples, independent seeds, resonance-resolved netting,
normalization damping, field amplitudes that follow the bank's (b = a), and
history-based flights.

    python stage8_followup_analysis.py STAGE8_DIR FOLLOWUP_DIR OUTDIR

STAGE8_DIR holds reference.json and combined.json (results/stage8);
FOLLOWUP_DIR the follow-up runs (results/stage8_followup). OUTDIR receives
followup_*.pdf, followup_table.tex, and followup_summary.json.

Replicas of different runs with the same seed share their residual samples
(the residual's random numbers depend on the seed, generation, and sample
index only), so their difference is a paired comparison; runs with different
seeds are independent.
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

# Runs of the collision-generation iteration: (label, file, directory key)
COLLISION = [
    ('baseline, seeds 2000-2002', 'combined.json', 's8'),
    ('baseline, seeds 3000-3002', 'combined_s3000.json', 'fu'),
    ('baseline, seeds 3003-3005', 'combined_s3003.json', 'fu'),
    ('$N = 16000$, seeds 2000-2001', 'combined_N16000.json', 'fu'),
    ('resonance-resolved netting', 'combined_resfine.json', 'fu'),
    ('damping $\\lambda = 0.25$', 'combined_damp.json', 'fu'),
    ('amplitudes $b = a$', 'combined_follow.json', 'fu'),
]
HISTORY = [
    ('$b = (1,1)$, $N = 2000$', 'history_fixed_N2000.json'),
    ('$b = (1,1)$, $N = 16000$', 'history_fixed_N16000.json'),
    ('$b = (1,1)$, $\\lambda = 0.1$, $N = 4000$', 'history_damped_N4000.json'),
    ('$b = a$, $N = 2000$', 'history_follow_N2000.json'),
    ('$b = a$, $N = 64000$', 'history_follow_N64000.json'),
]


def replica_stats(run):
    ina = run['args']['inactive']
    out = []
    for rep in run['replicas']:
        if rep.get('failure'):
            continue
        rows = rep['rows'][ina:]
        k = np.array([r['k_next'] for r in rows])
        out.append(dict(k=k.mean(), sd=k.std(ddof=1), n=len(k),
                        W=np.mean([r['W_abs'] for r in rows]),
                        T=sum(r['time'] for r in rows),
                        seed=run['args'].get('seed_base', 2000) + rep['replica'],
                        lag1=np.corrcoef(k[:-1], k[1:])[0, 1]))
    return out


def main():
    s8, fu, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    ref = json.loads((s8 / 'reference.json').read_text())
    k_ref = np.mean([r['k'] for r in ref['replicas']])
    sd_ref = np.std([r['k'] for r in ref['replicas']], ddof=1)
    T_ref = np.mean([r['time_active'] for r in ref['replicas']])
    fom_ref = 1.0 / (sd_ref**2 * T_ref)
    d = {'s8': s8, 'fu': fu}
    runs = {lab: json.loads((d[key] / f).read_text()) for lab, f, key in COLLISION}
    stats = {lab: replica_stats(r) for lab, r in runs.items()}

    # Independent baseline replicas: bias and the effective correlation time
    base = [x for lab in list(stats)[:3] for x in stats[lab]]
    kb = np.array([x['k'] for x in base])
    sd_rep = kb.std(ddof=1)
    tau_eff = np.mean([x['n'] for x in base]) * (sd_rep / np.mean([x['sd'] for x in base]))**2
    summary = dict(reference=dict(k=k_ref, sd=sd_ref, T=T_ref),
                   baseline=dict(k=kb.mean(), dk_pcm=1e5 * (kb.mean() - k_ref),
                                 sem_pcm=1e5 * sd_rep / math.sqrt(len(kb)),
                                 sd_rep=sd_rep, tau_eff=tau_eff, n=len(kb)),
                   runs={})
    paired_base = {x['seed']: x['k'] for x in stats[COLLISION[0][0]]}
    sig_b = sd_rep
    T_b = np.mean([x['T'] for x in base])
    summary['baseline']['fom_ratio'] = 1.0 / (sig_b**2 * T_b) / fom_ref
    rows_tex = [f"baseline, 9 independent replicas & 9 & "
                f"${b_dk:+.0f}\\pm{b_sem:.0f}$ & -- & "
                f"{np.mean([x['W'] for x in base]):.2f} & "
                f"{np.mean([x['sd'] for x in base]):.4f} & {T_b:.0f} & "
                f"{summary['baseline']['fom_ratio']:.2g}\\\\"
                for b_dk, b_sem in [(summary['baseline']['dk_pcm'], summary['baseline']['sem_pcm'])]]
    for lab, _, _ in COLLISION:
        S = stats[lab]
        k = np.array([x['k'] for x in S])
        sd_gen = np.mean([x['sd'] for x in S])
        n = np.mean([x['n'] for x in S])
        T = np.mean([x['T'] for x in S])
        # Error of a replica mean: the per-generation spread with the
        # correlation time of the pooled independent baseline replicas (three
        # replicas alone give an error uncertain by a factor of two)
        sig = sd_gen * math.sqrt(tau_eff / n)
        fom = 1.0 / (sig**2 * T)
        pairs = [x['k'] - paired_base[x['seed']] for x in S if x['seed'] in paired_base]
        pd = (np.mean(pairs), np.std(pairs, ddof=1) / math.sqrt(len(pairs))) \
            if len(pairs) > 1 and lab != COLLISION[0][0] else (np.nan, np.nan)
        summary['runs'][lab] = dict(k=list(k), dk_pcm=1e5 * (k.mean() - k_ref),
                                    sem_pcm=1e5 * sig / math.sqrt(len(k)),
                                    W=np.mean([x['W'] for x in S]), sd_gen=sd_gen, T=T,
                                    fom_ratio=fom / fom_ref,
                                    paired_pcm=1e5 * pd[0], paired_sem_pcm=1e5 * pd[1])
        s = summary['runs'][lab]
        paired = ('--' if np.isnan(pd[0]) else
                  f"${s['paired_pcm']:+.0f}\\pm{s['paired_sem_pcm']:.0f}$")
        rows_tex.append(f"{lab} & {len(k)} & ${s['dk_pcm']:+.0f}\\pm{s['sem_pcm']:.0f}$ & "
                        f"{paired} & {s['W']:.2f} & {sd_gen:.4f} & {T:.0f} & "
                        f"{s['fom_ratio']:.2g}\\\\")

    # History-based flights
    hist = {}
    for lab, f in HISTORY:
        r = json.loads((fu / f).read_text())
        hist[lab] = r
        rep = r['replicas'][0]
        ina = r['args']['inactive']
        ok = [x for x in r['replicas'] if not x.get('failure')]
        entry = dict(failed=sum(1 for x in r['replicas'] if x.get('failure')),
                     replicas=len(r['replicas']),
                     generations_to_failure=[len(x['rows']) for x in r['replicas']
                                             if x.get('failure')])
        if ok:
            rows = ok[0]['rows'][ina:]
            k = np.array([q['k_next'] for q in rows])
            lag1 = np.corrcoef(k[:-1], k[1:])[0, 1]
            tau = (1 + lag1) / (1 - lag1)
            sig = k.std(ddof=1) * math.sqrt(tau / len(k))
            T = sum(q['time'] for q in rows)
            entry.update(k=k.mean(), dk_pcm=1e5 * (k.mean() - k_ref), sem_pcm=1e5 * sig,
                         sd_gen=k.std(ddof=1), lag1=lag1, T=T,
                         W=np.mean([q['W_abs'] for q in rows]),
                         f_sd=np.std([q['f'] for q in rows]),
                         fom_ratio=1.0 / (sig**2 * T) / fom_ref)
        summary['runs']['history: ' + lab] = entry

    # Figure 1: replica means by run, same seed in the same colour; and the
    # bias against 1/N with the two simple models
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    seeds = sorted({x['seed'] for S in stats.values() for x in S})
    cmap = plt.get_cmap('tab10')
    for i, (lab, _, _) in enumerate(COLLISION):
        for x in stats[lab]:
            axes[0].plot(x['k'], i, 'o', ms=4, color=cmap(seeds.index(x['seed']) % 10))
    axes[0].axvline(k_ref, color='k', lw=1.0)
    axes[0].axvspan(k_ref - sd_ref, k_ref + sd_ref, color='0.85')
    axes[0].set_yticks(range(len(COLLISION)))
    axes[0].set_yticklabels([lab for lab, _, _ in COLLISION], fontsize=7)
    axes[0].invert_yaxis()
    axes[0].set_xlabel('replica mean of $k_n$ (colour: seed)')
    b2 = summary['baseline']
    b16 = summary['runs'][COLLISION[3][0]]
    x = np.array([1 / 2000, 1 / 16000])
    y = np.array([b2['dk_pcm'], b16['dk_pcm']])
    e = np.array([b2['sem_pcm'], b16['sem_pcm']])
    axes[1].errorbar(x * 1e4, y, yerr=e, fmt='o', color='#b03a2e', capsize=3,
                     label='measured (independent seeds at $N=2000$)')
    g = np.linspace(0, 1 / 2000, 50)
    axes[1].plot(g * 1e4, b2['dk_pcm'] * g * 2000, 'k--', lw=0.9, label=r'bias $\propto 1/N$')
    axes[1].plot(g * 1e4, np.full_like(g, b2['dk_pcm']), 'k:', lw=0.9, label='bias independent of $N$')
    axes[1].axhline(0, color='0.5', lw=0.6)
    axes[1].set_xlabel(r'$10^4/N$ (residual samples per tag)')
    axes[1].set_ylabel(r'$\Delta k$ (pcm)')
    axes[1].legend(fontsize=6.5, loc='upper left')
    fig.tight_layout()
    fig.savefig(out / 'followup_bias.pdf', bbox_inches='tight')
    plt.close(fig)

    # Figure 2: the scattering mass of the candidate bank against the field's
    # amplitude offset a_s - b_s (baseline), and f by generation with b = a
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    slopes = []
    for rep in runs[COLLISION[0][0]]['replicas']:
        rows = rep['rows'][1:]
        xa = np.array([q['a_s'] - q['b_s'] for q in rows])
        ys = np.array([q['sum_s'] for q in rows])
        axes[0].plot(xa, ys, '.', ms=2.5, alpha=0.6)
        slopes.append(np.polyfit(xa, ys, 1)[0])
        summary.setdefault('amplitude_term', []).append(
            dict(slope=slopes[-1], corr=float(np.corrcoef(xa, ys)[0, 1])))
    g = np.linspace(-0.12, 0.12, 10)
    axes[0].plot(g, np.mean(slopes) * g, 'k', lw=1.0,
                 label=f'slope {np.mean(slopes):.1f} (field scattering mass)')
    axes[0].set_xlabel('$a_s - b_s$')
    axes[0].set_ylabel(r'tag-s signed sum of the candidate bank')
    axes[0].legend(fontsize=7)
    for lab, col in ((HISTORY[0][0], '#555555'), (HISTORY[3][0], '#d68910'),
                     (HISTORY[4][0], '#1f5f8b')):
        for i, rep in enumerate(hist[lab]['replicas'][:1]):
            f = [q['f'] for q in rep['rows']]
            axes[1].plot(range(len(f)), f, 'o-', ms=2.5, lw=0.8, color=col,
                         label='history, ' + lab)
    axes[1].axhline(1.0, color='k', lw=0.6, ls=':')
    axes[1].set_ylim(-1.0, 2.5)
    axes[1].set_xlabel('fission generation')
    axes[1].set_ylabel('production $f_n$')
    axes[1].legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(out / 'followup_amplitude_history.pdf', bbox_inches='tight')
    plt.close(fig)

    (out / 'followup_table.tex').write_text('\n'.join(rows_tex) + '\n')
    (out / 'followup_summary.json').write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps(summary, indent=1, default=float))


if __name__ == '__main__':
    main()
