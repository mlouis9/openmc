"""Figures and tables of the speedup study: the residual estimator's
efficiency (residual_efficiency.py scans) and the iteration runs
(stage8.py) that followed from it.

    python speedup_analysis.py RESULTS_DIR STAGE8_DIR OUTDIR

RESULTS_DIR is results/speedup, STAGE8_DIR results/stage8 (the reference and
the baseline). OUTDIR receives speedup_*.pdf, speedup_runs.tex,
speedup_scans.tex and speedup_summary.json.

Errors of a replica mean use batch means of 10 generations pooled over the
replicas (the integrated autocorrelation time of the improved iteration is a
few generations; for the baseline, see stage8_followup_analysis.py). The
reference's figure of merit uses its replica spread.
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

# Scans: (label, file, configuration name), grouped
SCANS = [
    ('directions $J$', [('$J=1$', 'scan1.json', 'nd=1'), ('$J=2$', 'scan1.json', 'nd=2'),
                        ('$J=4$', 'scan1.json', 'nd=4'), ('$J=16$', 'scan1.json', 'nd=16'),
                        ('$J=32$', 'scan1.json', 'nd=32')]),
    ('outgoing energy', [('$\\beta=0.6$', 'scan1.json', 'beta=0.6'),
                         ('$\\beta=0.3$', 'scan1.json', 'beta=0.3'),
                         ('$\\beta=0.1$', 'scan1.json', 'beta=0.1')]),
    ('destinations', [('by volume', 'scan2.json', 'vox=volume'),
                      ('skin 0.2', 'scan8.json', 'beta=0.1+cut=3+skin=0.2'),
                      ('skin 0.4, $10^{-4}$', 'scan8.json', 'beta=0.1+cut=3+skin=0.4:1e-4'),
                      ('tabulated optimum', 'scan6.json', 'opt=s+beta=0.1+cut=3')]),
    ('controls', [('MG control (s)', 'scan2.json', 'ctrls=1'),
                  ('MG control (both)', 'scan2.json', 'ctrl=1'),
                  ('$K=2$ incident', 'scan2.json', 'K=2'),
                  ('forward cone 0.5:5', 'scan7.json', 'beta=0.1+cut=3+cone=0.5:5')]),
    ('field', [('40 groups', 'scan9.json', 'beta=0.1+cut=3'),
               ('MGXS $5\\times$ stats', 'scan10.json', 'beta=0.1+cut=3'),
               ('$24\\times24$ mesh', 'scan11.json', 'beta=0.1+cut=3')]),
]
BASE = ('base ($J=8$, $\\beta=1$)', 'scan1.json', None)

# Iteration runs: label, file, note
RUNS = [
    ('baseline (Stage 8 follow-up, seeds 3000--3002)', 'combined_s3000.json', 'fu'),
    ('$b=a$, $\\beta=0.1$, cutoff $e^{-3}$, $N_\\delta=5\\times10^4$', 'imp_nd50k.json', 'sp'),
    ('same, seeds 2000--2002', 'imp_s2000.json', 'sp'),
    ('same, $N_\\delta=2.5\\times10^4$, $N_B=2\\times10^4$, seeds 3003--3005', 'imp_small.json', 'sp'),
    ('same as row 2 with the direct field production', 'fp_N2000.json', 'sp'),
    ('$N=8000$, seeds 3003--3005', 'imp_N8000.json', 'sp'),
    ('$N=8000$, $N_f=2000$, faster tracer', 'fast_N8000.json', 'sp'),
    ('$N=8000$, field placement, seeds 4000--4002', 'final_field.json', 'sp'),
    ('$N=8000$, empirical placement, seeds 4000--4002', 'final_emp.json', 'sp'),
]


def scan_value(d, f, name):
    rows = json.loads((d / f).read_text())
    for r in rows:
        if r['name'] == name:
            return r
    return None


def run_stats(path, k_ref, b=10):
    d = json.loads(Path(path).read_text())
    ina = d['args']['inactive']
    bms, Ts, ks, Ws, sds = [], [], [], [], []
    for rep in d['replicas']:
        if rep.get('failure'):
            continue
        rows = rep['rows'][ina:]
        k = np.array([r['k_next'] for r in rows])
        n = len(k) // b
        bms.append(k[:n * b].reshape(n, b).mean(axis=1))
        Ts.append(sum(r['time'] for r in rows))
        ks.append(k.mean())
        Ws.append(np.mean([r['W_abs'] for r in rows]))
        sds.append(k.std(ddof=1))
    B = np.concatenate(bms)
    nb = len(B) / len(bms)
    sig = B.std(ddof=1) / math.sqrt(nb)
    ks = np.array(ks)
    return dict(k=ks.mean(), dk_pcm=1e5 * (ks.mean() - k_ref),
                sem_pcm=1e5 * sig / math.sqrt(len(ks)), sig=sig, T=np.mean(Ts),
                W=np.mean(Ws), sd_gen=np.mean(sds), n=len(ks))


def main():
    d, s8, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    ref = json.loads((s8 / 'reference.json').read_text())
    k_ref = np.mean([r['k'] for r in ref['replicas']])
    sd_ref = np.std([r['k'] for r in ref['replicas']], ddof=1)
    T_ref = np.mean([r['time_active'] for r in ref['replicas']])
    fom_ref = 1.0 / (sd_ref**2 * T_ref)
    summary = dict(reference=dict(k=k_ref, sd=sd_ref, T=T_ref), scans={}, runs={})

    # Scans: variance of Y_s per generation at N = 2000 (bar chart)
    base = scan_value(d, 'scan1.json', 'nd=1')  # placeholder for structure
    base = json.loads((d / 'scan13.json').read_text())[0]
    labels, vals, cols = [], [], []
    labels.append('reference config.\n($J=8$, $\\beta=0.1$, $e^{-3}$)')
    vals.append(base['s']['var'])
    cols.append('#1f5f8b')
    b1 = scan_value(d, 'bias1.json', 'beta=1+cut=3')
    labels.append('$\\beta=1$ (Stage 8)')
    vals.append(b1['s']['var'])
    cols.append('#555555')
    palette = ['#b03a2e', '#2e7d32', '#d68910', '#6c3483', '#117a65']
    rows_tex = []
    for gi, (group, items) in enumerate(SCANS):
        for lab, f, name in items:
            r = scan_value(d, f, name)
            if r is None:
                continue
            labels.append(lab)
            vals.append(r['s']['var'])
            cols.append(palette[gi % len(palette)])
            summary['scans'][f'{group}: {lab}'] = dict(var_s=r['s']['var'], var_f=r['f']['var'],
                                                     T_s=r['s']['T'], top10=r['s']['top10'])
            rows_tex.append(f"{group} & {lab} & {r['s']['var']:.3f} & {r['f']['var']:.4f} & "
                            f"{r['s']['top10']:.2f}\\\\")
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    x = np.arange(len(vals))
    ax.bar(x, vals, color=cols)
    ax.axhline(base['s']['var'], color='#1f5f8b', lw=0.8, ls=':')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha='right', fontsize=6.5)
    ax.set_yscale('log')
    ax.set_ylabel(r'Var$(Y_s)$ per generation, $N=2000$')
    fig.tight_layout()
    fig.savefig(out / 'speedup_scans.pdf', bbox_inches='tight')
    plt.close(fig)
    (out / 'speedup_scans.tex').write_text('\n'.join(rows_tex) + '\n')

    # Iteration runs
    fu = s8.parent / 'stage8_followup'
    rows_tex = []
    for lab, f, where in RUNS:
        p = (fu / f) if where == 'fu' else (d / f)
        if not p.exists():
            continue
        S = run_stats(p, k_ref)
        fom = 1.0 / (S['sig']**2 * S['T'])
        S['fom_ratio'] = fom / fom_ref
        summary['runs'][lab] = S
        rows_tex.append(f"{lab} & {S['n']} & ${S['dk_pcm']:+.0f}\\pm{S['sem_pcm']:.0f}$ & "
                        f"{S['W']:.2f} & {S['sd_gen']:.4f} & {S['T']:.0f} & "
                        f"{S['fom_ratio']:.3f}\\\\")
    (out / 'speedup_runs.tex').write_text('\n'.join(rows_tex) + '\n')

    # The bias building up: k_n by generation at N = 8000 (one replica each)
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    for lab, f, col in (('field placement', 'imp_N8000.json', '#1f5f8b'),
                        ('empirical placement', 'emp_3003.json', '#b03a2e'),
                        ('finer netting', 'fine_net.json', '#2e7d32'),
                        ('netting every 4th generation', 'net4_3003.json', '#d68910')):
        p = d / f
        if not p.exists():
            continue
        rows = json.loads(p.read_text())['replicas'][0]['rows']
        k = np.array([r['k_next'] for r in rows])
        m = np.convolve(k, np.ones(5) / 5, mode='valid')
        ax.plot(np.arange(len(m)) + 2, m, color=col, lw=1.0, label=lab)
    ax.axhline(k_ref, color='k', lw=1.0, label='CE reference')
    ax.set_xlabel('generation (seed 3003, $N=8000$; 5-generation running mean)')
    ax.set_ylabel('$k_n$')
    ax.legend(fontsize=6.5, ncol=2)
    fig.tight_layout()
    fig.savefig(out / 'speedup_bias_buildup.pdf', bbox_inches='tight')
    plt.close(fig)

    (out / 'speedup_summary.json').write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps(summary['runs'], indent=1, default=float))


if __name__ == '__main__':
    main()
