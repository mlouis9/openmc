"""Figures of the controlled residual study (controlled_residual_study.py).

    python controlled_residual_figures.py controlled_residual_study OUTDIR

Reads <prefix>.npz and <prefix>.json and writes controlled_*.png to OUTDIR.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from controlled_residual_study import (E_EDGES, E_RANGES, MU_EDGES, RANGE_NAMES,
                                       MATERIAL_NAMES, VARIANTS, SHOWN, CHANNELS,
                                       AMPLITUDES)

COLORS = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#ff7f0e', '#8c564b']
TAG_NAMES = {'s': 'scattering (tag s)', 'f': 'fission (tag f)'}


def _load(prefix):
    d = dict(np.load(prefix + '.npz'))
    j = json.loads(Path(prefix + '.json').read_text())
    return d, j


def _ref(d, t, b, a=(1.0, 1.0)):
    hi = a[0] * d[f'ref/s/{t}/{b}'][0] + a[1] * d[f'ref/f/{t}/{b}'][0]
    sd = np.hypot(a[0] * d[f'ref/s/{t}/{b}'][1], a[1] * d[f'ref/f/{t}/{b}'][1])
    return hi - a['sf'.index(t)] * d[f'F/{t}/{b}'], sd, hi


def _save(fig, out, name):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / name, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print('wrote', out / name)


def fig_energy(d, out):
    """Residual per unit lethargy against outgoing energy."""
    nE = len(E_EDGES) - 1
    du = np.log(E_EDGES[1:] / E_EDGES[:-1])
    Em = np.sqrt(E_EDGES[1:] * E_EDGES[:-1])
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharex=True)
    for i, t in enumerate('sf'):
        rr, sd, _ = _ref(d, t, 'energy')
        for m in (0, 1):
            ax = axes[i, m]
            sl = slice(m * nE, (m + 1) * nE)
            y, s = rr[sl] / du, sd[sl] / du
            ax.fill_between(Em, y - 2 * s, y + 2 * s, step='mid', color='0.75',
                            label='reference (2 s.e.)')
            ax.step(Em, y, where='mid', color='k', lw=1)
            for c, name in enumerate(SHOWN):
                e, se = d[f'est0/{name}/{t}/energy']
                off = 1 + 0.05 * (c - 1.5)
                ax.errorbar(Em[:] * off, e[sl] / du, yerr=se[sl] / du, fmt='o', ms=2.5,
                            color=COLORS[c], elinewidth=0.8, label=name)
            ax.set_xscale('log')
            ax.axhline(0, color='0.5', lw=0.5)
            ax.set_title(f'{TAG_NAMES[t]}, {MATERIAL_NAMES[m]}')
            if m == 0:
                ax.set_ylabel('residual per unit lethargy')
            if i == 1:
                ax.set_xlabel('outgoing energy [eV]')
            if t == 'f' and m == 1:
                ax.text(0.5, 0.5, 'no fission production', transform=ax.transAxes,
                        ha='center')
    axes[0, 0].legend(fontsize=7, loc='upper left')
    fig.suptitle('Residual binned in outgoing energy: estimators against the '
                 'first-collision reference')
    _save(fig, out, 'controlled_energy.png')


def fig_relative_noise(d, j, out):
    """Standard error over |reference residual| per energy bin."""
    nE = len(E_EDGES) - 1
    Em = np.sqrt(E_EDGES[1:] * E_EDGES[:-1])
    names = list(VARIANTS)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharex=True, sharey=True)
    for i, t in enumerate('sf'):
        rr, sd, hi = _ref(d, t, 'energy')
        for m in (0, 1):
            ax = axes[i, m]
            sl = slice(m * nE, (m + 1) * nE)
            for c, name in enumerate(names):
                if 'net' in name:
                    continue
                e, se = d[f'est0/{name}/{t}/energy']
                with np.errstate(divide='ignore', invalid='ignore'):
                    y = np.sqrt(j['n']) * se[sl] / np.abs(rr[sl])
                ax.plot(Em, y, '-' if 'density route' in name else '--',
                        color=plt.cm.tab10(c % 10), lw=1.2, label=name)
            with np.errstate(divide='ignore', invalid='ignore'):
                ax.plot(Em, np.abs(hi[sl]) / np.abs(rr[sl]), 'k:', lw=1.5,
                        label='|high term| / |residual|')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_title(f'{TAG_NAMES[t]}, {MATERIAL_NAMES[m]}')
            if m == 0:
                ax.set_ylabel(r'per-sample s.d. / |residual|')
            if i == 1:
                ax.set_xlabel('outgoing energy [eV]')
    axes[0, 0].legend(fontsize=6.5, loc='lower left')
    fig.suptitle('Noise relative to the residual, per energy bin (one sample)')
    _save(fig, out, 'controlled_relative_noise.png')


def fig_space(d, out):
    """Residual by region relative to the region's field, against centroid
    radius."""
    rad = d['centroid_radius']
    mat = d['region_mat']
    V = d['V']
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for i, t in enumerate('sf'):
        ax = axes[i]
        rr, sd, _ = _ref(d, t, 'space')
        F = d[f'F/{t}/space']
        live = F > 0
        ax.errorbar(rad[live], rr[live] / F[live], yerr=2 * sd[live] / F[live], fmt='s',
                    ms=4, color='k', mfc='none', label='reference (2 s.e.)', zorder=5)
        for c, name in enumerate(SHOWN):
            e, se = d[f'est0/{name}/{t}/space']
            ax.errorbar(rad[live] + 0.006 * (c - 1.5), e[live] / F[live],
                        yerr=se[live] / F[live], fmt='o', ms=2.5, color=COLORS[c],
                        elinewidth=0.8, label=name)
        ax.axvline(0.4096, color='0.5', ls='--', lw=0.8)
        ax.axhline(0, color='0.5', lw=0.5)
        ax.set_ylim(-0.6, 0.6)
        ax.set_xlabel('region centroid radius [cm] (pin radius dashed)')
        ax.set_ylabel(r'residual / field in the region, $r_R / F_R$')
        ax.set_title(TAG_NAMES[t])
    axes[0].legend(fontsize=7)
    fig.suptitle('Residual by source region relative to the field there (all energies; '
                 'axis clipped at +/- 0.6)')
    _save(fig, out, 'controlled_space.png')


def fig_angle(d, out):
    """Residual against the radial cosine, by material and energy range."""
    nm = len(MU_EDGES) - 1
    mu = 0.5 * (MU_EDGES[1:] + MU_EDGES[:-1])
    t = 's'
    rr, sd, _ = _ref(d, t, 'angle')
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    for m in (0, 1):
        for e in range(3):
            ax = axes[m, e]
            sl = slice((m * 3 + e) * nm, (m * 3 + e + 1) * nm)
            ax.fill_between(mu, rr[sl] - 2 * sd[sl], rr[sl] + 2 * sd[sl], step='mid',
                            color='0.75', label='reference (2 s.e.)')
            ax.step(mu, rr[sl], where='mid', color='k', lw=1)
            for c, name in enumerate(SHOWN):
                est, se = d[f'est0/{name}/{t}/angle']
                ax.errorbar(mu + 0.02 * (c - 1.5), est[sl], yerr=se[sl], fmt='o', ms=2.5,
                            color=COLORS[c], elinewidth=0.8, label=name)
            ax.axhline(0, color='0.5', lw=0.5)
            ax.set_title(f'{MATERIAL_NAMES[m]}, {RANGE_NAMES[e]}', fontsize=9)
            if m == 1:
                ax.set_xlabel(r'radial cosine $\mu_\rho = \Omega\cdot e_\rho$')
    axes[0, 0].legend(fontsize=6.5)
    fig.suptitle('Scattering residual against outgoing direction')
    _save(fig, out, 'controlled_angle.png')


def fig_scans(j, out):
    groups = [('incident energies K', ['K, uncontrolled', 'K, independent',
                                       'K, common numbers']),
              ('directions J', ['J']), ('region samples M_II', ['M_II']),
              ('roulette cutoff x (cutoff exp(-x))',
               ['cutoff, uncontrolled', 'cutoff, flux difference (M_II = 128)'])]
    groups = [g for g in groups if all(s in j['scans'] for s in g[1])]
    fig, axes = plt.subplots(2, len(groups), figsize=(4.8 * len(groups), 8))
    for i, t in enumerate('sf'):
        for c, (xlabel, scans) in enumerate(groups):
            ax = axes[i, c]
            for s in scans:
                rows = j['scans'][s]
                x = [r['x'] for r in rows]
                v = [r[t]['per_sample_variance'] for r in rows]
                vh = [r[t].get('high_only_variance', np.nan) for r in rows]
                vt = [r[t].get('high_only_variance', np.nan) * r[t]['time_per_sample']
                      for r in rows]
                l, = ax.plot(x, v, 'o-', label=f'{s}: residual')
                ax.plot(x, vh, 's--', color=l.get_color(), label=f'{s}: high term only')
                ax.plot(x, np.array(vt) / vt[0] * vh[0], 'x:', color=l.get_color(),
                        label=f'{s}: high term x time (scaled to K, J, M = 1)')
            if 'cutoff' not in xlabel:
                ax.set_xscale('log', base=2)
            ax.set_yscale('log')
            ax.set_xlabel(xlabel)
            ax.set_title(TAG_NAMES[t], fontsize=9)
            if c == 0:
                ax.set_ylabel('per-sample variance, summed over energy bins')
            ax.legend(fontsize=5.5)
    fig.suptitle('Density-free estimator: variance against K, J, M_II, and the '
                 'roulette cutoff (flux-difference control unless noted)')
    _save(fig, out, 'controlled_scans.png')


def fig_decomposition(j, out):
    names = [n for n in VARIANTS if 'net' not in n]
    keys = ['kind:density', 'kind:sampler', 'kind:region'] + [f'channel:{c}' for c in CHANNELS]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)
    for i, t in enumerate('sf'):
        ax = axes[i]
        x = np.arange(len(names))
        width = 0.11
        tot = [j['variants'][f'{n} | {t}']['binnings']['energy']['per_sample_variance']
               for n in names]
        ax.bar(x - 4 * width, tot, width, color='k', label='total')
        for c, key in enumerate(keys):
            v = [j['variants'][f'{n} | {t}']['decomposition'].get(key, 0.0) for n in names]
            ax.bar(x + (c - 3) * width, v, width, label=key)
        ax.set_yscale('log')
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace(', ', ',\n', 1) for n in names], rotation=60,
                           ha='right', fontsize=7)
        ax.set_title(TAG_NAMES[t])
        ax.set_ylabel('per-sample variance of the subset, summed over energy bins')
    axes[0].legend(fontsize=7)
    fig.suptitle('Where the variance comes from: record kind and channel')
    _save(fig, out, 'controlled_decomposition.png')


def noise_fits(d, j):
    """Per variant and tag: the least-squares fit, over every bin of every
    binning where the reference residual is resolved (|r| > 3 s.e.),
    log s = alpha + beta_r log|r| + beta_h log|h|, with s the per-sample
    standard deviation in the bin, r the reference residual, and h the
    first-collision production (|r| > 2 s.e. here). beta_r near 1 and beta_h near 0 is
    residual-scale noise; beta_r near 0 and beta_h near 1 is
    production-scale noise."""
    out = {}
    for name, prefix, suffix in [(n, p, x) for n in VARIANTS
                                 for p, x in (('est0', ''), ('high', ' (high term only)'))]:
        if f'{prefix}/{name}/s/energy' not in d:
            continue
        for t in 'sf':
            X, y = [], []
            for b in ('energy', 'space', 'angle'):
                rr, sd, hi = _ref(d, t, b)
                e, se = d[f'{prefix}/{name}/{t}/{b}']
                ok = (np.abs(rr) > 2 * sd) & (hi > 0) & (se > 0)
                X.append(np.column_stack([np.ones(ok.sum()), np.log(np.abs(rr[ok])),
                                          np.log(hi[ok])]))
                y.append(np.log(np.sqrt(j['n']) * se[ok]))
            X, y = np.vstack(X), np.concatenate(y)
            if len(y) < 5:
                continue
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            out[f'{name}{suffix} | {t}'] = dict(bins=int(len(y)), beta_r=float(coef[1]),
                                                beta_h=float(coef[2]))
    return out


def fig_noise_scaling(d, j, out):
    names = [n for n in VARIANTS if 'net' not in n and 'tandem' not in n]
    fits = noise_fits(d, j)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))
    for i, t in enumerate('sf'):
        ax, bx, cx = axes[i]
        lo, hi_max = np.inf, 0.0
        for c, name in enumerate(names):
            mk = 'o' if 'density route' in name else '^'
            first = True
            for b in ('energy', 'space', 'angle'):
                rr, sd, hi = _ref(d, t, b)
                e, se = d[f'est0/{name}/{t}/{b}']
                ok = (np.abs(rr) > 2 * sd) & (hi > 0) & (se > 0)
                okh = (hi > 0) & (se > 0)
                if not okh.any():
                    continue
                sig = np.sqrt(j['n']) * se[ok]
                f = fits.get(f'{name} | {t}')
                lab = (f'{name}: beta_r {f["beta_r"]:+.2f}, beta_h {f["beta_h"]:+.2f}'
                       if (first and f) else None)
                ax.loglog(np.abs(rr[ok]), sig, mk, ms=2.5, alpha=0.6,
                          color=plt.cm.tab10(c))
                bx.loglog(hi[okh], np.sqrt(j['n']) * se[okh], mk, ms=2.5, alpha=0.6,
                          color=plt.cm.tab10(c), label=lab)
                if ok.any():
                    lo = min(lo, np.abs(rr[ok]).min())
                hi_max = max(hi_max, hi[okh].max())
                first = False
        if np.isfinite(lo):
            xx = np.array([lo, hi_max])
            for ax_ in (ax, bx):
                ax_.plot(xx, xx, 'k:', lw=0.8)
        ax.set_xlabel('|reference residual| in the bin (resolved: > 2 s.e.)')
        bx.set_xlabel('first-collision production in the bin (every bin)')
        for ax_ in (ax, bx):
            ax_.set_ylabel('per-sample s.d. in the bin')
            ax_.set_title(f'{TAG_NAMES[t]}: bins of every binning', fontsize=9)
        bx.legend(fontsize=5.5, loc='lower right')

        # Amplitude scan: the scanned amplitude is the tag's own
        which = 'a_s' if t == 's' else 'a_f'
        x = np.array(AMPLITUDES)
        rnorm = []
        for v in x:
            a = (v, 1.0) if which == 'a_s' else (1.0, v)
            rr, sd, _ = _ref(d, t, 'energy', a)
            rnorm.append(math.sqrt(max(np.sum(rr**2) - np.sum(sd**2), 0.0)))
        rnorm = np.array(rnorm)
        i1 = list(x).index(1.0)
        cx.plot(x, rnorm / rnorm[i1], 'k--', lw=2, label='||reference residual|| (bias-corrected)')
        for c, name in enumerate(names):
            key = f'{name} | {t}'
            if key not in j['amplitude']:
                continue
            rows = [r for r in j['amplitude'][key] if r['which'] == which]
            rows = sorted(rows, key=lambda r: r['x'])
            nz = np.array([r['noise'] for r in rows])
            cx.plot([r['x'] for r in rows], nz / nz[i1], 'o-' if 'density route' in name
                    else '^-', ms=3, color=plt.cm.tab10(c), lw=0.9, label=name)
        cx.set_yscale('log')
        cx.set_xlabel(f'amplitude {which} (other amplitude 1, b = a)')
        cx.set_ylabel('relative to a = (1, 1)')
        cx.set_title(f'{TAG_NAMES[t]}: noise and residual against {which}', fontsize=9)
        cx.legend(fontsize=6)
    fig.suptitle('Does the noise scale with the residual? Bins (left, centre) and an '
                 'amplitude scan of the same records (right)')
    _save(fig, out, 'controlled_noise_scaling.png')
    return fits


def fig_low_term(j, out):
    """Per-sample variance of the whole residual against the high term
    alone (the low term removed exactly)."""
    names = [n for n in VARIANTS if 'net' not in n]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for i, t in enumerate('sf'):
        ax = axes[i]
        x = np.arange(len(names))
        for c, b in enumerate(('energy', 'space', 'angle')):
            tot = [j['variants'][f'{n} | {t}']['binnings'][b]['per_sample_variance']
                   for n in names]
            hi = [j['variants'][f'{n} | {t}'].get('high_only_variance', {}).get(b, np.nan)
                  for n in names]
            ax.bar(x + (2 * c - 2.5) * 0.13, tot, 0.13, color=plt.cm.tab10(c),
                   label=f'{b}: residual')
            ax.bar(x + (2 * c - 1.5) * 0.13, hi, 0.13, color=plt.cm.tab10(c), alpha=0.45,
                   hatch='//', label=f'{b}: high term only')
        ax.set_yscale('log')
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace(', ', ',\n', 1) for n in names], rotation=60,
                           ha='right', fontsize=7)
        ax.set_ylabel('per-sample variance, summed over bins')
        ax.set_title(TAG_NAMES[t])
    axes[0].legend(fontsize=7)
    fig.suptitle('The low term at the destination: residual against high term alone')
    _save(fig, out, 'controlled_low_term.png')


def fig_channels(j, out):
    Em = np.sqrt(E_EDGES[1:] * E_EDGES[:-1])
    nE = len(E_EDGES) - 1
    keys = [k for k in j['channels'] if k.endswith('| s')]
    fig, axes = plt.subplots(1, len(keys), figsize=(4.5 * len(keys), 4.5), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, key in zip(axes, keys):
        for c, (cn, v) in enumerate(j['channels'][key].items()):
            p = np.abs(np.array(v['production']))
            s = np.array(v['noise'])
            for m, ls in ((0, '-'), (1, '--')):
                sl = slice(m * nE, (m + 1) * nE)
                with np.errstate(divide='ignore', invalid='ignore'):
                    ax.plot(Em, s[sl] / p[sl], ls, color=plt.cm.tab10(c),
                            label=f'{cn}, {MATERIAL_NAMES[m]}')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim(0.5, 2e3)
        ax.set_xlabel('outgoing energy [eV]')
        ax.set_title(key.split(' | ')[0], fontsize=8)
    axes[0].set_ylabel('per-sample s.d. / channel production (per bin)')
    axes[0].legend(fontsize=6.5)
    fig.suptitle('Scattering channels on the sampler route: noise relative to the '
                 "channel's own production")
    _save(fig, out, 'controlled_channels.png')


def fig_z(d, j, out):
    names = list(VARIANTS)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    zz = np.linspace(-5, 5, 41)
    for c, name in enumerate(names):
        z = []
        for t in 'sf':
            for b in ('energy', 'space', 'angle'):
                rr, sd, hi = _ref(d, t, b)
                e, se = d[f'est0/{name}/{t}/{b}']
                err = np.hypot(se, sd)
                live = resolved(se, sd, hi)
                z.append((e[live] - rr[live]) / err[live])
        z = np.concatenate(z)
        h, _ = np.histogram(np.clip(z, -4.99, 4.99), zz, density=True)
        ax.step(0.5 * (zz[1:] + zz[:-1]), h, where='mid', color=plt.cm.tab10(c % 10),
                label=f'{name} (rms {np.sqrt(np.mean(z**2)):.2f})')
    x = np.linspace(-5, 5, 200)
    ax.plot(x, np.exp(-x**2 / 2) / math.sqrt(2 * math.pi), 'k', lw=2, label='N(0, 1)')
    ax.set_xlabel('z = (estimate - reference) / combined s.e., every bin')
    ax.legend(fontsize=6, loc='upper right')
    ax.set_title('Bias check over all binnings and both tags')
    _save(fig, out, 'controlled_z.png')


def resolved(se, sd, hi):
    """Bins where a z-score means something: the estimate has a standard
    error, and the reference has about ten events or more (relative standard
    error below 0.3). With fewer, the reference's own error is far from
    normal, and an estimator much quieter than it shows false bias."""
    return (se > 0) & (sd > 0) & (sd < 0.3 * np.abs(hi))


def bias_table(d):
    """chi^2 over the resolved bins, per variant, tag, and binning."""
    out = {}
    for name in VARIANTS:
        for t in 'sf':
            for b in ('energy', 'space', 'angle'):
                rr, sd, hi = _ref(d, t, b)
                e, se = d[f'est0/{name}/{t}/{b}']
                live = resolved(se, sd, hi)
                z = (e[live] - rr[live]) / np.hypot(se[live], sd[live])
                out[f'{name} | {t} | {b}'] = dict(
                    chi2=float(np.sum(z**2)), dof=int(live.sum()),
                    max_abs_z=float(np.abs(z).max()) if z.size else 0.0)
    return out


def make_all(prefix, out):
    d, j = _load(prefix)
    fig_energy(d, out)
    fig_relative_noise(d, j, out)
    fig_space(d, out)
    fig_angle(d, out)
    fig_scans(j, out)
    fig_decomposition(j, out)
    fits = fig_noise_scaling(d, j, out)
    Path(prefix + '_fits.json').write_text(json.dumps(
        dict(noise_fits=fits, bias=bias_table(d)), indent=1))
    fig_low_term(j, out)
    fig_channels(j, out)
    fig_z(d, j, out)


if __name__ == '__main__':
    make_all(sys.argv[1], sys.argv[2])
