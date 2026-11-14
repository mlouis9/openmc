"""Nuclear data figures of the RMC report (docs/report). Reads the HDF5
library named by OPENMC_CROSS_SECTIONS and the pincell's lift file.

    python nuclear_data_figures.py LIFTS_H5 OUTDIR FIELD_H5
"""
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import openmc.data

plt.rcParams.update({'font.size': 9, 'axes.titlesize': 9, 'legend.fontsize': 7.5,
                     'font.family': 'serif'})
LIB = Path(os.environ['OPENMC_CROSS_SECTIONS']).parent / 'neutron'
C = dict(blue='#1f5f8b', red='#b03a2e', green='#2e7d32', gray='#555555',
         orange='#d68910', purple='#6c3483')


def nuc(name):
    return openmc.data.IncidentNeutron.from_hdf5(LIB / f'{name}.h5')


def xs(n, mt, T='294K'):
    r = n.reactions[mt]
    x = r.xs[T]
    E = n.energy[T][x._threshold_idx:]
    return E, x(E)


def save(fig, out, name):
    fig.savefig(Path(out) / name, bbox_inches='tight')
    plt.close(fig)
    print('wrote', name)


def fig_cross_sections(out):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2))
    for ax, name in zip(axes.ravel(), ['U235', 'U238', 'C0', 'H1']):
        n = nuc(name)
        for mt, lab, col in [(1, 'total', 'black'), (2, 'elastic', C['blue']),
                             (102, r'capture $(n,\gamma)$', C['green']),
                             (18, 'fission', C['red']), (4, 'inelastic', C['orange'])]:
            if mt in n.reactions or mt == 1:
                if mt == 1:
                    E = n.energy['294K']
                    tot = sum(n.reactions[m].xs['294K'](E) for m in n.reactions
                              if m in (2, 102, 18, 4, 16, 17, 103, 107, 37, 22, 28))
                    ax.loglog(E, tot, color=col, lw=0.6, label=lab)
                else:
                    E, y = xs(n, mt)
                    ax.loglog(E, np.maximum(y, 1e-6), color=col, lw=0.6, label=lab)
        ax.set_title(name.replace('0', ' (natural)') if name == 'C0' else name)
        ax.set_xlim(1e-5, 2e7)
        ax.set_ylim(1e-4, 1e5 if name.startswith('U') else 1e3)
        ax.set_xlabel('incident energy [eV]')
        ax.set_ylabel('cross section [b]')
    axes[0, 0].legend(loc='lower left')
    fig.tight_layout()
    save(fig, out, 'nd_cross_sections.pdf')


def material_flux(field):
    """Volume-weighted MG flux per material id (ascending groups)."""
    with h5py.File(field) as f:
        G = int(f.attrs['n_groups'])
        phi = f['regions/phi'][()].reshape(-1, G)
        V = f['regions/volume_rr'][()]
        mat = f['regions/material_id'][()]
    return {int(m): (V[mat == m, None] * phi[mat == m]).sum(axis=0) for m in set(mat)}


def continuous_flux(lifts, field):
    """phi_g h^phi(E) on the fine grid, per material."""
    with h5py.File(lifts) as f:
        fine = f['fine_edges'][()]
        mids = f['material_ids'][()]
        hphi = f['h_phi'][()].reshape(len(mids), -1)
        groups = f['group_edges'][()]
    mid = 0.5 * (fine[:-1] + fine[1:])
    g = np.clip(np.searchsorted(groups, mid, side='right') - 1, 0, len(groups) - 2)
    phi = material_flux(field)
    return mid, {int(m): phi[int(m)][g] * hphi[i] for i, m in enumerate(mids)}


def fig_resonances(out, lifts):
    n = nuc('U238')
    E, el = xs(n, 2)
    _, cap = xs(n, 102)
    mid, flux = continuous_flux(lifts, FIELD)
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    m = (E > 1) & (E < 200)
    ax.semilogy(E[m], el[m] + cap[m], color='black', lw=0.6, label=r'U-238 $\sigma_t$ (elastic + capture)')
    ax.set_xlabel('energy [eV]')
    ax.set_ylabel('cross section [b]')
    ax.set_xlim(1, 200)
    ax.set_xscale('log')
    ax2 = ax.twinx()
    sel = (mid > 1) & (mid < 200)
    for mid_id, y0 in flux.items():
        lab = 'fuel' if mid_id == 1 else 'graphite'
        col = C['red'] if mid_id == 1 else C['blue']
        y = y0 * mid
        ax2.plot(mid[sel], y[sel] / np.nanmax(y[sel]), color=col, lw=0.8,
                 label=f'flux per unit lethargy, {lab} ($\\phi_g h^\\phi$)')
    ax2.set_ylabel(r'$E\,\phi(E)$ (normalized)')
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='lower left')
    save(fig, out, 'nd_resonances.pdf')


def fig_spectrum(out, lifts):
    with h5py.File(lifts) as f:
        fine = f['fine_edges'][()]
        M = len(f['material_ids'][()])
        hphi = f['h_phi'][()].reshape(M, -1)
        hs = f['h_s'][()].reshape(M, -1)
        hf = f['h_f'][()].reshape(M, -1)
        mids = f['material_ids'][()]
        groups = f['group_edges'][()]
    mid = 0.5 * (fine[:-1] + fine[1:])
    _, flux = continuous_flux(lifts, FIELD)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for i, mid_id in enumerate(mids):
        lab = 'fuel' if mid_id == 1 else 'graphite'
        col = C['red'] if mid_id == 1 else C['blue']
        y = mid * flux[int(mid_id)]
        axes[0].loglog(mid, np.maximum(y / y.max(), 1e-12), color=col, lw=0.7, label=lab)
        axes[1].loglog(mid, np.maximum(mid * hs[i], 1e-12), color=col, lw=0.7,
                       label=f'$h_s$, {lab}')
        if mid_id == 1:
            axes[1].loglog(mid, np.maximum(mid * hf[i], 1e-12), color=C['orange'], lw=0.7,
                           label='$h_f$, fuel')
    for ax in axes:
        for g in groups[1:-1]:
            ax.axvline(g, color='0.8', lw=0.5, zorder=0)
        ax.set_xlabel('energy [eV]')
        ax.set_xlim(1e-4, 2e7)
    axes[0].set_ylim(1e-6, 1)
    axes[1].set_ylim(1e-8, 1)
    axes[0].set_ylabel(r'$E\,\phi_g h^\phi(E)$ (normalized)')
    axes[1].set_ylabel(r'$E\,h_t(E)$ within group')
    axes[0].set_title('flux per unit lethargy (CASMO-8 groups in grey)')
    axes[1].set_title('emission shapes')
    axes[0].legend()
    axes[1].legend()
    fig.tight_layout()
    save(fig, out, 'nd_spectrum_lifts.pdf')


def fig_fission_spectra(out):
    E = np.geomspace(1e3, 2e7, 400)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for name, col in [('U235', C['red']), ('U238', C['blue'])]:
        n = nuc(name)
        rx = n.reactions[18]
        prompt = [p for p in rx.products if p.particle == 'neutron'
                  and p.emission_mode == 'prompt'][0]
        dist = prompt.distribution[0]
        law = dist.energy if hasattr(dist, 'energy') else dist
        for Ein, ls in [(0.0253, '-'), (2e6, '--'), (1.4e7, ':')]:
            if name == 'U238' and Ein < 1e6:
                continue
            try:
                i = np.searchsorted(law.energy, Ein) - 1
                i = max(i, 0)
                tab = law.energy_out[i]
                axes[0].semilogx(tab.x, tab.p, ls, color=col, lw=0.8,
                                 label=f'{name}, $E\'$ = {Ein:.3g} eV')
            except Exception:
                pass
        nu = rx.products[0].yield_
        Eg = np.geomspace(1e-5, 2e7, 300)
        axes[1].semilogx(Eg, nu(Eg), color=col, lw=0.8, label=fr'{name} $\bar\nu_p$')
    axes[0].set_xlim(1e3, 2e7)
    axes[0].set_xlabel('outgoing energy $E$ [eV]')
    axes[0].set_ylabel(r'$\chi_p(E\mid E\')$ [1/eV]')
    axes[0].legend()
    axes[1].set_xlabel("incident energy $E'$ [eV]")
    axes[1].set_ylabel(r'prompt neutrons per fission')
    axes[1].legend()
    fig.tight_layout()
    save(fig, out, 'nd_fission.pdf')


def fig_elastic(out):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    # Isotropic CM: E/E' uniform on [alpha, 1]
    for A, col in [(1.0, C['red']), (12.0, C['blue']), (238.0, C['green'])]:
        alpha = ((A - 1) / (A + 1))**2
        x = np.linspace(0, 1, 400)
        y = np.where(x >= alpha, 1 / (1 - alpha), 0.0)
        axes[0].plot(x, y if A > 1 else np.ones_like(x), color=col, lw=0.9,
                     label=fr'$A$ = {A:g}, $\alpha$ = {alpha:.3f}')
    axes[0].set_yscale('log')
    axes[0].set_ylim(0.5, 300)
    axes[0].set_xlabel(r"$E/E'$")
    axes[0].set_ylabel('density (isotropic in CM)')
    axes[0].legend()
    n = nuc('C0')
    ang = n.reactions[2].products[0].distribution[0].angle
    for Ein, ls in [(1e5, '-'), (2e6, '--'), (6e6, '-.'), (1.4e7, ':')]:
        i = max(np.searchsorted(ang.energy, Ein) - 1, 0)
        d = ang.mu[i]
        mu = np.linspace(-1, 1, 400)
        if hasattr(d, 'x') and hasattr(d, 'p'):
            pm = np.interp(mu, d.x, d.p)
        else:
            pm = d(mu)
        axes[1].plot(mu, pm, ls, color=C['blue'], lw=0.9,
                     label=f"C, $E'$ = {ang.energy[i] / 1e6:.2g} MeV")
    axes[1].set_xlabel(r'$\mu_{\rm cm}$')
    axes[1].set_ylabel(r'$p(\mu_{\rm cm}\mid E\')$')
    axes[1].legend()
    fig.tight_layout()
    save(fig, out, 'nd_elastic.pdf')


def fig_inelastic(out):
    n = nuc('U238')
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    for mt in range(51, 60):
        if mt in n.reactions:
            E, y = xs(n, mt)
            q = n.reactions[mt].q_value
            ax.semilogy(E / 1e6, y, lw=0.7, label=f'MT {mt} (level, $-Q$ = {-q / 1e3:.0f} keV)')
    for mt, lab in [(91, 'MT 91 (continuum)'), (16, r'MT 16 $(n,2n)$'), (18, 'MT 18 fission')]:
        E, y = xs(n, mt)
        ax.semilogy(E / 1e6, y, lw=1.1, label=lab)
    ax.set_xlim(0, 15)
    ax.set_ylim(1e-3, 5)
    ax.set_xlabel('incident energy [MeV]')
    ax.set_ylabel('cross section [b]')
    ax.legend(ncol=2, fontsize=6.5)
    save(fig, out, 'nd_inelastic.pdf')


FIELD = None

if __name__ == '__main__':
    lifts, out, FIELD = sys.argv[1], sys.argv[2], sys.argv[3]
    Path(out).mkdir(parents=True, exist_ok=True)
    fig_cross_sections(out)
    fig_resonances(out, lifts)
    fig_spectrum(out, lifts)
    fig_fission_spectra(out)
    fig_elastic(out)
    fig_inelastic(out)
