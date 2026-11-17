"""A preliminary RESOLVE energy representation (Mylotte and Forget,
"Reduced Spectral Operator by Latent Variable Encoding") for the forward
residual's partner (rmc.forward_points part 5).

Per material and per group (the energy blocks), on a fine lethargy grid:

1. a starting block of flux-per-lethargy columns: the constant column, the
   resolvent family 1/(Sigma_t + Sigma_b), the field's emission and flux
   lifts, and those lifts attenuated by exp(-l Sigma_t) of every material
   and of fuel-then-graphite paths;
2. compressed to width b (constant column first, then the leading singular
   vectors of the rest), in the inner product weighted by the lethargy width;
3. a block Lanczos recursion of depth m against Sigma_t (full
   reorthogonalization, deflation of dependent columns), giving the
   orthonormal basis Q with L <= m b columns;
4. the projected total cross section J = Q^T W Sigma_t Q diagonalized,
   J = S D S^T: the lanes are U = Q S, and lane k attenuates as exp(-D_k l).

Crossing matrices R = U_to^T W U_from map lane coefficients between
materials (all materials share the grid of a group). Source vectors are the
lane coefficients of each region's emission lift.

This is preliminary: the starting block uses the problem's own lifts instead
of a library of homogeneous slowing-down solutions, and only the attenuation
and the source are represented in lanes; the partner applies the exact CE
emission law to the decoded flux (forward_points part 5).
"""
import os

import numpy as np


def union_grid(nuclides, lo, hi, max_du=2e-3, temperature='294K'):
    """Union of the nuclides' energy grids in [lo, hi], refined so that no
    lethargy interval exceeds max_du. Returns ascending edges."""
    import openmc.data
    lib = openmc.data.DataLibrary.from_xml(os.environ['OPENMC_CROSS_SECTIONS'])
    pts = [np.array([lo, hi])]
    for n in nuclides:
        entry = lib.get_by_material(n)
        if entry is None:
            continue
        data = openmc.data.IncidentNeutron.from_hdf5(entry['path'])
        E = data.energy[temperature]
        pts.append(E[(E > lo) & (E < hi)])
    E = np.unique(np.concatenate(pts))
    # refine coarse intervals
    du = np.diff(np.log(E))
    extra = [np.exp(np.linspace(np.log(E[i]), np.log(E[i + 1]), int(np.ceil(du[i] / max_du)) + 1)[1:-1])
             for i in np.flatnonzero(du > max_du)]
    if extra:
        E = np.unique(np.concatenate([E] + extra))
    return E


def _orth_against(Z, Qs):
    for _ in range(2):
        for Q in Qs:
            Z = Z - Q @ (Q.T @ Z)
    return Z


def lanes(cols, sigma, w, b, m, tol=1e-9):
    """Lanes of one material and block. cols: starting columns [n, c] (flux
    per lethargy; column 0 is the constant column); sigma: Sigma_t on the
    grid [n]; w: lethargy widths [n]. Returns U [n, L] (function values) and
    D [L]."""
    sw = np.sqrt(w)
    Y = cols * sw[:, None]
    q0 = Y[:, :1] / np.linalg.norm(Y[:, 0])
    rest = Y[:, 1:] - q0 @ (q0.T @ Y[:, 1:])
    rest = rest / np.maximum(np.linalg.norm(rest, axis=0), 1e-300)
    # leading left singular vectors through the Gram matrix (memory: one copy)
    lam, V = np.linalg.eigh(rest.T @ rest)
    order = np.argsort(lam)[::-1]
    lam, V = lam[order], V[:, order]
    keep = min(b - 1, int((lam > tol**2 * lam[0]).sum()))
    u = rest @ (V[:, :keep] / np.sqrt(lam[:keep]))
    u, _ = np.linalg.qr(u - q0 @ (q0.T @ u))
    del rest
    blocks = [np.hstack([q0, u])]
    for _ in range(1, m):
        Z = sigma[:, None] * blocks[-1]
        Z = _orth_against(Z, blocks)
        uz, sz, _ = np.linalg.svd(Z, full_matrices=False)
        k = int((sz > tol * np.abs(sigma).max()).sum())
        if k == 0:
            break
        blocks.append(uz[:, :k])
    Q = np.hstack(blocks)
    J = Q.T @ (sigma[:, None] * Q)
    D, S = np.linalg.eigh(0.5 * (J + J.T))
    Uhat = Q @ S
    return Uhat / sw[:, None], D, Uhat


def generic_spectra(rmc, mats, Em, Sb):
    """Problem-independent spectra per lethargy: a Watt fission spectrum, a
    294 K Maxwellian, and for every material the narrow-resonance flux
    1/(Sigma_t + Sigma_b) and its elastic emission Sigma_el/(Sigma_t +
    Sigma_b), over the background cross sections Sb."""
    a, b = 0.988e6, 2.249e-6
    watt = np.exp(-Em / a) * np.sinh(np.sqrt(b * Em)) * Em
    kT = 0.0253
    maxw = (Em / kT) ** 2 * np.exp(-Em / kT)
    out = [watt, maxw]
    for mid in mats.values():
        st, sel, _ = rmc.xs(mid, Em)
        for s in Sb[::2]:
            out += [1.0 / (st + s), sel / (st + s)]
    out = [v / v.max() for v in out if v.max() > 0]
    return np.array(out)


def build(rmc, mats, region_mat, groups, grids, b, m, Sb=np.logspace(-2, 2, 9),
          ells=np.logspace(-3, 0.5, 8), pair=(0.03, 0.2, 0.8), generic=False):
    """The library for rmc.resolve_load. mats: {material index: material id};
    region_mat: material index of every field region (-1 void); groups: the
    field's group edges; grids: {g: edges}. With generic, the starting block
    holds no spectrum of the problem (Watt, Maxwellian and narrow-resonance
    shapes, attenuated); the sources are still the field's lifts, projected.
    Returns (blocks, cross, sources, info)."""
    R = len(region_mat)
    regions = np.arange(R)
    blocks, cross, sources, uhats, info = {}, {}, {}, {}, {}
    for g, edges in grids.items():
        Em = np.sqrt(edges[:-1] * edges[1:])
        w = np.diff(np.log(edges))
        sig = {mi: rmc.xs_total(mid, Em) for mi, mid in mats.items()}
        # the field's lifts in this group, per lethargy, distinct columns only
        spec = []
        for r in regions[region_mat >= 0]:
            for which in ('s', 'f', 'phi'):
                h = rmc.field_shape(np.full(Em.size, r), Em, which) * Em
                if h.sum() > 0:
                    spec.append(h)
        spec = np.unique(np.round(np.array(spec), 12), axis=0)
        lifts = []
        if generic:
            # 'src': the field's own lifts enter unattenuated (a RESOLVE field
            # would hold its emission in the basis by construction)
            if generic == 'src':
                lifts = list(spec)
            spec = generic_spectra(rmc, mats, Em, Sb)
        fuel = [mi for mi in mats if sig[mi].max() > 10 * sig[mi].min()]
        for mi in mats:
            cols = [np.ones_like(Em)]
            cols += [1.0 / (sig[mi] + s) for s in Sb]
            cols += list(spec) + lifts
            for mj in mats:
                for l in ells:
                    cols += list(spec * np.exp(-l * sig[mj])[None, :])
            others = [mj for mj in mats if mj != mi] + [mi]
            for l1 in (pair if not generic else pair[1:]):
                for l2 in (pair if not generic else pair[1:]):
                    for f in fuel:
                        for o in others:
                            cols += list(spec * np.exp(-l1 * sig[f] - l2 * sig[o])[None, :])
            cols = np.array(cols, dtype=np.float64).T
            U, D, Uhat = lanes(cols, sig[mi], w, b, m)
            del cols
            blocks[(mi, g)] = (edges, U, D)
            uhats[(mi, g)] = Uhat
            info[(mi, g)] = dict(n=Em.size, L=D.size, Dmin=float(D.min()), Dmax=float(D.max()))
        for a in mats:
            for c in mats:
                if a != c:
                    cross[(a, c, g)] = uhats[(c, g)].T @ uhats[(a, g)]
        sw = np.sqrt(w)
        for r in regions[region_mat >= 0]:
            Uhat = uhats[(int(region_mat[r]), g)]
            for t, which in enumerate(('s', 'f')):
                h = rmc.field_shape(np.full(Em.size, r), Em, which) * Em
                if h.sum() > 0:
                    sources[(int(r), t, g)] = Uhat.T @ (sw * h)
    return blocks, cross, sources, info


def build_subgroup(rmc, mats, region_mat, grids, n_windows, n_levels, spacing='quantile'):
    """The subgroup limit of RESOLVE: lanes are indicator functions of bands,
    each band the set of fine bins of one lethargy window of the group whose
    fuel total cross section lies in one level interval (quantiles of
    log Sigma_t in the window). The bands are orthogonal, the projected total
    cross section is diagonal (the band's lethargy-mean Sigma_t of each
    material), every material shares the bands (crossing matrices are the
    identity), and the decoded flux is positive. Returns the same tuple as
    build(); info[(m, g)] also holds 'band' and 'mass' (for the proposal
    library of forward_points part 8, and for a subgroup partner) and 'sigma_el', the elastic cross
    section on the grid (the incident-energy draw weight)."""
    R = len(region_mat)
    regions = np.arange(R)
    blocks, cross, sources, info = {}, {}, {}, {}
    for g, edges in grids.items():
        Em = np.sqrt(edges[:-1] * edges[1:])
        u = np.log(Em)
        w = np.diff(np.log(edges))
        sig = {mi: rmc.xs_total(mid, Em) for mi, mid in mats.items()}
        ref = max(sig, key=lambda mi: sig[mi].max() / max(sig[mi].min(), 1e-30))
        lw = np.log(np.maximum(sig[ref], 1e-30))
        win = np.minimum((n_windows * (u - u[0]) / max(u[-1] - u[0], 1e-30)).astype(int), n_windows - 1)
        band = np.empty(Em.size, dtype=np.int64)
        for k in range(n_windows):
            sel = np.flatnonzero(win == k)
            if sel.size == 0:
                continue
            if spacing == 'log':
                # levels uniform in log Sigma_t: resonance peaks get their own bands
                qs = np.linspace(lw[sel].min(), lw[sel].max(), n_levels + 1)[1:-1]
            else:
                qs = np.quantile(lw[sel], np.linspace(0, 1, n_levels + 1)[1:-1])
            band[sel] = k * n_levels + np.searchsorted(qs, lw[sel])
        ids, band = np.unique(band, return_inverse=True)
        L = ids.size
        mass = np.bincount(band, weights=w, minlength=L)
        U = None  # indicator lanes, carried by the bands (resolve_load bands=)
        for mi in mats:
            D = np.bincount(band, weights=w * sig[mi], minlength=L) / mass
            blocks[(mi, g)] = (edges, U, D)
            info[(mi, g)] = dict(n=Em.size, L=L, Dmin=float(D.min()), Dmax=float(D.max()),
                                 band=band, mass=mass, sigma_el=rmc.xs(mats[mi], Em)[1])
        for a in mats:
            for c in mats:
                if a != c:
                    cross[(a, c, g)] = np.eye(L)
        for r in regions[region_mat >= 0]:
            for t, which in enumerate(('s', 'f')):
                h = rmc.field_shape(np.full(Em.size, r), Em, which) * Em
                if h.sum() > 0:
                    sources[(int(r), t, g)] = np.bincount(band, weights=w * h, minlength=L) / np.sqrt(mass)
    return blocks, cross, sources, info


def elastic_band_transfer(rmc, mats, grids, info, group_edges, n_samples=64,
                          temperature='294K', seed=11):
    """Stationary-target elastic transfer of a band's flux into the field's
    groups: T[(m, g)] [L_g, G], the elastic emission into each group per unit
    band flux, for a flux flat per lethargy within the band (the band
    partner's decoded flux). Bins whose whole elastic window stays in group g
    transfer there; the others are resolved with n_samples draws from
    OpenMC's elastic scattering per bin. info from build_subgroup()."""
    import openmc.data
    import openmc.lib
    lib = openmc.data.DataLibrary.from_xml(os.environ['OPENMC_CROSS_SECTIONS'])
    kT = 294.0 * 8.617333262e-5
    group_edges = np.asarray(group_edges, dtype=float)
    G = len(group_edges) - 1
    data = {}
    T = {}
    for mi, mid in mats.items():
        mat = openmc.lib.materials[mid]
        for g, e in grids.items():
            Em = np.sqrt(e[:-1] * e[1:])
            wu = np.diff(np.log(e))
            band, mass = info[(mi, g)]['band'], info[(mi, g)]['mass']
            P = np.zeros((Em.size, G))
            for nuc, N in zip(mat.nuclides, mat.densities):
                if nuc not in data:
                    entry = lib.get_by_material(nuc)
                    data[nuc] = openmc.data.IncidentNeutron.from_hdf5(entry['path']) if entry else None
                if data[nuc] is None:
                    continue
                sel = N * data[nuc][2].xs[temperature](Em)
                A = data[nuc].atomic_weight_ratio
                alpha = ((A - 1) / (A + 1)) ** 2
                near = alpha * Em < e[0]
                P[~near, g] += sel[~near]
                idx = np.flatnonzero(near)
                if idx.size:
                    Ein = np.repeat(Em[idx], n_samples)
                    Eo, _ = rmc.sample_elastic(nuc, kT, Ein, seed=seed + g)
                    go = np.clip(np.searchsorted(group_edges, Eo, side='right') - 1, 0, G - 1)
                    h = np.zeros((idx.size, G))
                    np.add.at(h, (np.repeat(np.arange(idx.size), n_samples), go), 1.0 / n_samples)
                    P[idx] += sel[idx, None] * h
            Tg = np.zeros((mass.size, G))
            np.add.at(Tg, band, wu[:, None] * P)
            T[(mi, g)] = Tg / mass[:, None]
    return T


def band_closure(Phi, offset, T, region_mat, zone, n_zone):
    """The band partner's expected elastic emission by (zone, outgoing group)
    from region-integrated band fluxes Phi [regions, bands] (rmc.band_sweep)
    and the transfer tables of elastic_band_transfer()."""
    G = len(offset) - 1
    C = np.zeros((n_zone, G))
    for r, mi in enumerate(region_mat):
        if mi < 0:
            continue
        for g in range(G):
            C[zone[r]] += Phi[r, offset[g]:offset[g + 1]] @ T[(int(mi), g)]
    return C
