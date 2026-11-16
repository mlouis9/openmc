"""Sign cancellation for the RMC bank (Stage 8), and the correction tally.

Theory: docs/source/methods/rmc.rst, methods_rmc_cancellation and
methods_rmc_tallies. Four cancellers act on a signed bank of tag-s and tag-f
particles (numpy dicts r, u, E, w, tag):

- BinnedNetting: approximate. Bins b = (tag, field region R, group g,
  direction bin d); the signed weights in a bin are summed, and the net is
  re-sampled from the lift within the bin (uniform in R, h_{t,m,g} in
  energy, uniform within d). With exact_low, the low term -b_t F is not
  carried by particles: it is added to each bin as its closed-form
  integral F_b = b_t V_R Q_{t,R,g} / D. Biased only by the within-bin shape
  of what is netted.
- DestinationPooling (exact): at each destination of the fission residual
  with every channel on the sampler route, the low term and the sampled
  fission particles share their position, so their energies can be pooled
  into the region's lifted fission spectrum q_R with the identity
  w delta_y = w (1 - q(y)/p(y)) delta_y + [w q(y)/p(y)] delta_{Y'},
  Y' ~ q, which holds in expectation whenever p > 0 where q > 0. The low
  term's target density is known, so it goes to the pool whole.
- RegionalCancellation (exact): fission children of isotropic tag-f
  parents whose flight was shorter than d_max have a closed-form density
  in position (the near point kernel, summed over the reflective images of
  the parent) and energy (chi of the nuclide at the parent's energy), so
  the same identity pools part of each child's weight into one uniform
  distribution over the fuel pin with a reference fission spectrum, where
  opposite signs cancel exactly (Belanger et al.'s regional cancellation).
- No cancellation.

The correction tally (CorrectionTally) estimates generation n's fission
rate by field region as (1/f_n)(a_s X_s + a_f X_f + B), with X_y the
field's CE fission rate from fresh backward traces and B the bank's from a
collision estimator during the flights.
"""
import math

import h5py
import numpy as np

import problems

TAG = {'s': 0, 'f': 1}


# ---------------------------------------------------------------------------
# Representation: regions, groups, direction bins, and the lift
# ---------------------------------------------------------------------------

class Representation:
    """Field regions with their voxel boxes, the group structure, direction
    bins for tag s (n_mu polar x n_phi azimuthal, equal area), and samplers
    of the lift within a group."""

    def __init__(self, rmc, files, mesh, n_mu=2, n_phi=4, coarsen=(3, 3), n_sub=1):
        self.rmc = rmc
        with h5py.File(files['field']) as f:
            self.mesh_bin = f['regions/mesh_bin'][()]
            self.material_id = f['regions/material_id'][()]
        with h5py.File(files['volumes']) as f:
            self.V = f['V_A'][()]
        with h5py.File(files['lifts']) as f:
            self.fine = f['fine_edges'][()]
        self.R = len(self.V)
        self.groups = rmc.field_groups()
        self.G = len(self.groups) - 1
        self.n_mu, self.n_phi = n_mu, n_phi
        self.D = n_mu * n_phi
        ll = np.array(mesh.lower_left)
        ur = np.array(mesh.upper_right)
        dim = np.array(mesh.dimension)
        width = (ur - ll) / dim
        b = self.mesh_bin
        ix, iy, iz = b % dim[0], (b // dim[0]) % dim[1], b // (dim[0] * dim[1])
        self.lo = ll + np.column_stack([ix, iy, iz]) * width
        self.hi = self.lo + width
        self.Q = {t: np.array([rmc.field_emission(r, t) for r in range(self.R)])
                  for t in 'sf'}
        self._cdf = {}
        # Zones: material x a coarse grid of the source-region mesh. A bin is
        # (tag, zone, group, direction bin); within a zone the re-sampled
        # particles follow the field (region with probability V_R Q_{t,R,g}).
        cx = np.minimum(ix * coarsen[0] // dim[0], coarsen[0] - 1)
        cy = np.minimum(iy * coarsen[1] // dim[1], coarsen[1] - 1)
        mats = sorted(set(self.material_id.tolist()))
        mi = np.array([mats.index(m) for m in self.material_id])
        self.zone = (mi * coarsen[1] + cy) * coarsen[0] + cx
        self.Z = len(mats) * coarsen[0] * coarsen[1]
        self.regions_of = [np.flatnonzero(self.zone == z) for z in range(self.Z)]
        # Energy bins: each group split into n_sub contiguous runs of fine
        # bins (n_sub = 1: the groups). n_sub may be one number or one per
        # group (ascending energy); a value at least the group's number of
        # fine bins gives every fine bin of the group its own energy bin.
        mid = 0.5 * (self.fine[:-1] + self.fine[1:])
        g_of = np.clip(np.searchsorted(self.groups, mid, side='right') - 1, 0, self.G - 1)
        self.fine2e = np.empty(len(mid), dtype=np.int64)
        n_sub_g = np.broadcast_to(np.asarray(n_sub, dtype=np.int64), (self.G,))
        e2g = []
        for g in range(self.G):
            idx = np.flatnonzero(g_of == g)
            parts = np.array_split(idx, min(int(n_sub_g[g]), max(len(idx), 1)))
            for part in parts:
                if len(part):
                    self.fine2e[part] = len(e2g)
                    e2g.append(g)
        self.e2g = np.array(e2g)
        self.NE = len(e2g)
        self._hmass = {}

    def n_bins(self):
        return 2 * self.Z * self.NE * self.D

    def _lift_mass(self, region, t):
        """Fine-bin masses of the lift h_{t,m,g} of the region's material."""
        key = (int(region), t)
        if key not in self._hmass:
            mid = 0.5 * (self.fine[:-1] + self.fine[1:])
            h = self.rmc.field_shape(np.full(mid.size, region), mid, t)
            self._hmass[key] = h * np.diff(self.fine)
        return self._hmass[key]

    def _ebin_mass(self, region, t):
        """Mass of the lift in each energy bin (sums to 1 within a group)."""
        m = self._lift_mass(region, t)
        return np.bincount(self.fine2e, weights=m, minlength=self.NE)

    def bins(self, bank):
        """Bin index of every particle (-1 outside the field)."""
        reg = self.rmc.field_locate(bank['r'])
        j = np.clip(np.searchsorted(self.fine, bank['E'], side='right') - 1, 0,
                    len(self.fine2e) - 1)
        g = self.fine2e[j]
        u = bank['u']
        imu = np.clip(((u[:, 2] + 1.0) * 0.5 * self.n_mu).astype(int), 0, self.n_mu - 1)
        phi = np.mod(np.arctan2(u[:, 1], u[:, 0]), 2 * np.pi)
        iph = np.clip((phi / (2 * np.pi) * self.n_phi).astype(int), 0, self.n_phi - 1)
        d = np.where(bank['tag'] == 0, imu * self.n_phi + iph, 0)
        z = self.zone[np.maximum(reg, 0)]
        key = ((bank['tag'] * self.Z + z) * self.NE + g) * self.D + d
        return np.where(reg >= 0, key, -1)

    def low_term(self, b):
        """-b_t F_b per bin: the field's closed-form integral."""
        out = np.zeros(self.n_bins())
        for t in 'sf':
            it = TAG[t]
            Fz = np.zeros((self.Z, self.NE))
            for r in range(self.R):
                F = b[it] * self.V[r] * self.Q[t][r, self.e2g] * self._ebin_mass(r, t)
                Fz[self.zone[r]] += F
            D = self.D if t == 's' else 1
            for d in range(D):
                idx = ((it * self.Z + np.arange(self.Z))[:, None] * self.NE +
                       np.arange(self.NE)[None, :]) * self.D + d
                out[idx.ravel()] -= (Fz / D).ravel()
        return out

    def _lift_cdf(self, region, t):
        key = (int(region), t)
        if key not in self._cdf:
            mass = self._lift_mass(region, t)
            cdfs = []
            for e in range(self.NE):
                m = np.where(self.fine2e == e, mass, 0.0)
                c = np.cumsum(m)
                cdfs.append(c / c[-1] if c[-1] > 0 else None)
            self._cdf[key] = cdfs
        return self._cdf[key]

    def sample(self, keys, rng):
        """One state per bin key: uniform in the region (rejection in its
        voxel), the lift in energy, uniform in the direction bin."""
        n = len(keys)
        d = keys % self.D
        e = (keys // self.D) % self.NE
        g = self.e2g[e]
        zone = (keys // (self.D * self.NE)) % self.Z
        tag = keys // (self.D * self.NE * self.Z)
        # Region within the zone with probability V_R Q_{t,R,g} (one material
        # per zone, so the lift's mass in the energy bin is common)
        reg = np.empty(n, dtype=np.int64)
        for i in range(n):
            rs = self.regions_of[zone[i]]
            p = self.V[rs] * self.Q['sf'[tag[i]]][rs, g[i]]
            if not p.sum() > 0:
                p = self.V[rs]
            c = np.cumsum(p)
            reg[i] = rs[min(np.searchsorted(c, rng.random() * c[-1], side='right'),
                            len(rs) - 1)]
        r = np.empty((n, 3))
        todo = np.arange(n)
        while todo.size:
            lo, hi = self.lo[reg[todo]], self.hi[reg[todo]]
            cand = lo + (hi - lo) * rng.random((todo.size, 3))
            ok = self.rmc.field_locate(cand) == reg[todo]
            r[todo[ok]] = cand[ok]
            todo = todo[~ok]
        E = np.empty(n)
        for i in range(n):
            c = self._lift_cdf(reg[i], 'sf'[tag[i]])[e[i]]
            if c is None:
                js = np.flatnonzero(self.fine2e == e[i])
                lo_e, hi_e = self.fine[js[0]], self.fine[js[-1] + 1]
                E[i] = lo_e * (hi_e / lo_e) ** rng.random()
            else:
                j = min(np.searchsorted(c, rng.random(), side='right'), len(c) - 1)
                E[i] = self.fine[j] + rng.random() * (self.fine[j + 1] - self.fine[j])
        imu, iph = d // self.n_phi, d % self.n_phi
        mu_lo = -1 + 2 * imu / self.n_mu
        mu = np.where(tag == 0, mu_lo + 2 / self.n_mu * rng.random(n),
                      2 * rng.random(n) - 1)
        ph = np.where(tag == 0, 2 * np.pi * (iph + rng.random(n)) / self.n_phi,
                      2 * np.pi * rng.random(n))
        st = np.sqrt(np.maximum(0.0, 1 - mu**2))
        u = np.column_stack([st * np.cos(ph), st * np.sin(ph), mu])
        return dict(r=r, u=u, E=E, tag=tag.astype(np.int32))


# ---------------------------------------------------------------------------
# Approximate: binned netting
# ---------------------------------------------------------------------------

class BinnedNetting:
    """Net per bin and re-sample n_b ~ N |m_b| / sum |m| particles per bin
    (stochastic rounding), each with weight m_b / n_b.

    Where the re-sampled particles are placed within their bin:
    - 'field': the representation's shape (region by the field's emission,
      the lift in energy, uniform in the direction bin);
    - 'empirical': copies of the bin's own particles of the net's sign,
      drawn with probability proportional to |w|. A bin holding one sign is
      then re-sampled without bias; a mixed bin is biased only by the
      difference between the shapes of its positive and negative parts
      (zero when they are proportional), not by the difference between the
      correction's shape and the field's."""

    def __init__(self, rep, n_particles, exact_low=False, placement='field'):
        self.rep, self.n, self.exact_low = rep, n_particles, exact_low
        self.placement = placement
        if placement == 'empirical' and exact_low:
            raise ValueError('empirical placement has no particles for a closed-form low term')

    def __call__(self, bank, b, seed):
        rep = self.rep
        rng = np.random.default_rng(seed)
        key = rep.bins(bank)
        m = np.bincount(key[key >= 0], weights=bank['w'][key >= 0],
                        minlength=rep.n_bins())
        if self.exact_low:
            m = m + rep.low_term(b)
        a = np.abs(m)
        W = a.sum()
        if W == 0.0:
            return {k: v[:0] for k, v in bank.items()}, dict(W_before=0.0, W_after=0.0)
        expect = self.n * a / W
        nb = np.floor(expect + rng.random(len(expect))).astype(np.int64)
        live = np.flatnonzero(nb > 0)
        keys = np.repeat(live, nb[live])
        if self.placement == 'empirical':
            out = self._empirical(bank, key, m, keys, rng)
        else:
            out = rep.sample(keys, rng)
        # Stochastic rounding: E[n_b] = expect_b, so weight m_b / expect_b
        # per particle keeps every bin's net in expectation
        out['w'] = np.repeat(m[live] / expect[live], nb[live])
        info = dict(W_before=float(np.abs(bank['w']).sum()), W_after=float(W),
                    bins=int(len(live)))
        return out, info

    @staticmethod
    def _empirical(bank, key, m, keys, rng):
        """For each output key, a copy of one of the bin's particles whose
        sign is the net's, chosen with probability proportional to |w|."""
        ok = key >= 0
        idx = np.flatnonzero(ok)
        k_ok = key[idx]
        w_ok = bank['w'][idx]
        elig = np.where(np.sign(w_ok) == np.sign(m[k_ok]), np.abs(w_ok), 0.0)
        order = np.argsort(k_ok, kind='stable')
        ks, ws = k_ok[order], elig[order]
        cs = np.cumsum(ws)
        first = np.searchsorted(ks, keys, side='left')
        last = np.searchsorted(ks, keys, side='right')
        base = np.where(first > 0, cs[np.maximum(first - 1, 0)], 0.0)
        base = np.where(first > 0, base, 0.0)
        total = cs[last - 1] - base
        target = base + rng.random(len(keys)) * total
        j = np.searchsorted(cs, target, side='right')
        j = np.clip(j, first, last - 1)
        src = idx[order[j]]
        return dict(r=bank['r'][src], u=bank['u'][src], E=bank['E'][src],
                    tag=bank['tag'][src].astype(np.int32))


# ---------------------------------------------------------------------------
# Exact: pooling at destinations of the fission residual
# ---------------------------------------------------------------------------

class DestinationPooling:
    """Exact pooling of the fission residual at each destination (sampler
    route for every channel, box destinations). Applied to the records of
    one residual_sample('f', ...) call and their weights."""

    def __init__(self, rmc, files):
        self.rmc = rmc
        with h5py.File(files['field']) as f:
            box_ll = f['domain_lower_left'][()]
            box_ur = f['domain_upper_right'][()]
        with h5py.File(files['lifts']) as f:
            self.fine = f['fine_edges'][()]
        self.V_box = float(np.prod(np.asarray(box_ur) - np.asarray(box_ll)))
        n = rmc.field_n_regions()
        self.QfR = np.array([rmc.field_emission(r, 'f').sum() for r in range(n)])
        self._cdf = {}

    def q_R(self, region, E):
        """Lifted fission emission spectrum of the region, normalized."""
        E = np.atleast_1d(E)
        val = 4 * math.pi * self.rmc.field_evaluate(np.full(len(E), region), E, 'f')
        return val / self.QfR[region]

    def _sample_q(self, region, rng):
        if region not in self._cdf:
            mid = 0.5 * (self.fine[:-1] + self.fine[1:])
            c = np.cumsum(self.q_R(region, mid) * np.diff(self.fine))
            self._cdf[region] = c / c[-1]
        c = self._cdf[region]
        j = min(np.searchsorted(c, rng.random(), side='right'), len(c) - 1)
        return self.fine[j] + rng.random() * (self.fine[j + 1] - self.fine[j])

    def __call__(self, rec, w, N, b_f, rng):
        """Returns (bank, info): kept particles and one pooled particle per
        destination with a nonzero pool."""
        rmc = self.rmc
        dens = rec['sampler'] == 0
        sample_of = np.cumsum(np.r_[0, dens[:-1]].astype(np.int64))
        region = rmc.field_locate(rec['r'])
        pooled = np.zeros(int(dens.sum()))
        kept_w = np.zeros(len(w))
        # Low term: its target -b_f Q_f(r, E) / (N rho_x) is known, so its
        # whole mass -b_f Q_f,R V_box / N goes to the destination's pool
        di = np.flatnonzero(dens)
        reg_d = region[di]
        mass = np.where(reg_d >= 0, self.QfR[np.maximum(reg_d, 0)], 0.0)
        pooled += -b_f * mass * self.V_box / N
        # Sampled fission particles: kept w (1 - q/p), pooled w q/p
        si = np.flatnonzero((~dens) & (rec['channel'] == 3))
        for i in si:
            p = rmc.fission_chi(problems.MAT_FUEL, int(rec['nuclide'][i]),
                                float(rec['E_in'][i]), np.array([rec['E'][i]]))[0]
            q = self.q_R(region[i], rec['E'][i])[0]
            if p > 0.0:
                kept_w[i] = w[i] * (1.0 - q / p)
                pooled[sample_of[i]] += w[i] * q / p
            else:
                kept_w[i] = w[i]
        other = (~dens) & (rec['channel'] != 3)
        kept_w[other] = w[other]
        keep = kept_w != 0.0
        out = dict(r=rec['r'][keep], u=rec['u'][keep], E=rec['E'][keep],
                   w=kept_w[keep], tag=np.ones(int(keep.sum()), dtype=np.int32))
        live = np.flatnonzero(pooled != 0.0)
        n = len(live)
        E = np.array([self._sample_q(reg_d[i], rng) for i in live])
        mu = 2 * rng.random(n) - 1
        ph = 2 * np.pi * rng.random(n)
        st = np.sqrt(1 - mu**2)
        pool = dict(r=rec['r'][di][live], E=E, w=pooled[live],
                    u=np.column_stack([st * np.cos(ph), st * np.sin(ph), mu]),
                    tag=np.ones(n, dtype=np.int32))
        out = {k: np.concatenate([out[k], pool[k]]) for k in out}
        info = dict(W_in=float(np.abs(w).sum()), W_out=float(np.abs(out['w']).sum()),
                    pooled=n)
        return out, info


# ---------------------------------------------------------------------------
# Exact: regional cancellation of fission children
# ---------------------------------------------------------------------------

class RegionalCancellation:
    """Fission children of tag-f parents with flight path < d_max, pooled
    over the fuel pin (radius R_FUEL, the box's height) with a reference
    fission spectrum q_E. A child at y = (x, E) has sub-density
    p(y) = c K(x | x_p, E_p) chi_m(E | E_p) with K the near kernel and c a
    constant per parent and nuclide, which cancels in q/p once s is chosen
    as below. Kept weight w (1 - s q(y)/p'(y)), pooled w s q(y)/p'(y)
    with p' = K chi and s = beta V_pin min_pin K min_E chi/q_E (any s > 0 is
    unbiased; this one keeps most kept weights non-negative)."""

    def __init__(self, rmc, d_max=3.0, beta=1.0, E_ref=1.0e6, n_probe=6):
        self.rmc, self.d_max, self.beta = rmc, d_max, beta
        self.R, self.H = problems.R_FUEL, problems.HEIGHT
        self.V_pin = math.pi * self.R**2 * self.H
        self.Eg = np.geomspace(1e3, 1.5e7, 400)       # support of q_E
        qe = rmc.fission_chi(problems.MAT_FUEL, 'U235', E_ref, self.Eg)
        qe[(self.Eg < 1e3) | (self.Eg > 1.5e7)] = 0.0
        self.qE_norm = np.trapezoid(qe, self.Eg)
        self.E_ref = E_ref
        self.n_probe = n_probe
        self._chi_cache = {}

    def q_E(self, E):
        val = self.rmc.fission_chi(problems.MAT_FUEL, 'U235', self.E_ref, np.atleast_1d(E))
        val = np.where((E >= 1e3) & (E <= 1.5e7), val, 0.0)
        return val / self.qE_norm

    def _probes(self, xp):
        """Points of the pin far from the parent (for min K)."""
        ang = math.atan2(xp[1], xp[0]) + math.pi
        pts = []
        for a in (ang, ang + 2.0, ang - 2.0):
            for z in (-self.H / 2 * 0.999, self.H / 2 * 0.999):
                pts.append([self.R * 0.999 * math.cos(a), self.R * 0.999 * math.sin(a), z])
        return np.array(pts[:self.n_probe])

    def __call__(self, parents, children, aux, rng):
        rmc = self.rmc
        par = aux['parent']
        elig = np.flatnonzero((children['tag'] == 1) & (parents['tag'][par] == 1) &
                              (aux['path'] < self.d_max))
        w = children['w'].copy()
        pooled = 0.0
        if elig.size:
            xp = parents['r'][par[elig]]
            Ep = parents['E'][par[elig]]
            x = children['r'][elig]
            Ec = children['E'][elig]
            K = rmc.near_kernel(x, xp, Ep, self.d_max)
            # min of K over the pin, from probe points
            probes = np.concatenate([self._probes(p) for p in xp])
            Kp = rmc.near_kernel(probes, np.repeat(xp, self.n_probe, axis=0),
                                 np.repeat(Ep, self.n_probe), self.d_max)
            Kmin = Kp.reshape(-1, self.n_probe).min(axis=1)
            qE = self.q_E(Ec)
            for j, i in enumerate(elig):
                nuc = int(aux['nuclide'][i])
                chi_c = rmc.fission_chi(problems.MAT_FUEL, nuc, float(Ep[j]),
                                        np.array([Ec[j]]))[0]
                chi_g = rmc.fission_chi(problems.MAT_FUEL, nuc, float(Ep[j]), self.Eg)
                qg = self.q_E(self.Eg)
                m = qg > 1e-3 * qg.max()
                rmin = float(np.min(chi_g[m] / qg[m]))
                s = self.beta * self.V_pin * Kmin[j] * rmin
                p = K[j] * chi_c
                if not (p > 0.0) or s <= 0.0:
                    continue
                ratio = s * (qE[j] / self.V_pin) / p
                pooled += w[i] * ratio
                w[i] = w[i] * (1.0 - ratio)
        keep = w != 0.0
        out = {k: v[keep] for k, v in children.items()}
        out['w'] = w[keep]
        n_pool = 0
        if pooled != 0.0:
            # Represent the pool by particles of about the mean child weight
            mean_w = np.abs(children['w']).mean() if len(children['w']) else 1.0
            n_pool = max(1, int(round(abs(pooled) / mean_w)))
            rr = self.R * np.sqrt(rng.random(n_pool))
            th = 2 * np.pi * rng.random(n_pool)
            z = self.H * (rng.random(n_pool) - 0.5)
            r = np.column_stack([rr * np.cos(th), rr * np.sin(th), z])
            E = self._sample_qE(n_pool, rng)
            mu = 2 * rng.random(n_pool) - 1
            ph = 2 * np.pi * rng.random(n_pool)
            st = np.sqrt(1 - mu**2)
            u = np.column_stack([st * np.cos(ph), st * np.sin(ph), mu])
            pool = dict(r=r, u=u, E=E, w=np.full(n_pool, pooled / n_pool),
                        tag=np.ones(n_pool, dtype=np.int32))
            out = {k: np.concatenate([out[k], pool[k]]) for k in out}
        info = dict(eligible=int(elig.size), W_in=float(np.abs(children['w']).sum()),
                    W_out=float(np.abs(out['w']).sum()), pooled=float(pooled),
                    n_pool=n_pool)
        return out, info

    def _sample_qE(self, n, rng):
        q = self.q_E(self.Eg)
        c = np.concatenate([[0.0], np.cumsum(0.5 * (q[1:] + q[:-1]) * np.diff(self.Eg))])
        c /= c[-1]
        x = rng.random(n)
        j = np.clip(np.searchsorted(c, x, side='right') - 1, 0, len(self.Eg) - 2)
        f = (x - c[j]) / np.maximum(c[j + 1] - c[j], 1e-300)
        return self.Eg[j] + f * (self.Eg[j + 1] - self.Eg[j])


# ---------------------------------------------------------------------------
# Correction tally
# ---------------------------------------------------------------------------

class CorrectionTally:
    """Fission (or nu-fission) rate by field region for each generation:
    (1/f_n) (a_s X_s + a_f X_f + B_n). X from n_points fresh backward points
    per generation (field_tally), B from the flights' collision estimator.

    With mg_points > 0, the field part is a low-order tally plus a
    correction: X = X^MG + (X^CE - X^MG on the same paths), with X^MG, the
    MG nu-fission rate of the MG flux the field produces, estimated once
    from mg_points points (MG and CE scores share each path, so their
    difference has much less variance than either). Its error is then
    frozen, at the level of mg_points."""

    def __init__(self, rmc, n_points=20_000, n_E=4, mg_points=0, seed=987654321):
        self.rmc, self.n_points, self.n_E = rmc, n_points, n_E
        self.mg_points = mg_points
        self.X_mg = None
        if mg_points > 0:
            out = rmc.field_tally(mg_points, 1, seed=seed, mg=True)
            self.X_mg = out[3]

    def field_part(self, a, seed):
        if self.X_mg is None:
            f, nf, sq = self.rmc.field_tally(self.n_points, self.n_E, seed=seed)
            return a[0] * f[:, 0] + a[1] * f[:, 1]
        f, nf, sq, nm, dq = self.rmc.field_tally(self.n_points, self.n_E, seed=seed, mg=True)
        X = self.X_mg + (nf - nm)
        return a[0] * X[:, 0] + a[1] * X[:, 1]
