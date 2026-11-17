"""The RMC iteration with sign cancellation and correction tallies (Stage 8),
and its first end-to-end measurement on the graphite pincell.

    python stage8.py --work rmc_visual_work --variant combined --replicas 3
    python stage8.py --work rmc_visual_work --reference   # CE Monte Carlo reference

The tuned campaign of results/stage8 (combined variants):

    python stage8.py --variant combined --inactive 40 --active 60 \
        --N-delta 200000 --N-B 40000 --tally-points 10000 --tally-mg-points 400000
    python stage8.py --reference --ref-batches 70 --ref-inactive 20 --replicas 4

Variants (VARIANTS): the residual estimator per tag, the cancellers, and the
low term's treatment. Each replica runs --inactive + --active generations of
driver.generation with Stage8Backend; per generation it records k, f, the
amplitudes, W_abs before and after cancellation, and the correction
tally of the fission rate by field region. The reference is OpenMC's
eigenvalue calculation with the same physics, tallying fission by fuel
region (cell x source-region mesh), in independent replicas, with its run
time, so that figures of merit compare like with like.
"""
import argparse
import json
import math
import time
from pathlib import Path

import h5py
import numpy as np

import openmc

import cancellation as canc
import controlled_residual_study as cs
import driver
import pipeline
import problems

TAIL = dict(tail='roulette', cutoff=math.exp(-4))
DENSITY_ROUTE = dict(directions='isotropic', n_directions=8, **TAIL)
DENSITY_FREE_F = dict(directions='isotropic', n_directions=8, all_sampler=True, **TAIL)

# name: residual options per tag, cancellers
VARIANTS = {
    'none': dict(opts={'s': DENSITY_ROUTE, 'f': DENSITY_ROUTE}),
    'binned': dict(opts={'s': DENSITY_ROUTE, 'f': DENSITY_ROUTE}, netting=True),
    'binned_exact_low': dict(opts={'s': DENSITY_ROUTE, 'f': DENSITY_FREE_F},
                             netting=True, low_exact={'f'}),
    'exact': dict(opts={'s': DENSITY_ROUTE, 'f': DENSITY_FREE_F}, pooling=True,
                  regional=True),
    'combined': dict(opts={'s': DENSITY_ROUTE, 'f': DENSITY_FREE_F}, pooling=True,
                     regional=True, netting=True),
    # History-based flights: a generation is fission to fission. The regional
    # kernel needs isotropic tag-f parents of each child, which a history
    # no longer has, so it is left out.
    # Grouped cancellation at fission births (deterministic mixture over
    # emission records), without and with netting
    'grouped': dict(opts={'s': DENSITY_ROUTE, 'f': DENSITY_FREE_F}, pooling=True,
                    grouped=True),
    'grouped_net': dict(opts={'s': DENSITY_ROUTE, 'f': DENSITY_FREE_F}, pooling=True,
                        grouped=True, netting=True),
    # The field's residual in the netting bins from forward rays (CE minus
    # multigroup deposits on the same rays), accumulated over the
    # generations; no residual particles are sampled
    'forward': dict(opts={}, netting=True, forward=True),
    # Binned control variate: the residual records of 'combined' (unbiased,
    # with their own within-bin structure), whose bin totals are replaced by
    # the forward rays' correlated totals; the difference is placed with the
    # field's shape. Unbiased as a measure for any placement shape.
    'bcv': dict(opts={'s': DENSITY_ROUTE, 'f': DENSITY_FREE_F}, pooling=True,
                regional=True, netting=True, forward=True, records=True),
    # The same without the regional cancellation of fission children
    'bcv_nr': dict(opts={'s': DENSITY_ROUTE, 'f': DENSITY_FREE_F}, pooling=True,
                   netting=True, forward=True, records=True),
    'history': dict(opts={'s': DENSITY_ROUTE, 'f': DENSITY_FREE_F}, pooling=True,
                    netting=True, history=True),
}


class Stage8Backend(driver.OpenMCBackend):
    def __init__(self, files, mesh, variant, N_B, tally_points=20_000, tally_mg_points=0,
                 n_sub=1, coarsen=(3, 3)):
        import openmc.lib.rmc as rmc
        super().__init__({})
        self.v = variant
        self.files = files
        self.rep = canc.Representation(rmc, files, mesh, n_sub=n_sub, coarsen=coarsen)
        placement = variant.get('placement', 'field')
        self.netting = (canc.BinnedNetting(self.rep, N_B, exact_low=(placement == 'field'),
                                           placement=placement)
                        if variant.get('netting') else None)
        self.low_exact = set(variant.get('low_exact', set()))
        self.pool = canc.DestinationPooling(rmc, files) if variant.get('pooling') else None
        self.regional = canc.RegionalCancellation(rmc) if variant.get('regional') else None
        self.grouped = (canc.GroupedFissionCancellation(rmc, self.rep)
                        if variant.get('grouped') else None)
        self.tallier = (canc.CorrectionTally(rmc, n_points=tally_points,
                                             mg_points=tally_mg_points)
                        if tally_points > 0 else None)
        self.nu = tally_mg_points > 0      # the low + correction tally is nu-fission
        self.no_tally = tally_points <= 0  # eigenvalue only
        self._bank_tally = np.zeros(rmc.field_n_regions())
        self.info = {}
        self.res_opts = {}   # per-tag overrides of the residual options
        self._X_mg = None    # field_production: the MG part, once
        self._fp_sum, self._fp_n = np.zeros(2), 0
        self._fl_sum, self._fl_n, self.fl_points = np.zeros(2), 0, 5000
        self._flight_stats = None
        # Precomputed (X_s, X_f) and (L_s, L_f) of the frozen field, if given:
        # they replace the per-generation estimates
        self.constants = None
        self.loss_estimator = 'backward'
        # Forward residual in the bin basis: running sums over generations
        self.fwd_rays, self.fwd_energies = 40_000, 8
        self._fwd_d = self._fwd_m = 0.0
        self._fwd_n = self._fwd_mn = 0
        self.fwd_mg_rays = 10_000
        self.n_control = 4000
        self._F = None
        self._P_s = None
        self._M_fixed = None
        self.fwd_n_out, self.fwd_n_out_f = 1, 0
        self.net_every = 1   # netting every net_every generations
        self.fp_points, self.fp_energies, self.k_mg = 10_000, 4, None

    def residual(self, t, n, a, b, k, seed):
        if not self.v.get('forward'):
            return self._records(t, n, a, b, k, seed)
        if t == 's':
            self._forward_step(seed)
            self._abk = (a, b, k)
        if not self.v.get('records'):
            # Biased form: the bin totals are injected at netting with the
            # field's shape (_forward_residual)
            return self.empty()
        out = self._records(t, n, a, b, k, seed)
        if t == 's':
            self._P_s = out
            return out
        # Binned control variate: the records' bin totals are replaced by
        # the forward totals c; the difference has zero expectation in every
        # bin and is placed with the field's shape
        rep = self.rep
        P = self.concat(self._P_s, out)
        key = rep.bins(P)
        ok = key >= 0
        delta = self._forward_control() - np.bincount(key[ok], weights=P['w'][ok],
                                                      minlength=rep.n_bins())
        ctrl, info = self._control_particles(delta, seed)
        self.info['control'] = info
        return self.concat(out, ctrl)

    def _records(self, t, n, a, b, k, seed):
        rmc = self.rmc
        rec = rmc.residual_sample(t, n, seed=seed, **{**self.v['opts'][t],
                                                      **self.res_opts.get(t, {})})
        bb = list(b)
        if t in self.low_exact:
            bb['sf'.index(t)] = 0.0            # the low term enters at netting
        w = rmc.residual_weights(rec, n, a, tuple(bb), k, 0.0, 0.0)
        if t == 'f' and self.pool is not None:
            out, info = self.pool(rec, w, n, b[1], np.random.default_rng(seed))
            self.info['pooling'] = info
            return self._rec(out)
        keep = w != 0.0
        return self._rec(dict(r=rec['r'][keep], u=rec['u'][keep], E=rec['E'][keep],
                              w=w[keep], tag=rec['tag'][keep].astype(np.int32)))

    def _forward_step(self, seed):
        """One generation's forward rays: the CE-minus-multigroup deposits
        (fwd_rays rays of fwd_energies energies) and the multigroup-only
        deposits (fwd_mg_rays rays), each added to its running sum. Both are
        constants of the frozen field."""
        rmc = self.rmc
        tb = self.rep.forward_tables()
        d, _ = rmc.forward_residual_bins(self.fwd_rays, tb, n_E=self.fwd_energies,
                                         n_out=self.fwd_n_out, n_out_f=self.fwd_n_out_f,
                                         seed=seed)
        self._fwd_d = self._fwd_d + d
        self._fwd_n += 1
        if self.fwd_mg_rays > 0 and self._M_fixed is None:
            _, m = rmc.forward_residual_bins(self.fwd_mg_rays, tb, seed=seed + 7,
                                             mg_only=True)
            self._fwd_m = self._fwd_m + m
            self._fwd_mn += 1

    def _forward_control(self):
        """Bin totals of the residual for the current amplitudes, from the
        running means D (CE minus multigroup deposits) and M (multigroup
        deposits) per source tag; F_t is the field's own emission by bin:
          tag s: a . (D + M) - b_s F_s,    tag f: a . (D + M) / k - b_f F_f.
        Without multigroup-only rays, M is replaced by the multigroup
        identities of the field (M_s + M_f = F_s in tag s, k_MG F_f in tag f),
        which hold only as far as the field is converged."""
        rep = self.rep
        if self._F is None:
            self._F = (-rep.low_term((1.0, 0.0)), -rep.low_term((0.0, 1.0)))
        a, b, k = self._abk
        h = rep.n_bins() // 2
        D = self._fwd_d / self._fwd_n
        c = np.empty(rep.n_bins())
        if self._M_fixed is not None or self._fwd_mn > 0:
            M = self._M_fixed if self._M_fixed is not None else self._fwd_m / self._fwd_mn
            tot = a[0] * (D[0] + M[0]) + a[1] * (D[1] + M[1])
            c[:h] = tot[:h] - b[0] * self._F[0][:h]
            c[h:] = tot[h:] / k - b[1] * self._F[1][h:]
        else:
            # M_f from the identities is not available separately: use the
            # biased form of the earlier 'forward' variant with M_f = 0
            tot = a[0] * D[0] + a[1] * D[1]
            c[:h] = tot[:h] + (a[0] - b[0]) * self._F[0][:h]
            c[h:] = tot[h:] / k + (a[0] * self.k_mg / k - b[1]) * self._F[1][h:]
        return c

    def _control_particles(self, delta, seed):
        out = canc.control_particles(self.rep, delta, self.n_control,
                                     np.random.default_rng(seed + 11))
        if out is None:
            return self.empty(), dict(W=0.0)
        return self._rec(out), dict(W=float(np.abs(delta).sum()), n=int(len(out['w'])))

    def _rec(self, bank):
        """Fresh residual particles are directional (no emission record)."""
        return canc.with_records(bank) if self.grouped is not None else bank

    def empty(self):
        return self._rec(driver.OpenMCBackend.empty())

    def propagate(self, bank, n, k, seed):
        rmc = self.rmc
        self._flight_stats = None
        if len(bank['w']) == 0:
            self._bank_tally = np.zeros(rmc.field_n_regions())
            return self.empty()
        core = {key: bank[key] for key in ('r', 'u', 'E', 'w', 'tag')}
        b = rmc.Bank(**core)
        if self.grouped is not None:
            gid, goff, gmem = self.grouped.groups(bank, np.random.default_rng(seed + 2))
        out, st, aux = rmc.propagate_aux(b, n, k, seed=seed,
                                         children=(self.regional is not None or
                                                   self.grouped is not None),
                                         history=bool(self.v.get('history')))
        arr = out.arrays()
        out.free()
        b.free()
        if self.grouped is not None:
            arr, ginfo = self.grouped(bank, arr, aux, gid, goff, gmem, n, k)
            self.info['grouped'] = ginfo
            # The children's own records: elastic scattering keeps its
            # incident state, a fission birth is isotropic at its energy
            arr = canc.child_records(arr, aux)
        self._bank_tally = aux['region_nu_fission'] if self.nu else aux['region_fission']
        self.info['flights'] = st
        self._flight_stats = st
        if self.regional is not None:
            arr, info = self.regional(bank, arr, aux, np.random.default_rng(seed + 1))
            self.info['regional'] = info
        return arr

    def cancel(self, cand, b, seed):
        info = dict(self.info)
        self.info = {}
        self._n_cancel = getattr(self, '_n_cancel', -1) + 1
        if self.netting is None or self._n_cancel % self.net_every != 0:
            return cand, info or None
        # Low term per bin only for tags whose records carry none
        bl = tuple(b[i] if t in self.low_exact else 0.0 for i, t in enumerate('sf'))
        out, ninfo = self.netting(cand, bl, seed, extra=self._forward_residual())
        info['netting'] = ninfo
        if self.grouped is not None and 'rk' not in out:
            out = canc.with_records(out)       # field placement: no records
        return out, info

    def _forward_residual(self):
        """The biased form (variant 'forward'): the forward bin totals are
        added at netting and placed with the field's shape."""
        if not self.v.get('forward') or self.v.get('records') or self._fwd_n == 0:
            return None
        return self._forward_control()

    def field_production(self, a, seed):
        """a_s X_s + a_f X_f: the CE nu-fission production of the field's
        first-flight flux per source tag."""
        rmc = self.rmc
        # X_s and X_f are constants of the frozen field, so every
        # generation's estimate is averaged with the earlier ones: the
        # estimate's noise falls with the number of generations instead of
        # entering each k_n afresh. Points are drawn from the field's own
        # lifted fission rate (rmc.field_production).
        if self.constants is not None:
            X = self.constants['X']
            return float(a[0] * X[0] + a[1] * X[1])
        X, _ = rmc.field_production(self.fp_points, self.fp_energies, seed=seed,
                                    cutoff=math.exp(-3))
        self._fp_sum = self._fp_sum + X
        self._fp_n += 1
        X = self._fp_sum / self._fp_n
        return float(a[0] * X[0] + a[1] * X[1])

    def field_loss(self, a, seed):
        """a_s L_s + a_f L_f: the net loss rate (absorption less the gain of
        multiplying scatterings) of the field's first-flight flux per source
        tag, a running mean like field_production."""
        if self.constants is not None:
            L = self.constants['L']
            return float(a[0] * L[0] + a[1] * L[1])
        if self.loss_estimator == 'forward':
            # Forward rays integrated analytically (rmc.field_first_flight_rates)
            _, L, _, _ = self.rmc.field_first_flight_rates(self.fl_points, 1, seed=seed)
        else:
            L, _ = self.rmc.field_production(self.fl_points, self.fp_energies, seed=seed,
                                             cutoff=math.exp(-3), score='loss')
        self._fl_sum = self._fl_sum + L
        self._fl_n += 1
        L = self._fl_sum / self._fl_n
        return float(a[0] * L[0] + a[1] * L[1])

    def bank_balance(self):
        """(production, net loss) of the last flights' flux: collision
        estimates of nu-fission and absorption, less the analog gain of
        multiplying scatterings."""
        st = self._flight_stats
        if st is None:
            return 0.0, 0.0
        return st['nu_fission'], st['absorption'] - st['multiplied']

    def tally(self, a, f, seed):
        if self.no_tally:
            return None
        X = self.tallier.field_part(a, seed)
        return ((X + self._bank_tally) / f).tolist()


# ---------------------------------------------------------------------------
# Reference
# ---------------------------------------------------------------------------

def fuel_regions(files):
    with h5py.File(files['field']) as f:
        mat = f['regions/material_id'][()]
        cell = f['regions/cell_id'][()]
        mb = f['regions/mesh_bin'][()]
    idx = np.flatnonzero(mat == problems.MAT_FUEL)
    return idx, cell[idx], mb[idx]


def reference(model, mesh, files, workdir, particles, batches, inactive, replicas,
              seed_base=101):
    """CE eigenvalue replicas: k and fission by fuel region, with run times."""
    idx, cells, mbins = fuel_regions(files)
    out = []
    for rep in range(replicas):
        m = openmc.Model(geometry=model.geometry, materials=model.materials,
                         settings=model.settings)
        m.settings.particles, m.settings.batches = particles, batches
        m.settings.inactive = inactive
        m.settings.seed = seed_base + rep
        t = openmc.Tally(name='fuel_fission')
        t.filters = [openmc.CellFilter([problems.CELL_FUEL]), openmc.MeshFilter(mesh)]
        t.scores = ['fission', 'nu-fission']
        m.tallies = openmc.Tallies([t])
        t0 = time.perf_counter()
        sp_path = m.run(cwd=Path(workdir) / f'rep{rep}', output=False, threads=1)
        dt = time.perf_counter() - t0
        with openmc.StatePoint(sp_path) as sp:
            k = sp.keff
            tal = sp.get_tally(name='fuel_fission')
            mean = tal.mean.reshape(-1, 2)
            # active time only, from the statepoint's timers
            t_active = float(sp.runtime['active batches'])
        rates = mean[mbins, 0]
        nu_rates = mean[mbins, 1]
        out.append(dict(k=k.n, k_sd=k.s, rates=(rates / rates.sum()).tolist(),
                        nu_rates=(nu_rates / nu_rates.sum()).tolist(),
                        time=dt, time_active=t_active))
        print(f'  reference replica {rep}: k = {k.n:.5f} +/- {k.s:.5f}, '
              f'{dt:.0f} s ({t_active:.0f} s active)', flush=True)
    return dict(regions=idx.tolist(), replicas=out, particles=particles,
                batches=batches, inactive=inactive)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--work', default='rmc_visual_work')
    ap.add_argument('--variant', choices=list(VARIANTS), default='combined')
    ap.add_argument('--replicas', type=int, default=3)
    ap.add_argument('--inactive', type=int, default=5)
    ap.add_argument('--active', type=int, default=30)
    ap.add_argument('--N', type=int, default=2000)
    ap.add_argument('--N-f', type=int, default=None,
                    help='residual samples of tag f (default: --N)')
    ap.add_argument('--N-delta', type=int, default=20_000)
    ap.add_argument('--N-B', type=int, default=20_000)
    ap.add_argument('--tally-points', type=int, default=20_000)
    ap.add_argument('--n-sub', type=int, default=1,
                    help='energy bins per group of the netting representation')
    ap.add_argument('--fine-groups', type=int, nargs='*', default=[],
                    help='groups (ascending index) whose every fine lift bin is '
                         'its own netting energy bin (resonance-resolved netting)')
    ap.add_argument('--res-beta', type=float, default=None,
                    help='log-uniform weight of the outgoing-energy proposal (both tags)')
    ap.add_argument('--res-cut', type=float, default=None,
                    help='roulette cutoff of the backward tails, in optical depth')
    ap.add_argument('--res-ko', type=int, default=None,
                    help='outgoing states per destination of the scattering residual')
    ap.add_argument('--res-ein', type=float, default=None,
                    help='tabulated incident-energy proposal of the scattering sampler '
                         'route, with this log-uniform weight')
    ap.add_argument('--res-ir', type=float, default=None,
                    help='image roulette floor of the scattering residual')
    ap.add_argument('--res-nd', type=int, nargs=2, default=None,
                    help='directions per destination for tags s and f')
    ap.add_argument('--group-size', type=int, default=8,
                    help='records per group of the grouped fission cancellation')
    ap.add_argument('--group-dmax', type=float, default=3.0,
                    help='longest path (cm) of the grouped fission cancellation')
    ap.add_argument('--net-every', type=int, default=1,
                    help='apply netting every this many generations')
    ap.add_argument('--placement', choices=['field', 'empirical'], default='field',
                    help='where netting places re-sampled particles within a bin')
    ap.add_argument('--amplitudes', choices=['fixed', 'follow'], default='fixed',
                    help='field amplitudes b: fixed (1, 1), or follow a (b = a)')
    ap.add_argument('--field-production', type=int, default=0,
                    help='backward points per generation of the direct estimate of the '
                         "field's fission production (0: through the fission residual)")
    ap.add_argument('--k-update', choices=['production', 'balance'], default='production',
                    help='eigenvalue update (driver.Config.k_update)')
    ap.add_argument('--balance-window', type=int, default=0,
                    help='generations in the balance sums (0: all)')
    ap.add_argument('--field-loss', type=int, default=5000,
                    help="backward points per generation of the field's net loss rate "
                         '(balance update)')
    ap.add_argument('--forward-rays', type=int, default=40_000,
                    help='rays per generation of the forward residual (variant forward)')
    ap.add_argument('--forward-energies', type=int, default=8,
                    help='energies per ray of the forward residual')
    ap.add_argument('--forward-n-out', type=int, nargs=2, default=[1, 0],
                    help='forced scatterings and fissions per material and source energy '
                         '(0 fissions: as many as scatterings)')
    ap.add_argument('--forward-mg-rays', type=int, default=10_000,
                    help='multigroup-only rays per generation (0: the multigroup '
                         'identities of the field, biased by its convergence)')
    ap.add_argument('--mg-flux', nargs='+', default=None,
                    help='files of multigroup_deposits.py (the uncollided multigroup '
                         'fluxes of the field from random ray sweeps); replica r uses '
                         'file r modulo their number, for M instead of multigroup-only '
                         'rays, and its time is added to the replica time')
    ap.add_argument('--n-control', type=int, default=4000,
                    help='particles per generation carrying the control term (bcv)')
    ap.add_argument('--loss-estimator', choices=['backward', 'forward'], default='backward',
                    help="estimator of the field's net loss: backward points "
                         '(rmc.field_production) or forward rays integrated analytically '
                         '(rmc.field_first_flight_rates; --field-loss is then the number '
                         'of rays)')
    ap.add_argument('--field-constants', default=None,
                    help="JSON file with the field's production X and net loss L per "
                         'source tag (field_constants.py); replaces the per-generation '
                         'estimates')
    ap.add_argument('--collision-production', action='store_true',
                    help="the bank's fission production from the flights' collision "
                         'estimator')
    ap.add_argument('--damping', type=float, default=1.0,
                    help='normalization damping (driver.Config.damping)')
    ap.add_argument('--seed-base', type=int, default=2000,
                    help='replica r uses seed seed-base + r')
    ap.add_argument('--coarsen', type=int, nargs=2, default=[3, 3],
                    help='zones per axis of the netting representation')
    ap.add_argument('--tally-mg-points', type=int, default=0,
                    help='points of the one-off MG low tally (0: CE field tally only)')
    ap.add_argument('--reference', action='store_true')
    ap.add_argument('--ref-particles', type=int, default=20_000)
    ap.add_argument('--ref-batches', type=int, default=120)
    ap.add_argument('--ref-inactive', type=int, default=20)
    ap.add_argument('--ref-seed-base', type=int, default=101,
                    help='reference replica r uses seed ref-seed-base + r')
    ap.add_argument('--out', default=None)
    ap.add_argument('--field-groups', default=None,
                    help='field of field_variants.field_files with this group structure '
                         '(default: the 8-group field of the controlled study)')
    ap.add_argument('--mgxs-particles', type=int, default=20_000)
    ap.add_argument('--fine-lifts', type=int, nargs='+', default=None,
                    help='fine lift bins per group (fine_lifts.py)')
    ap.add_argument('--zone-lifts', type=int, nargs=2, default=None,
                    help='lifts per zone (fuel rings, graphite rings) from zone_lifts.py')
    ap.add_argument('--field-mesh', type=int, default=12,
                    help='source-region mesh per side of the field variant')
    ap.add_argument('--run-name', default=None,
                    help='name of the working subdirectory (default: the variant), so '
                         'that runs of one variant can run at once')
    args = ap.parse_args()

    if args.field_groups:
        import field_variants as fv
        w, files, k_mg = fv.field_files(args.work, args.field_groups, args.mgxs_particles,
                                        n_per_side=args.field_mesh)
    else:
        w, files, k_mg = cs.exact_volume_files(args.work)
    if args.fine_lifts:
        import fine_lifts as fl
        base = (w, files, k_mg) if args.field_groups else None
        w, files, k_mg = fl.fine_lift_files(args.work, args.fine_lifts, base=base)
    if args.zone_lifts:
        import zone_lifts as zl
        w, files, k_mg = zl.zone_lift_files(args.work, tuple(args.zone_lifts))
    if args.reference:
        res = reference(w.model, w.mesh, files, files['work'] / f"stage8_reference{'_' + args.run_name if args.run_name else ''}",
                        args.ref_particles, args.ref_batches, args.ref_inactive,
                        args.replicas, seed_base=args.ref_seed_base)
        Path(args.out or 'stage8_reference.json').write_text(json.dumps(res, indent=1))
        return

    import openmc.lib.rmc as rmc
    results = []
    with pipeline.ce_session(w.model, w.mesh, files['work'] / f'stage8_{args.run_name or args.variant}'):
        rmc.field_load(files['field'], files['lifts'], files['volumes'])
        n_sub = args.n_sub
        if args.fine_groups:
            n_sub = np.full(len(rmc.field_groups()) - 1, args.n_sub)
            n_sub[args.fine_groups] = 1 << 30
        backend = Stage8Backend(files, w.mesh, {**VARIANTS[args.variant],
                                                'placement': args.placement}, args.N_B,
                                args.tally_points, args.tally_mg_points,
                                n_sub, tuple(args.coarsen))
        backend.k_mg = k_mg
        backend.net_every = args.net_every
        if backend.grouped is not None:
            backend.grouped.g, backend.grouped.d_max = args.group_size, args.group_dmax
        backend.fp_points = max(args.field_production, 1)
        backend.fl_points = args.field_loss
        backend.loss_estimator = args.loss_estimator
        backend.fwd_rays, backend.fwd_energies = args.forward_rays, args.forward_energies
        backend.fwd_mg_rays, backend.n_control = args.forward_mg_rays, args.n_control
        backend.fwd_n_out, backend.fwd_n_out_f = args.forward_n_out
        if args.field_constants:
            c = json.loads(Path(args.field_constants).read_text())
            backend.constants = dict(X=np.array(c['X']), L=np.array(c['L']))
        ein_table = None
        if args.res_ein is not None:
            import residual_efficiency as reff
            ein_table = reff.incident_table(rmc, files)
        for i, t in enumerate('sf'):
            o = {}
            if t == 's' and ein_table is not None:
                o['Ein_table'] = ein_table
                o['beta_in'] = args.res_ein
            if args.res_beta is not None:
                o['beta'] = args.res_beta
            if args.res_cut is not None:
                o['cutoff'] = math.exp(-args.res_cut)
            if args.res_nd is not None:
                o['n_directions'] = args.res_nd[i]
            if t == 's' and args.res_ko is not None:
                o['n_outgoing'] = args.res_ko
            if t == 's' and args.res_ir is not None:
                o['image_roulette'] = args.res_ir
            backend.res_opts[t] = o
        for rep in range(args.replicas):
            backend._fp_sum, backend._fp_n = np.zeros(2), 0
            backend._fl_sum, backend._fl_n = np.zeros(2), 0
            backend._fwd_d = backend._fwd_m = 0.0
            backend._fwd_n = backend._fwd_mn = 0
            backend._M_fixed, time_mg = None, 0.0
            if args.mg_flux:
                mf = np.load(args.mg_flux[rep % len(args.mg_flux)])
                backend._M_fixed = backend.rep.multigroup_bins(files['field'], mf['phi'])
                time_mg = float(mf['time'])
            cfg = driver.Config(N={'s': args.N, 'f': args.N_f or args.N}, N_delta=args.N_delta,
                                N_B=args.N_B, seed=args.seed_base + rep,
                                damping=args.damping, k_update=args.k_update,
                                balance_window=args.balance_window,
                                collision_production=args.collision_production,
                                field_production=args.field_production > 0,
                                amplitudes=(driver.FollowAmplitudes()
                                            if args.amplitudes == 'follow'
                                            else driver.FixedAmplitudes()))
            t0 = time.perf_counter()
            rows, failure = [], None

            def cb(state, d, rep=rep, rows=rows):
                rows.append(d)
                print(f"  {args.variant} rep {rep} gen {d['n']:3d}  k {d['k_next']:.5f}  "
                      + (f"k_bal {d['k_balance']:.5f}  " if d['k_balance'] else '') +
                      f"W_abs {d['W_abs']:.4g}  bank {d['size']} -> "
                      f"{d['size_resampled']}  {d['time']:.1f} s", flush=True)
            try:
                driver.run(backend, cfg, k_mg, args.inactive + args.active, callback=cb)
            except driver.EstimatorFailure as e:
                failure = str(e)
                print('  estimator failure:', failure, flush=True)
            results.append(dict(replica=rep, rows=rows, failure=failure, time_mg=time_mg,
                                time=time.perf_counter() - t0))
        rmc.field_free()
    idx, _, _ = fuel_regions(files)
    Path(args.out or f'stage8_{args.variant}.json').write_text(json.dumps(
        dict(args=vars(args), k_mg=k_mg, fuel_regions=idx.tolist(), replicas=results),
        default=float))


if __name__ == '__main__':
    main()
