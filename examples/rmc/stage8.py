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
        self.netting = (canc.BinnedNetting(self.rep, N_B, exact_low=True)
                        if variant.get('netting') else None)
        self.low_exact = set(variant.get('low_exact', set()))
        self.pool = canc.DestinationPooling(rmc, files) if variant.get('pooling') else None
        self.regional = canc.RegionalCancellation(rmc) if variant.get('regional') else None
        self.tallier = canc.CorrectionTally(rmc, n_points=tally_points,
                                            mg_points=tally_mg_points)
        self.nu = tally_mg_points > 0      # the low + correction tally is nu-fission
        self._bank_tally = np.zeros(rmc.field_n_regions())
        self.info = {}

    def residual(self, t, n, a, b, k, seed):
        rmc = self.rmc
        rec = rmc.residual_sample(t, n, seed=seed, **self.v['opts'][t])
        bb = list(b)
        if t in self.low_exact:
            bb['sf'.index(t)] = 0.0            # the low term enters at netting
        w = rmc.residual_weights(rec, n, a, tuple(bb), k, 0.0, 0.0)
        if t == 'f' and self.pool is not None:
            out, info = self.pool(rec, w, n, b[1], np.random.default_rng(seed))
            self.info['pooling'] = info
            return out
        keep = w != 0.0
        return dict(r=rec['r'][keep], u=rec['u'][keep], E=rec['E'][keep],
                    w=w[keep], tag=rec['tag'][keep].astype(np.int32))

    def propagate(self, bank, n, k, seed):
        rmc = self.rmc
        if len(bank['w']) == 0:
            self._bank_tally = np.zeros(rmc.field_n_regions())
            return self.empty()
        b = rmc.Bank(**bank)
        out, st, aux = rmc.propagate_aux(b, n, k, seed=seed,
                                         children=self.regional is not None,
                                         history=bool(self.v.get('history')))
        arr = out.arrays()
        out.free()
        b.free()
        self._bank_tally = aux['region_nu_fission'] if self.nu else aux['region_fission']
        self.info['flights'] = st
        if self.regional is not None:
            arr, info = self.regional(bank, arr, aux, np.random.default_rng(seed + 1))
            self.info['regional'] = info
        return arr

    def cancel(self, cand, b, seed):
        info = dict(self.info)
        self.info = {}
        if self.netting is None:
            return cand, info or None
        # Low term per bin only for tags whose records carry none
        bl = tuple(b[i] if t in self.low_exact else 0.0 for i, t in enumerate('sf'))
        out, ninfo = self.netting(cand, bl, seed)
        info['netting'] = ninfo
        return out, info

    def tally(self, a, f, seed):
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


def reference(model, mesh, files, workdir, particles, batches, inactive, replicas):
    """CE eigenvalue replicas: k and fission by fuel region, with run times."""
    idx, cells, mbins = fuel_regions(files)
    out = []
    for rep in range(replicas):
        m = openmc.Model(geometry=model.geometry, materials=model.materials,
                         settings=model.settings)
        m.settings.particles, m.settings.batches = particles, batches
        m.settings.inactive = inactive
        m.settings.seed = 101 + rep
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
    ap.add_argument('--N-delta', type=int, default=20_000)
    ap.add_argument('--N-B', type=int, default=20_000)
    ap.add_argument('--tally-points', type=int, default=20_000)
    ap.add_argument('--n-sub', type=int, default=1,
                    help='energy bins per group of the netting representation')
    ap.add_argument('--fine-groups', type=int, nargs='*', default=[],
                    help='groups (ascending index) whose every fine lift bin is '
                         'its own netting energy bin (resonance-resolved netting)')
    ap.add_argument('--amplitudes', choices=['fixed', 'follow'], default='fixed',
                    help='field amplitudes b: fixed (1, 1), or follow a (b = a)')
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
    ap.add_argument('--out', default=None)
    ap.add_argument('--run-name', default=None,
                    help='name of the working subdirectory (default: the variant), so '
                         'that runs of one variant can run at once')
    args = ap.parse_args()

    w, files, k_mg = cs.exact_volume_files(args.work)
    if args.reference:
        res = reference(w.model, w.mesh, files, files['work'] / 'stage8_reference',
                        args.ref_particles, args.ref_batches, args.ref_inactive,
                        args.replicas)
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
        backend = Stage8Backend(files, w.mesh, VARIANTS[args.variant], args.N_B,
                                args.tally_points, args.tally_mg_points,
                                n_sub, tuple(args.coarsen))
        for rep in range(args.replicas):
            cfg = driver.Config(N={'s': args.N, 'f': args.N}, N_delta=args.N_delta,
                                N_B=args.N_B, seed=args.seed_base + rep,
                                damping=args.damping,
                                amplitudes=(driver.FollowAmplitudes()
                                            if args.amplitudes == 'follow'
                                            else driver.FixedAmplitudes()))
            t0 = time.perf_counter()
            rows, failure = [], None

            def cb(state, d, rep=rep, rows=rows):
                rows.append(d)
                print(f"  {args.variant} rep {rep} gen {d['n']:3d}  k {d['k_next']:.5f}  "
                      f"W_abs {d['W_abs']:.4g}  bank {d['size']} -> "
                      f"{d['size_resampled']}  {d['time']:.1f} s", flush=True)
            try:
                driver.run(backend, cfg, k_mg, args.inactive + args.active, callback=cb)
            except driver.EstimatorFailure as e:
                failure = str(e)
                print('  estimator failure:', failure, flush=True)
            results.append(dict(replica=rep, rows=rows, failure=failure,
                                time=time.perf_counter() - t0))
        rmc.field_free()
    idx, _, _ = fuel_regions(files)
    Path(args.out or f'stage8_{args.variant}.json').write_text(json.dumps(
        dict(args=vars(args), k_mg=k_mg, fuel_regions=idx.tolist(), replicas=results),
        default=float))


if __name__ == '__main__':
    main()
