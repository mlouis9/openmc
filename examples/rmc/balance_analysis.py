"""Replica statistics of iteration runs with the balance eigenvalue update
(stage8.py --k-update balance).

    python balance_analysis.py STAGE8_DIR LABEL=FILE[,FILE...] ...

STAGE8_DIR holds reference.json (the analog reference). For each label, the
replicas of its files are pooled. A replica's estimate is sum P / sum L over
its active generations (P the production, L the net loss of each
generation's flux). The error of one replica is the spread of the replica
estimates, and the figure of merit 1 / (spread^2 T) is given relative to the
reference's, with its 68 % interval from the chi-square distribution of the
spread. With few replicas the spread is uncertain, so the error of a replica
is also estimated from batch means of 20 generations pooled over the
replicas (biased low when the correlation between generations is longer).
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats


BATCH = 20


def replicas(files):
    out = []
    for f in files:
        d = json.loads(Path(f).read_text())
        ina = d['args']['inactive']
        for rep in d['replicas']:
            if rep.get('failure'):
                out.append(None)
                continue
            rows = rep['rows'][ina:]
            P = np.array([r['f'] for r in rows])
            L = np.array([r['loss'] for r in rows])
            x = P / L - np.mean(P / L)
            nb = len(P) // BATCH
            kb = np.array([P[i * BATCH:(i + 1) * BATCH].sum() / L[i * BATCH:(i + 1) * BATCH].sum()
                           for i in range(nb)])
            out.append(dict(k=P.sum() / L.sum(), sd_gen=np.std(P / L, ddof=1),
                            batch_dev=(kb - kb.mean()).tolist(),
                            T=sum(r['time'] for r in rows),
                            W=np.mean([r['W_abs'] for r in rows]),
                            acf1=np.dot(x[:-1], x[1:]) / np.dot(x, x)))
    return out


def main():
    ref = json.loads((Path(sys.argv[1]) / 'reference.json').read_text())
    k_ref = np.mean([r['k'] for r in ref['replicas']])
    sd_ref = np.std([r['k'] for r in ref['replicas']], ddof=1)
    T_ref = np.mean([r['time_active'] for r in ref['replicas']])
    fom_ref = 1.0 / (sd_ref**2 * T_ref)
    print(f'reference k {k_ref:.5f}  sd {sd_ref:.5f}  T {T_ref:.0f} s')
    for arg in sys.argv[2:]:
        label, files = arg.split('=')
        reps = replicas(files.split(','))
        ok = [r for r in reps if r is not None]
        k = np.array([r['k'] for r in ok])
        n, sd, T = len(k), k.std(ddof=1), np.mean([r['T'] for r in ok])
        fom = 1.0 / (sd**2 * T) / fom_ref
        dev = np.concatenate([r['batch_dev'] for r in ok])
        nb = len(ok[0]['batch_dev'])
        sd_bm = math.sqrt(dev @ dev / (len(dev) - n) / nb)
        # sd^2 (n - 1) / sigma^2 is chi-square with n - 1 degrees of freedom
        lo, hi = stats.chi2.ppf([0.16, 0.84], n - 1) / (n - 1)
        print(f'{label}: {n} replicas ({len(reps) - n} failed)  k {k.mean():.5f}  '
              f'dk {1e5 * (k.mean() - k_ref):+.0f} +- {1e5 * sd / math.sqrt(n):.0f} pcm  '
              f'spread {sd:.5f}  per generation {np.mean([r["sd_gen"] for r in ok]):.5f}  '
              f'lag-1 {np.mean([r["acf1"] for r in ok]):.2f}  '
              f'W_abs {np.mean([r["W"] for r in ok]):.2f}  T {T:.0f} s  '
              f'FOM/ref {fom:.2f} ({fom * lo:.2f}-{fom * hi:.2f})  '
              f'batch means: {sd_bm:.5f}, FOM/ref {1.0 / (sd_bm**2 * T) / fom_ref:.2f}')


if __name__ == '__main__':
    main()
