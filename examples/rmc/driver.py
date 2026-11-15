"""The residual Monte Carlo iteration (Stage 6), with a frozen field shape.

Each generation (docs/source/methods/rmc.rst, methods_rmc_iteration):

1. Amplitudes b of the field candidate F = b Q^MG are chosen from earlier
   generations only.
2. The candidate bank is the residual of the field (records weighted with
   the current a, b, k) plus the children of the current bank's forward
   flights.
3. f = k (b_f <1, Q^MG_f> + <1, candidate_f>), k_next = k g with
   g = f**damping (g = f by default).
4. Tag s is scaled by 1/g and tag f by k / (k_next g); the field amplitudes
   become a_next = (b_s / g, b_f k / (k_next g)).
5. The bank is compressed to at most N_B sites by unbiased resampling.

The iteration only sees a backend, with the operations below, so the same
code runs on the deterministic toy (ToyBackend, gate G6.1) and on OpenMC
(OpenMCBackend):

- field_mass(t): <1, Q^MG_t> of the frozen field at unit amplitude;
- residual(t, n, a, b, k, seed): a bank of the field's residual of tag t;
- propagate(bank, n, k, seed): the children of a bank's forward flights;
- empty(), concat(x, y), tag_sum(bank, t), abs_weight(bank), size(bank),
  scale(bank, s_s, s_f), resample(bank, n, seed).
"""
from dataclasses import dataclass, field
import math
import time
import zlib

import numpy as np


class EstimatorFailure(RuntimeError):
    """The estimated fission production f was not positive. It is never
    clipped: the run stops so that the cause can be diagnosed."""


# ---------------------------------------------------------------------------
# Amplitude rules: chosen before the generation's samples
# ---------------------------------------------------------------------------

class FixedAmplitudes:
    """b = (b_s, b_f), the default (1, 1) of Scheme C."""

    def __init__(self, b=(1.0, 1.0)):
        self.b = tuple(b)

    def __call__(self, state, backend):
        return self.b


class FollowAmplitudes:
    """b = a: the field emits what its normalized flux produces, so its
    residual a K phi^MG - a Q^MG carries no part proportional to a - b. With
    fixed b, a fluctuation of f moves a away from b and adds (a - b) times
    the field's production, which for scattering is many times f, to the
    residual of the next generation."""

    def __call__(self, state, backend):
        return tuple(state.a)


class MassMatching:
    """Per-tag mass matching: the field carries each tag's current mass,
    b_t = a_t + <1, dQ_t> / <1, Q^MG_t>, from the bank's exact signed tag
    sums, approached from the previous b by the damping factor."""

    def __init__(self, damping=1.0):
        self.damping = damping

    def __call__(self, state, backend):
        out = []
        for i, t in enumerate('sf'):
            target = state.a[i] + backend.tag_sum(state.bank, t) / backend.field_mass(t)
            out.append(state.b[i] + self.damping * (target - state.b[i]))
        return tuple(out)


# ---------------------------------------------------------------------------
# The iteration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    N: dict = field(default_factory=lambda: {'s': 1000, 'f': 1000})
    N_delta: int = 1000          # flights per generation
    N_B: int = 10_000            # bank size after resampling
    amplitudes: object = field(default_factory=FixedAmplitudes)
    seed: int = 1
    # Normalization damping lam in (0, 1]: the bank and field are normalized
    # by g = f**lam instead of f, and k_{n+1} = k_n g. lam = 1 is the plain
    # iteration; smaller lam lets a noisy f move the normalization less, at
    # the price of a slower approach of the fundamental mode. The generation
    # estimate k_n f / P_n (P_n the production of the incoming state) is
    # recorded as k_next either way.
    damping: float = 1.0


@dataclass
class State:
    k: float
    a: tuple
    b: tuple
    bank: object
    n: int = 0
    P: float = 1.0      # production of this state relative to unit (damping)


def seed_of(base, n, label):
    """A seed per generation and purpose, independent of the others."""
    return (base * 1_000_003 + n * 7919 + zlib.crc32(label.encode())) % (2**63)


def initial_state(backend, k):
    """The converged MG field at unit amplitudes, an empty bank."""
    return State(k=k, a=(1.0, 1.0), b=(1.0, 1.0), bank=backend.empty())


def generation(state, backend, cfg):
    t0 = time.perf_counter()
    n, k = state.n, state.k
    b = cfg.amplitudes(state, backend)

    # Residual of the field, then the bank's children
    cand = backend.empty()
    for t in 'sf':
        r = backend.residual(t, cfg.N[t], state.a, b, k, seed_of(cfg.seed, n, 'res_' + t))
        cand = backend.concat(cand, r)
    residual_f = backend.tag_sum(cand, 'f')
    t1 = time.perf_counter()
    children = backend.propagate(state.bank, cfg.N_delta, k, seed_of(cfg.seed, n, 'flight'))
    cand = backend.concat(cand, children)
    t2 = time.perf_counter()

    # Sign cancellation (Stage 8), if the backend has one: it acts on the
    # unnormalized candidate, before f is formed from its tag sums
    cancel_info = None
    if hasattr(backend, 'cancel'):
        cand, cancel_info = backend.cancel(cand, b, seed_of(cfg.seed, n, 'cancel'))
    t3 = time.perf_counter()

    # Fission production, eigenvalue, normalization (never clipped)
    field_f = b[1] * backend.field_mass('f')
    f = k * (field_f + backend.tag_sum(cand, 'f'))
    if not f > 0.0:
        raise EstimatorFailure(f'generation {n}: fission production f = {f}')
    g = f ** cfg.damping
    k_next = k * g
    s_s, s_f = 1.0 / g, k / (k_next * g)
    cand = backend.scale(cand, s_s, s_f)
    a_next = (b[0] * s_s, b[1] * s_f)

    # Generation estimate: production after one generation over production
    # before it (state.P = 1 without damping)
    diag = dict(n=n, k=k, k_next=k * f / state.P, f=f, a_s=state.a[0], a_f=state.a[1],
                b_s=b[0], b_f=b[1],
                rho=k * (field_f + residual_f) / f,
                W_abs=backend.abs_weight(cand), sum_s=backend.tag_sum(cand, 's'),
                sum_f=backend.tag_sum(cand, 'f'), size=backend.size(cand),
                time_residual=t1 - t0, time_flights=t2 - t1, time_cancel=t3 - t2,
                cancel=cancel_info)
    # Correction tally of this generation's flux (field part plus the
    # flights of the incoming bank), if the backend has one
    if hasattr(backend, 'tally'):
        diag['tally'] = backend.tally(state.a, f, seed_of(cfg.seed, n, 'tally'))
    cand = backend.resample(cand, cfg.N_B, seed_of(cfg.seed, n, 'resample'))
    diag['size_resampled'] = backend.size(cand)
    diag['time'] = time.perf_counter() - t0
    return State(k=k_next, a=a_next, b=b, bank=cand, n=n + 1, P=f / g), diag


def run(backend, cfg, k0, n_generations, state=None, callback=None):
    """n_generations from the initial state (or a given one). Returns the
    final state and the per-generation diagnostics."""
    state = initial_state(backend, k0) if state is None else state
    rows = []
    for _ in range(n_generations):
        state, d = generation(state, backend, cfg)
        rows.append(d)
        if callback is not None:
            callback(state, d)
    return state, rows


# ---------------------------------------------------------------------------
# The deterministic toy backend (gate G6.1)
# ---------------------------------------------------------------------------

class ToyBackend:
    """Expected values on the four-region toy (normalization_toy.py): a bank
    is an exact pair of signed vectors (tag s, tag f); the residual and the
    flights apply the CE operators; resampling is the identity, which is its
    expectation."""

    def __init__(self):
        import normalization_toy as toy
        self.toy = toy

    def field_mass(self, t):
        return float((self.toy.us if t == 's' else self.toy.uf).sum())

    def empty(self):
        return {'s': np.zeros(self.toy.R), 'f': np.zeros(self.toy.R)}

    def concat(self, x, y):
        return {t: x[t] + y[t] for t in 'sf'}

    def tag_sum(self, bank, t):
        return float(bank[t].sum())

    def abs_weight(self, bank):
        return float(np.abs(bank['s'] + bank['f']).sum())

    def size(self, bank):
        return self.toy.R

    def scale(self, bank, s_s, s_f):
        return {'s': s_s * bank['s'], 'f': s_f * bank['f']}

    def resample(self, bank, n, seed):
        return bank

    def _produce(self, Qs, Qf, k):
        t = self.toy
        Ss, Ff, _ = t.step(Qs, Qf, k, t.S_ce, t.F_ce, t.G_ce)
        return Ss, Ff

    def residual(self, t, n, a, b, k, seed):
        toy = self.toy
        Ss, Ff = self._produce(a[0] * toy.us, a[1] * toy.uf, k)
        out = self.empty()
        if t == 's':
            out['s'] = Ss - b[0] * toy.us
        else:
            out['f'] = Ff - b[1] * toy.uf
        return out

    def propagate(self, bank, n, k, seed):
        Ss, Ff = self._produce(bank['s'], bank['f'], k)
        return {'s': Ss, 'f': Ff}


# ---------------------------------------------------------------------------
# The OpenMC backend
# ---------------------------------------------------------------------------

class OpenMCBackend:
    """Residual records and forward flights in an openmc.lib session with
    the frozen field loaded (with volumes, so that its masses are defined).
    Banks are dicts of numpy arrays (r, u, E, w, tag) between generations;
    flights copy them into a C++ bank."""

    def __init__(self, residual_options=None, c=0.0, lam=0.0):
        import openmc.lib.rmc as rmc
        self.rmc = rmc
        self.opts = dict(residual_options or {})
        self.c, self.lam = c, lam
        self._mass = {t: rmc.field_mass(t) for t in 'sf'}
        if not all(math.isfinite(m) for m in self._mass.values()):
            raise ValueError('Load the field with a volume file: its masses are needed.')

    def field_mass(self, t):
        return self._mass[t]

    @staticmethod
    def empty():
        return dict(r=np.zeros((0, 3)), u=np.zeros((0, 3)), E=np.zeros(0),
                    w=np.zeros(0), tag=np.zeros(0, dtype=np.int32))

    @staticmethod
    def concat(x, y):
        return {key: np.concatenate([x[key], y[key]]) for key in x}

    @staticmethod
    def tag_sum(bank, t):
        return float(bank['w'][bank['tag'] == {'s': 0, 'f': 1}[t]].sum())

    @staticmethod
    def abs_weight(bank):
        return float(np.abs(bank['w']).sum())

    @staticmethod
    def size(bank):
        return len(bank['w'])

    @staticmethod
    def scale(bank, s_s, s_f):
        out = dict(bank)
        out['w'] = bank['w'] * np.where(bank['tag'] == 1, s_f, s_s)
        return out

    @staticmethod
    def resample(bank, n, seed):
        """Unbiased compression to n sites (systematic selection by |w|,
        each kept site with weight sgn(w) W_abs / n). A bank of at most n
        sites is kept as it is."""
        m = len(bank['w'])
        if m <= n:
            return bank
        a = np.abs(bank['w'])
        W = a.sum()
        if W == 0.0:
            return OpenMCBackend.empty()
        rng = np.random.default_rng(seed)
        points = (rng.random() + np.arange(n)) * (W / n)
        idx = np.searchsorted(np.cumsum(a), points, side='right')
        idx = np.minimum(idx, m - 1)
        out = {key: bank[key][idx] for key in bank}
        out['w'] = np.sign(bank['w'][idx]) * (W / n)
        return out

    def residual(self, t, n, a, b, k, seed):
        rmc = self.rmc
        rec = rmc.residual_sample(t, n, seed=seed, **self.opts)
        w = rmc.residual_weights(rec, n, a, b, k, self.c, self.lam)
        keep = w != 0.0
        return dict(r=rec['r'][keep], u=rec['u'][keep], E=rec['E'][keep],
                    w=w[keep], tag=rec['tag'][keep].astype(np.int32))

    def propagate(self, bank, n, k, seed):
        rmc = self.rmc
        if len(bank['w']) == 0:
            return self.empty()
        b = rmc.Bank(**bank)
        out, _, _ = rmc.propagate(b, n, k, seed=seed)
        arr = out.arrays()
        out.free()
        b.free()
        return arr
