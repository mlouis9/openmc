"""A deterministic toy of the tagged emission-source iteration, used to check
the bookkeeping of the RMC driver (tags, 1/k, normalization, amplitudes)
before it touches transport (gate G6.1, docs/source/methods/rmc.rst,
methods_rmc_iteration).

Four regions (fuel, moderator, fuel, moderator), a transport operator G, and
diagonal scattering S and fission F production, each in a "CE" and a slightly
different "MG" version. The iteration is in expected values: no sampling
noise.

- step(): one application of the operators to tagged sources (Q_s, Q_f) at
  eigenvalue k: the tagged candidate (S psi, F psi / k) and the fission
  production f.
- converge(): the fixed point of the emission-source iteration.
- run_emission(): the field-plus-bank iteration with a field u (the MG
  fixed point, frozen), amplitudes chosen before each generation (Scheme C,
  b = (b_s, b_f)), and the bank dQ of what the field does not carry.
"""
import numpy as np

R = 4
G_ce = np.array([[1.10, 0.35, 0.08, 0.02], [0.35, 1.05, 0.30, 0.06],
                 [0.08, 0.30, 1.00, 0.25], [0.02, 0.06, 0.25, 0.90]])
S_ce = np.diag([0.30, 0.55, 0.30, 0.55])
F_ce = np.diag([0.40, 0.00, 0.33, 0.00])
G_mg = G_ce * np.array([1.00, 1.01, 0.99, 1.02])[:, None]
S_mg = np.diag([0.305, 0.548, 0.30, 0.552])
F_mg = np.diag([0.392, 0.0, 0.328, 0.0])


def step(Qs, Qf, k, S, F, G):
    """Tagged unnormalized candidate (S psi, F psi / k) and fission
    production f = <1, F psi> of the flux psi = G (Q_s + Q_f)."""
    psi = G @ (Qs + Qf)
    return S @ psi, F @ psi / k, (F @ psi).sum()


def converge(S, F, G, n=4000):
    Qs, Qf, k = np.ones(R) / R, np.ones(R) / R, 1.0
    for _ in range(n):
        Ss, Ff, f = step(Qs, Qf, k, S, F, G)
        kn = k * f
        Qs, Qf, k = Ss / f, Ff * k / (kn * f), kn
    return Qs, Qf, k


Qs_ce, Qf_ce, kE_ce = converge(S_ce, F_ce, G_ce)
us, uf, kE_mg = converge(S_mg, F_mg, G_mg)


def run_emission(bs=1.0, bf=1.0, n_gen=300, rule=None):
    """Scheme C with a frozen field shape u = (us, uf). The amplitudes b are
    fixed, or given by rule(a_s, a_f, dQs, dQf, b_prev) before each
    generation.
    Returns one row per generation: k, a_s, a_f, b_s, b_f, |dQ|_1, rho (the
    field's share of fission production)."""
    a_s, a_f, dQs, dQf, k = 1.0, 1.0, np.zeros(R), np.zeros(R), kE_mg
    rows = []
    for n in range(n_gen + 1):
        Qs, Qf = a_s * us + dQs, a_f * uf + dQf
        field_prod = (F_ce @ G_ce @ (a_s * us + a_f * uf)).sum()
        if rule is not None:
            bs, bf = rule(a_s, a_f, dQs, dQf, (bs, bf))
        rows.append(dict(n=n, k=k, a_s=a_s, a_f=a_f, b_s=bs, b_f=bf,
                         l1=np.abs(dQs + dQf).sum(),
                         rho=field_prod / (F_ce @ G_ce @ (Qs + Qf)).sum()))
        Ss, Ff, f = step(Qs, Qf, k, S_ce, F_ce, G_ce)
        kn = k * f
        cs, cf = bs * us, bf * uf
        a_s, a_f = bs / f, bf * k / (kn * f)
        dQs, dQf = (Ss - cs) / f, (Ff - cf) * k / (kn * f)
        k = kn
    return rows


def mass_matching(damping=1.0):
    """Per-tag mass matching: the target b_t = a_t + <1, dQ_t> / <1, u_t>,
    approached from the previous b by the damping factor."""
    def rule(a_s, a_f, dQs, dQf, b_prev):
        ts = a_s + dQs.sum() / us.sum()
        tf = a_f + dQf.sum() / uf.sum()
        return (b_prev[0] + damping * (ts - b_prev[0]),
                b_prev[1] + damping * (tf - b_prev[1]))
    return rule
