"""Measurement of the composition-multiplication feedback loop (Sec. 5-7
of the theory note): the forward injection vector g, the reactivity
sensitivity s, their product (the loop gain), the cumulative gain identity,
and the discrete-step stability margin.

Three independent routes to the scalar loop gain gamma_A = M k' are
provided, matching Sec. 7.6:

    (a) measure_forward_injection + measure_return_path: a deliberate,
        paired perturbation experiment (Ensembles IV/V).
    (b) predicted_loop_gain: a free prediction from smoothed finite
        differencing of the ensemble-mean k(tau) curve that every run
        already produces.

Agreement between (a) and (b) is the validation described in Sec. 7.6;
disagreement localizes whether the fault is in the injection-vector
measurement, the reactivity-sensitivity measurement, or the assumption
that a single dominant longitudinal mode governs the feedback.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from .scm_transport import KFactors, override_k

__all__ = [
    "measure_forward_injection", "measure_return_path",
    "predicted_loop_gain", "cumulative_gain",
    "step_stability_margin", "critical_step_size",
]


def _flatten_norm(vec_list: Sequence[np.ndarray]) -> float:
    return float(np.sqrt(sum(np.sum(v.astype(np.float64) ** 2) for v in vec_list)))


def measure_forward_injection(integrator, vec: List[np.ndarray],
                               base_result, h: float, eta: float) -> List[np.ndarray]:
    """Sec. 7.6(a): the forward path g, by central-differencing k in the
    depletion solve only. No transport is repeated: ``base_result`` is a
    single already-completed SCMOperatorResult (e.g. the BOS or midpoint
    result of a step already taken), and this function only rebuilds and
    re-solves the depletion matrix with k perturbed by +/- eta.

    Returns g_hat = d N_{l+1} / d k_l, a list of per-material arrays with
    the same shape as ``vec``.
    """
    from .scm_integrators import _cram_step  # local import to avoid a cycle

    k_plus = override_k(base_result.k_factors, +eta)
    k_minus = override_k(base_result.k_factors, -eta)

    A_plus = integrator._build_matrix(base_result.rates, k_plus,
                                       base_result.fission_yields)
    A_minus = integrator._build_matrix(base_result.rates, k_minus,
                                        base_result.fission_yields)

    n_plus = _cram_step(integrator.cram, A_plus, vec, h)
    n_minus = _cram_step(integrator.cram, A_minus, vec, h)

    return [(np_ - nm) / (2.0 * eta) for np_, nm in zip(n_plus, n_minus)]


def measure_return_path(operator, vec: List[np.ndarray],
                         g_hat: List[np.ndarray], eta_n: float,
                         source_rate=None) -> float:
    """Sec. 7.6(b): the return path s^T g, by correlated-sampling paired
    perturbation along the measured injection direction.

    The caller is responsible for ensuring ``operator`` (or the underlying
    ``openmc.lib`` RNG state) uses an identical seed for the two transport
    solves issued here, so that the fresh Monte Carlo noise cancels between
    them and a small, safely linear ``eta_n`` can be used.
    """
    norm_g = _flatten_norm(g_hat)
    if norm_g == 0.0:
        raise ValueError("measured injection vector g_hat is zero")
    direction = [g / norm_g for g in g_hat]

    vec_plus = [v + eta_n * d for v, d in zip(vec, direction)]
    vec_minus = [v - eta_n * d for v, d in zip(vec, direction)]

    res_plus = operator(vec_plus, source_rate)
    res_minus = operator(vec_minus, source_rate)

    delta_k = res_plus.k_factors.k - res_minus.k_factors.k
    return (delta_k / (2.0 * eta_n)) * norm_g


def predicted_loop_gain(tau: np.ndarray, k_mean: np.ndarray,
                         smooth_window: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Sec. 7.6(c): the free prediction gamma_A(tau) = M(tau) k'(tau),
    obtained by smoothed finite differencing of the ensemble-mean k(tau)
    curve. Requires no additional transport.

    Parameters
    ----------
    tau, k_mean : 1-D arrays
        The exposure grid and the replica-averaged multiplication at each
        point (from :func:`scm_depletion.common.replica_stats` applied to
        an ensemble of trajectories' ``k_at_tau`` values).
    smooth_window : int
        Width (in grid points) of the moving-average smoothing applied to
        k_mean before differencing. Set to 1 to disable smoothing.

    Returns
    -------
    (tau, gamma_A) : the exposure grid and the predicted loop gain there.
    """
    tau = np.asarray(tau, dtype=np.float64)
    k_mean = np.asarray(k_mean, dtype=np.float64)
    if smooth_window > 1:
        kernel = np.ones(smooth_window) / smooth_window
        k_smooth = np.convolve(k_mean, kernel, mode="same")
    else:
        k_smooth = k_mean
    k_prime = np.gradient(k_smooth, tau)
    M = 1.0 / (1.0 - k_smooth)
    gamma_A = M * k_prime
    return tau, gamma_A


def cumulative_gain(M_start: float, M_end: float) -> float:
    """The integral identity of Eq. 32: exp(int gamma_A dtau) = M_end/M_start.
    Compare against the product of per-step (1 + h*gamma_A) factors, or
    against exp(cumsum(gamma_A * h)), as an internal consistency check.
    """
    return M_end / M_start


def step_stability_margin(h: float, M: float, k_prime: float) -> float:
    """The dimensionless group of the explicit-coupling stability
    criterion (Eq. 35): |Delta ln M| per step. Stable for values <= 2.
    """
    return h * M * abs(k_prime)


def critical_step_size(M: float, k_prime: float) -> float:
    """h_crit = 2 / (M |k'|); the predicted collapse curve of Sec. 7.8."""
    if k_prime == 0.0:
        return np.inf
    return 2.0 / (M * abs(k_prime))
