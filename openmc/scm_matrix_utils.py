"""Bateman matrix assembly for exposure- and calendar-domain stepping.

Ordinary ``openmc.deplete`` builds one Bateman matrix per material via
``chain.form_matrix(rates, fission_yields)``, where ``rates`` are already
scaled to reactions/second and the transmutation and (fixed, physical)
decay contributions are summed inside that one call. There is no separate
hook for a decay-only matrix, so to assemble the exposure-domain matrix

    A_tau = A_trans(rates) + (q / S0) * D                       (theory Eq. 6)

or the calendar-domain matrix

    A_t   = S0 * M_hat * A_trans(rates) + D                     (theory Eq. 4)

the transmutation and decay contributions are separated by exploiting the
fact that ``form_matrix`` is affine in its rate argument with
``form_matrix(0, *) = D``: the decay-only matrix is recovered by forming
the matrix from an all-zero rate array, and the pure transmutation matrix
follows by subtraction. This requires no change to ``Chain.form_matrix``
and holds for any depletion chain, SCM or otherwise.
"""

from __future__ import annotations

import copy
from typing import List

from .scm_transport import KFactors

__all__ = ["split_full_matrix", "exposure_matrix", "calendar_matrix"]


def _zero_like(rates, n_mats: int):
    zero = copy.deepcopy(rates)
    for i in range(n_mats):
        zero[i] = rates[i] * 0.0
    return zero


def _scaled(rates, n_mats: int, factor: float):
    scaled = copy.deepcopy(rates)
    for i in range(n_mats):
        scaled[i] = rates[i] * factor
    return scaled


def split_full_matrix(chain, rates, fission_yields: list) -> tuple:
    """Decompose the per-material Bateman matrices into transmutation and
    decay parts.

    Returns
    -------
    (A_trans, D) : tuple of lists of scipy.sparse matrices
        ``A_trans[i]`` is the pure transmutation matrix for material ``i``,
        linear in ``rates[i]``; ``D[i]`` is the physical decay matrix,
        independent of ``rates``.
    """
    n_mats = len(fission_yields)
    zero_rates = _zero_like(rates, n_mats)

    A_full = [chain.form_matrix(rates[i], fission_yields[i])
              for i in range(n_mats)]
    D = [chain.form_matrix(zero_rates[i], fission_yields[i])
         for i in range(n_mats)]
    A_trans = [full - d for full, d in zip(A_full, D)]
    return A_trans, D


def exposure_matrix(chain, rates, k_factors: KFactors,
                     source_strength: float, fission_yields: list) -> List:
    """Assemble d N / d tau = A_tau N (theory Eq. 6), per material.

    A_tau = A_trans(rates) + [(1 - k) / S0] * D

    ``rates`` must be the raw per-source-particle rates returned by
    :class:`~scm_depletion.scm_transport.SCMCoupledOperator` (i.e. obtained
    with ``source_rate=1.0``); this function performs no additional
    rescaling of the transmutation part.
    """
    A_trans, D = split_full_matrix(chain, rates, fission_yields)
    q_over_s0 = (1.0 - k_factors.k) / source_strength
    return [At + q_over_s0 * Dm for At, Dm in zip(A_trans, D)]


def calendar_matrix(chain, rates, k_factors: KFactors,
                     source_strength: float, fission_yields: list) -> List:
    """Assemble d N / d t = A_t N (theory Eq. 4), per material.

    A_t = S0 * M_hat * A_trans(rates) + D

    ``rates`` must again be the raw per-source-particle rates; the
    S0 * M_hat rescaling that converts them to reactions/second is applied
    here, once, so that the same ``SCMOperatorResult`` serves either
    integration mode.
    """
    A_trans, D = split_full_matrix(chain, rates, fission_yields)
    scale = source_strength * k_factors.M
    return [scale * At + Dm for At, Dm in zip(A_trans, D)]
