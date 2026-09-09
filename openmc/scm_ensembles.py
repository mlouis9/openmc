"""Constructors for the five replica ensembles of Sec. 7.3.

Each factory wraps a base :class:`~scm_depletion.scm_transport.SCMCoupledOperator`
with the appropriate perturbation hook and returns a ready-to-run
:class:`~scm_depletion.integrators.ExposureIntegrator` (or
:class:`~scm_depletion.integrators.CalendarIntegrator`). The wrapped
operator remains a transparent stand-in for the base operator everywhere
the integrator needs it (chain access, source_strength, ...), via
``_SCMOperatorWrapper.__getattr__``.

Ensemble I is the reference (ordinary closed-loop depletion); II-V isolate
individual channels of the error model, as described at the top of Sec.
7.3 and repeated in each factory's docstring below.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from .scm_transport import (
    SCMCoupledOperator, KOverride, ForcedKSequence,
    FrozenComposition, CompositionPerturbation,
)
from .scm_integrators import ExposureIntegrator, CalendarIntegrator

__all__ = [
    "ensemble_I_closed_loop", "ensemble_II_frozen_normalization",
    "ensemble_III_frozen_composition", "ensemble_IV_injected_noise",
    "ensemble_V_paired_perturbation",
]

_INTEGRATORS = {"exposure": ExposureIntegrator, "calendar": CalendarIntegrator}


def _make_integrator(mode: str, operator, initial_vec, source_strength,
                      **integrator_kwargs):
    if mode not in _INTEGRATORS:
        raise ValueError(f"mode must be 'exposure' or 'calendar', got {mode!r}")
    return _INTEGRATORS[mode](operator, initial_vec, source_strength,
                               **integrator_kwargs)


def ensemble_I_closed_loop(base_operator: SCMCoupledOperator,
                            initial_vec: List[np.ndarray],
                            source_strength: float,
                            mode: str = "exposure",
                            **integrator_kwargs):
    """The reference ensemble: ordinary coupled SCM depletion with the
    replica's own random seed. Compared against every other ensemble and
    used directly for the master scaling table (Sec. 7.4) and the method
    comparison (Sec. 7.5).
    """
    return _make_integrator(mode, base_operator, initial_vec,
                             source_strength, **integrator_kwargs)


def ensemble_II_frozen_normalization(base_operator: SCMCoupledOperator,
                                      initial_vec: List[np.ndarray],
                                      source_strength: float,
                                      k_bar_sequence: Sequence[float],
                                      mode: str = "exposure",
                                      **integrator_kwargs):
    """Frozen normalization: every replica is forced to use a common
    high-precision k_bar(tau) (e.g. from a very-high-N reference run),
    retaining its own shape noise. Removes the normalization channel and
    the feedback, isolating the shape-noise channel alone.
    """
    wrapped = ForcedKSequence(base_operator, k_bar_sequence)
    return _make_integrator(mode, wrapped, initial_vec, source_strength,
                             **integrator_kwargs)


def ensemble_III_frozen_composition(base_operator: SCMCoupledOperator,
                                     nominal_vec_sequence: Sequence[List[np.ndarray]],
                                     source_strength: float,
                                     mode: str = "exposure",
                                     **integrator_kwargs):
    """Frozen composition: every replica transports the same nominal
    composition sequence, with no propagation. The innovation reference:
    measures a_k, the shape-noise magnitude a_f, and their cross-
    covariance, uncontaminated by feedback.

    Note this ensemble does not "run" a trajectory in the usual sense --
    the composition never advances according to the replica's own tallies
    -- so the returned integrator's ``step`` method should be called
    directly, once per entry of ``nominal_vec_sequence``, and the
    resulting per-step k/rate estimates collected for statistics rather
    than treated as a depletion trajectory.
    """
    wrapped = FrozenComposition(base_operator, nominal_vec_sequence)
    initial_vec = nominal_vec_sequence[0]
    return _make_integrator(mode, wrapped, initial_vec, source_strength,
                             **integrator_kwargs)


def ensemble_IV_injected_noise(base_operator: SCMCoupledOperator,
                                initial_vec: List[np.ndarray],
                                source_strength: float,
                                delta_k_sequence,
                                mode: str = "exposure",
                                **integrator_kwargs):
    """Injected noise: a prescribed, known input delta_k_l = eta * xi_l is
    added to k at every step (transport noise itself should be suppressed
    by the caller, e.g. via a very large particle count or a deterministic
    reference k(tau), so that the injected signal dominates). Turns the
    identification of the closed-loop operator L into a controlled-input
    (ARX) system-identification problem (Sec. 7.6, controlled-input
    identification).
    """
    wrapped = KOverride(base_operator, delta_k_sequence)
    return _make_integrator(mode, wrapped, initial_vec, source_strength,
                             **integrator_kwargs)


def ensemble_V_paired_perturbation(base_operator: SCMCoupledOperator,
                                    initial_vec: List[np.ndarray],
                                    source_strength: float,
                                    direction: Sequence[np.ndarray],
                                    magnitude: float,
                                    mode: str = "exposure",
                                    **integrator_kwargs):
    """Paired perturbation: the composition is displaced by
    ``magnitude * direction`` before every transport solve. Pair a run at
    ``magnitude = +eta`` against one at ``magnitude = -eta`` (or ``0``),
    run at an *identical* OpenMC seed (set by the caller), to obtain the
    correlated-sampling estimate of s^T g (Sec. 7.6b) or, with
    ``direction`` chosen orthogonal to g, to probe the transverse
    spectrum of L (Sec. 7.7).
    """
    wrapped = CompositionPerturbation(base_operator, direction, magnitude)
    return _make_integrator(mode, wrapped, initial_vec, source_strength,
                             **integrator_kwargs)
