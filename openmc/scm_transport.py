"""Transport-side plumbing for SCM depletion.

This module contains the *one* integration point that is specific to a
particular SCM-enabled OpenMC build: :func:`_fetch_k_factors`, which reads
the per-cycle multiplication estimators (k, kq, ks, and their statistical
uncertainties) off the just-completed transport solve. Everything else in
this package is written against the public ``openmc.deplete`` API and the
existing ``openmc.deplete.CoupledOperator``/``cram`` machinery, and does not
depend on fork-specific internals.

Unit convention
---------------
``SCMCoupledOperator`` always runs the transport solve exactly once per
call and always extracts reaction rates *normalized per source particle*
of the current normalized total-source bank, i.e. the quantity the theory
note calls ``A(N)`` (dimension: reactions per starting particle, per atom).
This is obtained by forcing ``normalization_mode="source-rate"`` and calling
the inherited ``_calculate_reaction_rates`` with ``source_rate=1.0``: in
source-rate normalization the scale factor is exactly the value passed in,
so passing 1.0 recovers the raw per-particle tally rather than a rate in
reactions/second.

Both integration modes are then just different choices of what to multiply
this per-particle rate by before building the Bateman matrix:

- Exposure stepping (Method C) uses it unscaled, since d tau already counts
  source particles.
- Calendar stepping (Method A) scales it by ``S0 * M_hat`` to obtain an
  ordinary reactions-per-second rate.

Both cases are handled in :mod:`scm_depletion.matrix_utils`; this module is
only responsible for producing the per-particle rate and the k/kq/ks triple
alongside it.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from typing import Optional, Sequence, Callable

import numpy as np
from uncertainties import ufloat

import openmc
import openmc.lib
from openmc.deplete import CoupledOperator
from openmc.deplete.abc import OperatorResult

__all__ = [
    "KFactors", "SCMOperatorResult", "SCMCoupledOperator",
    "KOverride", "ForcedKSequence", "FrozenComposition", "CompositionPerturbation",
    "VariableParticleCount",
]


# --------------------------------------------------------------------------
# The one fork-specific integration point
# --------------------------------------------------------------------------

def _fetch_k_factors() -> "KFactors":
    """Read k, kq, ks and their statistical uncertainties from the just-
    completed SCM transport solve.

    This mirrors the existing pattern ``ufloat(*openmc.lib.keff())`` used
    for ordinary eigenvalue/fixed-source depletion in
    ``openmc.deplete.CoupledOperator.__call__``. The SCM fork's C API is
    expected to expose an analogous accessor; several plausible names are
    tried below so that a single edit to this function is enough to adapt
    the package to the exact binding names of a given build.

    Required fork behaviour (see ``settings.calculate_subcritical_k`` /
    ``settings.print_all_k_factors`` and ``simulation::k``, ``k_std``,
    ``kq``, ``kq_std``, ``ks``, ``ks_std`` in ``simulation.cpp``): after
    ``openmc.lib.run()`` returns in ``SUBCRITICAL_MULTIPLICATION`` mode, the
    three estimators and their standard errors must be readable from Python.
    """
    for name in ("subcritical_k_factors", "subcritical_k", "k_factors"):
        fn = getattr(openmc.lib, name, None)
        if fn is not None:
            k, k_std, kq, kq_std, ks, ks_std = fn()
            return KFactors(k, k_std, kq, kq_std, ks, ks_std)

    raise NotImplementedError(
        "No SCM k/kq/ks accessor found on openmc.lib. This build's C API "
        "must expose the per-cycle subcritical multiplication estimators "
        "(simulation::k, k_std, kq, kq_std, ks, ks_std in simulation.cpp) "
        "through a Python binding, e.g. 'openmc.lib.subcritical_k_factors()' "
        "returning the 6-tuple (k, k_std, kq, kq_std, ks, ks_std). Add the "
        "binding to the fork and update the `name` tuple at the top of "
        "scm_transport._fetch_k_factors accordingly."
    )


# --------------------------------------------------------------------------
# k, kq, ks and the derived multiplicity
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class KFactors:
    """The three subcritical multiplication factors and their statistics.

    ``M`` and its standard error are computed by the delta method,
    dM/dk = M**2 (Sec. 6.5 of the theory note), so that ``k_factors.M_ufloat``
    is directly usable in uncertainty propagation.
    """
    k: float
    k_std: float
    kq: float
    kq_std: float
    ks: float
    ks_std: float

    @property
    def M(self) -> float:
        return 1.0 / (1.0 - self.k)

    @property
    def M_std(self) -> float:
        return self.M ** 2 * self.k_std

    @property
    def k_ufloat(self):
        return ufloat(self.k, self.k_std)

    @property
    def kq_ufloat(self):
        return ufloat(self.kq, self.kq_std)

    @property
    def ks_ufloat(self):
        return ufloat(self.ks, self.ks_std)

    @property
    def M_ufloat(self):
        return 1.0 / (1.0 - self.k_ufloat)

    def with_k(self, k_new: float, k_std_new: Optional[float] = None) -> "KFactors":
        """Return a copy with k (and optionally k_std) overridden.

        Used by :func:`override_k` to implement the forward-injection
        perturbation of Sec. 7.6(a): "hold the transport result fixed,
        impose k -> k +/- eta in the depletion solve only."
        """
        return replace(self, k=k_new,
                        k_std=self.k_std if k_std_new is None else k_std_new)


def override_k(k_factors: KFactors, delta_k: float) -> KFactors:
    """Perturb k by ``delta_k`` with rates held fixed (Sec. 7.6a)."""
    return k_factors.with_k(k_factors.k + delta_k)


# --------------------------------------------------------------------------
# Operator result carrying the SCM-specific quantities
# --------------------------------------------------------------------------

@dataclass
class SCMOperatorResult:
    """Result of one SCM transport solve.

    Attributes
    ----------
    k_factors : KFactors
        The (k, kq, ks) triple and uncertainties for this cycle.
    rates : openmc.deplete.reaction_rates.ReactionRates
        Reaction rates normalized *per source particle* of the current
        normalized total-source bank (i.e. A(N), theory note Eq. 4).
        Never rescaled by S0 or M; scaling is applied in matrix_utils.
    fission_yields : list
        Per-material fission product yield dictionaries, as returned by
        ``OpenMCOperator._calculate_reaction_rates`` via
        ``self.chain.fission_yields``.
    """
    k_factors: KFactors
    rates: "object"
    fission_yields: list

    # Backward-compatible alias so this can stand in for OperatorResult
    # wherever the stock integrators' bookkeeping expects a `.rates`
    # attribute (e.g. cost accounting).
    @property
    def k(self):
        return self.k_factors.k_ufloat


class SCMCoupledOperator(CoupledOperator):
    """CoupledOperator specialized for SCM depletion.

    Parameters
    ----------
    model : openmc.model.Model
        Model whose ``settings.run_mode`` must already be configured for
        subcritical multiplication (and, if desired,
        ``settings.calculate_subcritical_k = True``).
    source_strength : float
        S0, the external source strength in neutrons/second. This is the
        only new physical input relative to ordinary fixed-source
        depletion; it converts the dimensionless per-particle rates into
        either an exposure-domain or calendar-domain Bateman matrix (see
        :mod:`scm_depletion.matrix_utils`).
    kwargs :
        Forwarded to ``CoupledOperator.__init__``. ``normalization_mode``
        is forced to ``"source-rate"`` regardless of what is passed, since
        that is the only normalization convention for which passing
        ``source_rate=1.0`` recovers the raw per-particle rate.
    """

    def __init__(self, model: "openmc.Model", source_strength: float, **kwargs):
        kwargs["normalization_mode"] = "source-rate"
        super().__init__(model, **kwargs)
        if source_strength <= 0:
            raise ValueError("source_strength (S0) must be positive")
        self.source_strength = source_strength

    def __call__(self, vec, source_rate=None) -> SCMOperatorResult:
        """Run one transport solve and return the per-particle rates.

        The ``source_rate`` argument is accepted (and ignored, beyond a
        sanity check) purely so that this operator remains a drop-in
        replacement wherever the stock ``openmc.deplete`` machinery calls
        ``operator(n, source_rate)``; SCM transport always normalizes to
        one particle drawn from the current total source, independent of
        the physical source strength, which enters only afterwards.
        """
        # Reset results in OpenMC (mirrors CoupledOperator.__call__).
        openmc.lib.reset()
        if self._n_calls > 0:
            openmc.lib.reset_timers()

        self._update_materials_and_nuclides(vec)
        openmc.lib.run()

        # Always extract the raw per-source-particle rate.
        rates = self._calculate_reaction_rates(1.0)
        k_factors = _fetch_k_factors()
        fission_yields = copy.deepcopy(self.chain.fission_yields)

        self._n_calls += 1
        return SCMOperatorResult(k_factors, copy.deepcopy(rates), fission_yields)


# --------------------------------------------------------------------------
# Perturbation-hook wrappers (Sections 7.3, 7.6)
# --------------------------------------------------------------------------

class _SCMOperatorWrapper:
    """Base class for operator decorators used by the numerical program.

    Delegates attribute access to the wrapped operator so that these
    wrappers are transparent to the integrators and to any code that
    inspects e.g. ``operator.chain`` or ``operator.source_strength``.
    """

    def __init__(self, base_operator: SCMCoupledOperator):
        self._base = base_operator

    def __getattr__(self, name):
        return getattr(self._base, name)

    def __call__(self, vec, source_rate=None) -> SCMOperatorResult:
        raise NotImplementedError


class KOverride(_SCMOperatorWrapper):
    """Ensemble-IV / Sec. 7.6(a) hook: perturb k by a prescribed sequence
    with the transport result (rates, composition response) held fixed.

    ``delta_k_sequence`` is indexed by call count; a scalar broadcasts to
    every step. This does *not* rerun transport with a different k -- it
    is the "depletion solve only" perturbation of Sec. 7.6(a), used to
    measure the forward injection vector g by finite differencing the
    resulting composition against delta_k.
    """

    def __init__(self, base_operator: SCMCoupledOperator,
                 delta_k_sequence):
        super().__init__(base_operator)
        self._deltas = delta_k_sequence
        self._call_index = 0

    def __call__(self, vec, source_rate=None) -> SCMOperatorResult:
        result = self._base(vec, source_rate)
        delta = (self._deltas if np.isscalar(self._deltas)
                 else self._deltas[self._call_index])
        self._call_index += 1
        return SCMOperatorResult(
            override_k(result.k_factors, delta),
            result.rates, result.fission_yields)


class ForcedKSequence(_SCMOperatorWrapper):
    """Ensemble II ("frozen normalization"): replace the measured k with a
    prescribed high-precision reference sequence k_bar(tau), while keeping
    this replica's own (noisy) reaction rates.

    This removes the normalization-noise channel from the replica's error
    budget while retaining the shape-noise channel, isolating a_f and its
    covariance with k (Sec. 7.3, Ensemble II).
    """

    def __init__(self, base_operator: SCMCoupledOperator, k_bar_sequence):
        super().__init__(base_operator)
        self._k_bar = k_bar_sequence
        self._call_index = 0

    def __call__(self, vec, source_rate=None) -> SCMOperatorResult:
        result = self._base(vec, source_rate)
        k_bar = self._k_bar[self._call_index]
        self._call_index += 1
        forced = result.k_factors.with_k(k_bar, k_std_new=0.0)
        return SCMOperatorResult(forced, result.rates, result.fission_yields)


class FrozenComposition(_SCMOperatorWrapper):
    """Ensemble III ("frozen composition"): every call transports the same
    prescribed nominal composition sequence, regardless of what ``vec`` the
    caller passes in, and does not let the trajectory advance.

    Used to measure the innovation statistics a_k, a_f, and their
    cross-covariance directly, uncontaminated by propagation (Sec. 7.3).
    """

    def __init__(self, base_operator: SCMCoupledOperator, nominal_vec_sequence):
        super().__init__(base_operator)
        self._nominal = nominal_vec_sequence
        self._call_index = 0

    def __call__(self, vec, source_rate=None) -> SCMOperatorResult:
        nominal_vec = self._nominal[self._call_index]
        self._call_index += 1
        return self._base(nominal_vec, source_rate)


class CompositionPerturbation(_SCMOperatorWrapper):
    """Ensemble V ("paired perturbation"): add a fixed offset to the
    composition vector before every transport solve.

    Pair a run using ``magnitude = +eta`` against one using ``-eta`` (or
    ``magnitude = 0``) *at an identical OpenMC RNG seed* to obtain the
    correlated-sampling estimate of s^T g used in Sec. 7.6(b). Seed control
    itself is the caller's responsibility (set ``model.settings.seed``
    identically for both runs); this wrapper only perturbs the state.
    """

    def __init__(self, base_operator: SCMCoupledOperator,
                 direction: Sequence[np.ndarray], magnitude: float):
        super().__init__(base_operator)
        self._direction = direction
        self._magnitude = magnitude

    def __call__(self, vec, source_rate=None) -> SCMOperatorResult:
        perturbed = [v + self._magnitude * d
                     for v, d in zip(vec, self._direction)]
        return self._base(perturbed, source_rate)


class VariableParticleCount(_SCMOperatorWrapper):
    """Sec. 6.6 / 7.9 hook: set a per-step particle count before each
    transport solve, for the budget-allocation experiment (proportional
    allocation n_l ~ h_l on a non-uniform grid, Eq. 44).

    Best-effort: attempts ``openmc.lib.settings.particles = n`` before
    each call. If the installed ``openmc.lib`` does not expose a settable
    particle count at run time, this degrades to a one-time warning and a
    no-op, leaving whatever particle count the model was built with.
    """

    def __init__(self, base_operator: SCMCoupledOperator,
                 particles_schedule: Sequence[int]):
        super().__init__(base_operator)
        self._schedule = particles_schedule
        self._call_index = 0
        self._warned = False

    def __call__(self, vec, source_rate=None) -> SCMOperatorResult:
        n = self._schedule[self._call_index]
        self._call_index += 1
        try:
            import openmc.lib as lib
            lib.settings.particles = int(n)
        except Exception:
            if not self._warned:
                import warnings
                warnings.warn(
                    "VariableParticleCount: openmc.lib does not expose a "
                    "settable 'particles' count at run time in this build; "
                    "per-step budget allocation has no effect.")
                self._warned = True
        return self._base(vec, source_rate)
