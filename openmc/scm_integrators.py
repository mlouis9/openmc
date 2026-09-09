"""Exposure- and calendar-domain depletion integrators for SCM.

Both integrators use the same two-stage constant-extrapolation-predictor /
constant-midpoint-corrector (CE/CM) scheme already used by
``openmc.deplete.CECMIntegrator`` for ordinary depletion: deplete a half
step with beginning-of-step rates, evaluate rates at the resulting midpoint
composition, then deplete the full step from the beginning-of-step
composition using the midpoint rates. This is a deliberate choice: it is
the natural point of comparison for the two methods (Sec. 7.5 of the theory
note requires that they be compared under an identical integration scheme),
and averaging the beginning- and midpoint-of-step k estimates is exactly
the predictor-corrector damping analyzed in Sec. 5.6.

Neither integrator modifies the OpenMC C++ core; both are built entirely
from :mod:`scm_depletion.matrix_utils` and the existing CRAM solvers.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Callable

import numpy as np

from openmc.deplete.cram import CRAM16, CRAM48

from .scm_transport import SCMCoupledOperator, SCMOperatorResult, KFactors
from .scm_matrix_utils import exposure_matrix, calendar_matrix
from .scm_trajectory import SCMTrajectory, StepRecord

__all__ = ["ExposureIntegrator", "CalendarIntegrator", "StepStats"]

_CRAM = {16: CRAM16, 48: CRAM48}


@dataclass
class StepStats:
    """Lightweight per-step cost/diagnostic record, independent of the
    persisted trajectory (which lives in SCMTrajectory)."""
    index: int
    proc_time: float
    wall_time: float
    n_transport_calls: int


def _cram_step(cram, A_list: List, n_list: List[np.ndarray], dt: float):
    return [cram(A, n, dt) for A, n in zip(A_list, n_list)]


class _CECMBase:
    """Shared CE/CM stepping logic; subclasses differ only in how the
    Bateman matrix and the per-step "duration" (dtau or dt) are computed.
    """

    def __init__(self, operator: SCMCoupledOperator,
                 initial_vec: List[np.ndarray],
                 source_strength: float,
                 cram_order: int = 48,
                 n_inactive_schedule: Optional[Callable[[int], int]] = None,
                 diagnostic_fn: Optional[Callable[[SCMOperatorResult, SCMOperatorResult], dict]] = None):
        self.operator = operator
        self.source_strength = source_strength
        self.cram = _CRAM[cram_order]
        self.n_inactive_schedule = n_inactive_schedule
        self.diagnostic_fn = diagnostic_fn
        self._vec = [np.array(v, dtype=np.float64) for v in initial_vec]
        self._step_index = 0

    def _maybe_set_n_inactive(self):
        """Best-effort warm-start hook (Sec. 6.6 / theory Sec. 5.6): shrink
        the per-step inactive-cycle count using a user-supplied schedule,
        exploiting that the source bank is only slightly perturbed between
        consecutive depletion steps once it has been warm-started once.

        This degrades gracefully (a no-op with a one-time warning) on an
        OpenMC build whose ``openmc.lib`` does not expose a settable
        ``inactive`` batch count at run time.
        """
        if self.n_inactive_schedule is None:
            return
        n_inactive = self.n_inactive_schedule(self._step_index)
        try:
            import openmc.lib as lib
            lib.settings.inactive = int(n_inactive)
        except Exception:
            if not getattr(self, "_warned_n_inactive", False):
                import warnings
                warnings.warn(
                    "n_inactive_schedule was supplied but openmc.lib does "
                    "not expose a settable 'inactive' batch count in this "
                    "build; warm-start scheduling has no effect.")
                self._warned_n_inactive = True

    def _build_matrix(self, rates, k_factors: KFactors, fission_yields):
        raise NotImplementedError

    def _duration_of(self, k_bos: KFactors, k_mid: KFactors, h: float) -> float:
        """Physical increment corresponding to one step of size h in this
        integrator's native coordinate (used for the companion coordinate's
        clock/exposure bookkeeping)."""
        raise NotImplementedError

    def step(self, h: float) -> "StepRecord":
        t0 = time.perf_counter()
        n_transport_calls = 0

        self._maybe_set_n_inactive()
        res0 = self.operator(self._vec)
        n_transport_calls += 1

        A0 = self._build_matrix(res0.rates, res0.k_factors, res0.fission_yields)
        n_half = _cram_step(self.cram, A0, self._vec, h / 2.0)

        res_mid = self.operator(n_half)
        n_transport_calls += 1

        A_mid = self._build_matrix(res_mid.rates, res_mid.k_factors,
                                    res_mid.fission_yields)
        n_end = _cram_step(self.cram, A_mid, self._vec, h)

        record = self._make_record(res0, res_mid, h, n_end)
        if self.diagnostic_fn is not None:
            record.diagnostics.update(self.diagnostic_fn(res0, res_mid))

        self._vec = n_end
        self._step_index += 1

        wall_time = time.perf_counter() - t0
        record.wall_time = wall_time
        record.n_transport_calls = n_transport_calls
        return record

    def _make_record(self, res0, res_mid, h, n_end) -> "StepRecord":
        raise NotImplementedError

    @property
    def vec(self):
        return self._vec


class ExposureIntegrator(_CECMBase):
    """Method C: step in the exposure coordinate tau.

    ``h`` passed to :meth:`step` (or the ``dtau_grid`` used by
    :meth:`run`) is an exposure increment. The physical clock is
    accumulated as ``dt = (h/S0) * q_mid`` with ``q_mid = 1 - k_mid``
    (theory Eq. 6, evaluated with the CE/CM midpoint k for the same reason
    the midpoint rate is used for the transmutation matrix).
    """

    def _build_matrix(self, rates, k_factors, fission_yields):
        return exposure_matrix(self.operator.chain, rates, k_factors,
                                self.source_strength, fission_yields)

    def _make_record(self, res0, res_mid, h, n_end) -> StepRecord:
        q_mid = 1.0 - res_mid.k_factors.k
        dt = h * q_mid / self.source_strength
        return StepRecord(
            index=self._step_index,
            dtau=h, dt=dt,
            k_bos=res0.k_factors, k_mid=res_mid.k_factors,
            vec=n_end,
        )

    def run(self, trajectory: SCMTrajectory, dtau_grid: Sequence[float],
            resume: bool = True):
        """Integrate across a (possibly non-uniform) exposure grid,
        appending one record per step to ``trajectory`` and checkpointing
        after every step so the campaign is resumable.

        With ``resume=False`` any records already loaded into
        ``trajectory`` from a pre-existing cache file are discarded first
        (:meth:`SCMTrajectory.reset`), so a fresh run never silently
        appends to stale on-disk state.
        """
        if resume:
            n_done = trajectory.n_steps
            if n_done > 0:
                self._vec = trajectory.last_vec()
                self._step_index = n_done
                dtau_grid = dtau_grid[n_done:]
        else:
            trajectory.reset()
            self._step_index = 0
        for h in dtau_grid:
            record = self.step(h)
            trajectory.append(record)
            trajectory.checkpoint()
        return trajectory

    def extend(self, trajectory: SCMTrajectory, additional_dtau: Sequence[float]):
        """Continue an exposure run whose guessed window (Sec. 4.4, step 1)
        turned out to be too short. Nothing already computed is discarded;
        the clock and exposure continue from where the trajectory left off.
        """
        return self.run(trajectory, additional_dtau, resume=True)


class CalendarIntegrator(_CECMBase):
    """Method A: step in physical time t directly.

    ``h`` is a calendar-time increment [s]. The accumulated exposure is
    tracked as a derived diagnostic, ``dtau = S0 * M_mid * h``, purely for
    comparison against Method C (Sec. 7.5); it plays no role in the update
    itself.
    """

    def _build_matrix(self, rates, k_factors, fission_yields):
        return calendar_matrix(self.operator.chain, rates, k_factors,
                                self.source_strength, fission_yields)

    def _make_record(self, res0, res_mid, h, n_end) -> StepRecord:
        M_mid = res_mid.k_factors.M
        dtau = h * self.source_strength * M_mid
        return StepRecord(
            index=self._step_index,
            dtau=dtau, dt=h,
            k_bos=res0.k_factors, k_mid=res_mid.k_factors,
            vec=n_end,
        )

    def run(self, trajectory: SCMTrajectory, dt_grid: Sequence[float],
            resume: bool = True):
        if resume:
            n_done = trajectory.n_steps
            if n_done > 0:
                self._vec = trajectory.last_vec()
                self._step_index = n_done
                dt_grid = dt_grid[n_done:]
        else:
            trajectory.reset()
            self._step_index = 0
        for h in dt_grid:
            record = self.step(h)
            trajectory.append(record)
            trajectory.checkpoint()
        return trajectory

    def step_stability_ratio(self, h: float, k_mid: KFactors, k_prime: float) -> float:
        """Return h * M * |k'| for the just-taken (or about-to-be-taken)
        step, the dimensionless group of the step-size stability criterion
        (theory Eq. 35). Values above 2 indicate the explicit-coupling
        instability threshold has been crossed.
        """
        return h * k_mid.M * abs(k_prime)


class RegulatedBeamIntegrator:
    """Mode P (regulated beam): step in physical time with the beam
    current adjusted every cycle to hold thermal power fixed (theory
    Eq. 9), so that neither k nor M appears in the update at all.

    d N / dt = (P / R_f'(N)) * A_trans(N) + D

    ``energy_rate_fn(SCMOperatorResult) -> float`` must return R_f', the
    normalized (per-source-particle) energy-weighted fission rate,
    kappa * <Sigma_f, phi-bar> in the notation of the theory note --
    i.e. exactly the quantity ordinary ``openmc.deplete`` normalization
    helpers compute internally for "energy-deposition"/"fission-q"
    normalization modes, but not otherwise exposed by
    ``SCMOperatorResult``. Wire this up against the installed library's
    reaction-rate indexing (mirrors the caveat already flagged in
    ``_total_fission_rate`` in the E1 experiment driver).

    Same CE/CM stepping scheme as the other two integrators, for a
    like-for-like comparison in E8.
    """

    def __init__(self, operator: SCMCoupledOperator,
                 initial_vec: List[np.ndarray],
                 power: float,
                 energy_rate_fn,
                 cram_order: int = 48):
        self.operator = operator
        self.power = power
        self.energy_rate_fn = energy_rate_fn
        self.cram = _CRAM[cram_order]
        self._vec = [np.array(v, dtype=np.float64) for v in initial_vec]
        self._step_index = 0

    def _build_matrix(self, rates, fission_yields, rf_prime: float):
        from .scm_matrix_utils import split_full_matrix
        A_trans, D = split_full_matrix(self.operator.chain, rates, fission_yields)
        scale = self.power / rf_prime
        return [scale * At + Dm for At, Dm in zip(A_trans, D)]

    def step(self, h: float) -> "StepRecord":
        res0 = self.operator(self._vec)
        rf0 = self.energy_rate_fn(res0)
        A0 = self._build_matrix(res0.rates, res0.fission_yields, rf0)
        n_half = _cram_step(self.cram, A0, self._vec, h / 2.0)

        res_mid = self.operator(n_half)
        rf_mid = self.energy_rate_fn(res_mid)
        A_mid = self._build_matrix(res_mid.rates, res_mid.fission_yields, rf_mid)
        n_end = _cram_step(self.cram, A_mid, self._vec, h)

        beam_current = self.power / rf_mid  # implied S0-equivalent for reporting
        record = StepRecord(
            index=self._step_index, dtau=0.0, dt=h,
            k_bos=res0.k_factors, k_mid=res_mid.k_factors, vec=n_end,
            diagnostics={"rf_prime": rf_mid, "implied_beam_current": beam_current},
        )
        self._vec = n_end
        self._step_index += 1
        return record

    def run(self, trajectory: SCMTrajectory, dt_grid: Sequence[float],
            resume: bool = True):
        if resume:
            n_done = trajectory.n_steps
            if n_done > 0:
                self._vec = trajectory.last_vec()
                self._step_index = n_done
                dt_grid = dt_grid[n_done:]
        else:
            trajectory.reset()
            self._step_index = 0
        for h in dt_grid:
            record = self.step(h)
            trajectory.append(record)
            trajectory.checkpoint()
        return trajectory

    @property
    def vec(self):
        return self._vec
