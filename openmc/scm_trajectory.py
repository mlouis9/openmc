"""Cache-keyed, resumable storage for one SCM depletion trajectory, and the
interpolated-calendar-delivery algorithm of Sec. 4.4.

A trajectory is the parametric curve {(tau_j, t_j, N_j)} produced by the
exposure integrator (or, for a calendar run, {(t_j, tau_j, N_j)} with the
roles of independent/derived variable swapped). It is stored as a single
pickle file per replica under a cache directory, written atomically after
every step so that a campaign can be resumed after a preemption on the
cluster without repeating any transport solves.
"""

from __future__ import annotations

import os
import pickle
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
from scipy.interpolate import PchipInterpolator

from .scm_transport import KFactors

__all__ = ["StepRecord", "SCMTrajectory", "WindowTooShort"]


class WindowTooShort(Exception):
    """Raised by :meth:`SCMTrajectory.interpolate_at_time` when the
    requested calendar date lies beyond the computed clock; the caller
    should extend the integration (Sec. 4.4, step 4) and retry."""
    def __init__(self, t_requested: float, t_available: float):
        super().__init__(
            f"requested t*={t_requested:.6g} exceeds the computed clock "
            f"(t_available={t_available:.6g}); extend the exposure grid.")
        self.t_requested = t_requested
        self.t_available = t_available


@dataclass
class StepRecord:
    index: int
    dtau: float
    dt: float
    k_bos: KFactors
    k_mid: KFactors
    vec: List[np.ndarray]
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    wall_time: Optional[float] = None
    n_transport_calls: Optional[int] = None


class SCMTrajectory:
    """One replica's parametric depletion trajectory.

    Parameters
    ----------
    cache_dir : path-like
        Directory in which ``trajectory.pkl`` is stored. Created if it
        does not exist.
    metadata : dict, optional
        Free-form case/replica metadata (case name, k target, seed,
        particle/generation counts, ...) persisted alongside the
        trajectory for later bookkeeping.
    """

    def __init__(self, cache_dir, metadata: Optional[dict] = None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata = metadata or {}
        self.records: List[StepRecord] = []
        self.tau: List[float] = [0.0]
        self.t: List[float] = [0.0]
        if self._path.exists():
            self._load_existing()

    @property
    def _path(self) -> Path:
        return self.cache_dir / "trajectory.pkl"

    @property
    def n_steps(self) -> int:
        return len(self.records)

    def last_vec(self) -> List[np.ndarray]:
        if not self.records:
            raise IndexError("trajectory has no steps yet")
        return [v.copy() for v in self.records[-1].vec]

    def append(self, record: StepRecord):
        self.records.append(record)
        self.tau.append(self.tau[-1] + record.dtau)
        self.t.append(self.t[-1] + record.dt)

    def reset(self):
        """Discard any records loaded from a pre-existing cache file. Used
        by the integrators' ``run(..., resume=False)`` path so that a
        fresh run does not silently append to stale on-disk state.
        """
        self.records = []
        self.tau = [0.0]
        self.t = [0.0]

    def checkpoint(self):
        """Atomically persist the trajectory to disk (write-then-rename),
        so a preempted SLURM job resumes cleanly rather than corrupting
        the cache file.
        """
        payload = {
            "metadata": self.metadata,
            "records": self.records,
            "tau": self.tau,
            "t": self.t,
        }
        fd, tmp_path = tempfile.mkstemp(dir=self.cache_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path, self._path)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    def _load_existing(self):
        with open(self._path, "rb") as f:
            payload = pickle.load(f)
        self.metadata = payload["metadata"]
        self.records = payload["records"]
        self.tau = payload["tau"]
        self.t = payload["t"]

    @classmethod
    def load(cls, cache_dir) -> "SCMTrajectory":
        return cls(cache_dir)

    # -- exposure-indexed access: multiplication-free, no interpolation
    #    subtlety since tau is the trajectory's own independent variable --

    def nuclide_at_tau(self, tau_star: float, mat_index: int = 0) -> np.ndarray:
        """N(tau*), interpolated on the exposure grid. M-free (theory
        Sec. 4.1, Eq. 12); no clock is involved.
        """
        if tau_star > self.tau[-1]:
            raise WindowTooShort(tau_star, self.tau[-1])
        stacked = np.stack([np.zeros_like(self.records[0].vec[mat_index])]
                            + [r.vec[mat_index] for r in self.records])
        interp = PchipInterpolator(self.tau, stacked, axis=0)
        return interp(tau_star)

    def k_at_tau(self, tau_star: float) -> float:
        taus = self.tau[1:]
        ks = [r.k_mid.k for r in self.records]
        return float(np.interp(tau_star, taus, ks))

    # -- calendar-indexed access: the interpolation algorithm of Sec. 4.4 --

    def interpolate_at_time(self, t_star: float, mat_index: int = 0,
                             method: str = "pchip") -> dict:
        """Deliver N(t*) from the exposure trajectory (Sec. 4.4).

        Steps 5-6 of the algorithm: locate tau_hat such that t(tau_hat) =
        t_star by monotone root-finding on the tabulated clock (t is
        strictly increasing in tau since dt/dtau = q/S0 > 0 for any
        subcritical k), then interpolate N on the exposure grid at
        tau_hat.

        Raises
        ------
        WindowTooShort
            If ``t_star`` exceeds the clock computed so far (step 4 of the
            algorithm: extend the integration and retry).

        Returns
        -------
        dict with keys 'tau_hat', 'N', 'k_hat' (the multiplication at the
        interpolated point, useful for reconstructing rates at t*).
        """
        if t_star > self.t[-1]:
            raise WindowTooShort(t_star, self.t[-1])

        if method == "pchip":
            tau_of_t = PchipInterpolator(self.t, self.tau)
            tau_hat = float(tau_of_t(t_star))
        else:
            tau_hat = float(np.interp(t_star, self.t, self.tau))

        N_hat = self.nuclide_at_tau(tau_hat, mat_index=mat_index)
        k_hat = self.k_at_tau(tau_hat)
        return {"tau_hat": tau_hat, "N": N_hat, "k_hat": k_hat}

    def rate_at_time(self, t_star: float, diagnostic_key: str,
                      method: str = "pchip") -> float:
        """Interpolate a scalar diagnostic (e.g. a normalized reaction
        rate stashed in ``StepRecord.diagnostics``) at a calendar date,
        then apply the normalization channel to obtain the absolute rate
        R(t*) = S0 * M_hat * r(t*), for the table row of Sec. 6.3 that
        needs it (endpoint-local normalization, no window contribution
        beyond what interpolation already carries)."""
        info = self.interpolate_at_time(t_star, method=method)
        taus = self.tau[1:]
        vals = [r.diagnostics[diagnostic_key] for r in self.records]
        r_hat = float(np.interp(info["tau_hat"], taus, vals))
        M_hat = 1.0 / (1.0 - info["k_hat"])
        return self.metadata.get("source_strength", 1.0) * M_hat * r_hat

    # -- reporting the clock as a stretch band (Sec. 6.6) --

    def clock_stretch_band(self) -> dict:
        """Decompose the accumulated clock deviation into its dominant
        Karhunen-Loeve mode (a near-uniform stretch of the time axis,
        carrying 8/pi^2 ~ 81% of the variance for a pure random walk) plus
        residual jitter, per the reporting prescription of Sec. 6.6.

        This is a convenience for *reporting*; the actual variance must
        come from a replica ensemble (see common.replica_stats applied to
        a set of SCMTrajectory objects at matched tau).
        """
        tau = np.asarray(self.tau)
        tau_star = tau[-1]
        mode_shape = np.sin(np.pi * tau / (2.0 * tau_star))
        return {"tau": tau, "mode_shape": mode_shape,
                "leading_mode_variance_fraction": 8.0 / np.pi**2}
