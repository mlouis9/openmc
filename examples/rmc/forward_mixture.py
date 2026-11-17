"""Weights for a continuous signed difference sampled from a positive mixture.

This is estimator infrastructure, not a transport implementation.  The caller
must supply densities with respect to the same measure, including any path,
reaction and energy marginalization.  In particular, a fixed-incident-energy
elastic law cannot be treated as an ordinary energy-angle density.
"""

import numpy as np


def difference_weights(ce_density, low_density, ce_mass, low_mass, ce_probability):
    """Return (CE - low) / proposal at states sampled from the mixture.

    The proposal is ``p * ce_density / ce_mass + (1-p) * low_density /
    low_mass``. Masses must be integrals over the *same* state space and
    conditional variables as the supplied densities. Source amplitude,
    branch-selection probabilities and sample-count normalization are the
    caller's responsibility.
    """
    ce, low = np.broadcast_arrays(np.asarray(ce_density, dtype=float),
                                  np.asarray(low_density, dtype=float))
    if (not np.isfinite(ce_mass) or not np.isfinite(low_mass)
            or ce_mass <= 0 or low_mass <= 0):
        raise ValueError('both mixture masses must be finite and positive')
    if not np.isfinite(ce_probability) or not 0 < ce_probability < 1:
        raise ValueError('mixture probability must be strictly between 0 and 1')
    if (np.any(~np.isfinite(ce)) or np.any(~np.isfinite(low))
            or np.any(ce < 0) or np.any(low < 0)):
        raise ValueError('densities must be finite and nonnegative')
    proposal = ce_probability * ce / ce_mass + (1 - ce_probability) * low / low_mass
    if np.any(proposal <= 0):
        raise ValueError('sampled state has zero proposal density')
    return (ce - low) / proposal
