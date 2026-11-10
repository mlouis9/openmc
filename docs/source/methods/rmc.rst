.. _methods_rmc:

==========================================
Multilevel Residual Monte Carlo Correction
==========================================

.. note::

    This chapter documents an experimental method under development.
    Sections are added as the corresponding code is implemented, so that
    every equation referenced from the source code is described here.

.. _methods_rmc_overview:

--------
Overview
--------

Multilevel residual Monte Carlo (RMC) correction solves the
continuous-energy (CE) :math:`k`-eigenvalue problem starting from a converged
multigroup random ray solution (see :ref:`methods_random_ray`). The random
ray emission source is frozen and lifted to continuous energy; this is the
*field*. Each generation, the difference between the CE emission produced
from the field and the field itself is estimated at sampled destinations by
backward characteristics, and stored as signed CE source particles, the
*bank*. The bank is transported forward with analog CE flights. The
eigenvalue is updated from the combined fission production of field and
bank.

The residual is estimated on two levels that share random inputs. On a small
number of backward paths, CE production is evaluated together with the
multigroup production of the same field on the same paths, so that only their
difference contributes noise. On a larger number of paths, the multigroup
production alone is evaluated, which is inexpensive. This is a two-level
(multilevel) Monte Carlo estimator with the multigroup model as the coarse
level.

Write the CE transport operators as

.. math::
    :label: rmc-operators

    \mathcal{T} = \mathbf{\Omega}\cdot\nabla + \Sigma_t, \qquad
    \mathcal{P}(k) = \mathcal{S} + \frac{1}{k}\mathcal{F}, \qquad
    \mathcal{G} = \mathcal{T}^{-1}, \qquad
    \mathcal{B} = \mathcal{P}\mathcal{G},

where :math:`\mathcal{S}` is nonfission neutron production (scattering and
other multiplying reactions, with expected multiplicities) and
:math:`\mathcal{F}` is fission production. The emission source :math:`Q`
satisfies :math:`Q = \mathcal{B}(k)\,Q` with unit fission production.

Notation follows the rest of this manual: :math:`E', \mathbf{\Omega}'` are
incident energy and direction at a production event, and
:math:`E, \mathbf{\Omega}` the outgoing ones.

.. _methods_rmc_kinematics:

-------------------------------------
Elastic Kinematics, Stationary Target
-------------------------------------

The common-output elastic evaluator needs the laboratory relations of
two-body elastic scattering written as functions of the *laboratory* cosine
:math:`\mu_0 = \mathbf{\Omega}'\cdot\mathbf{\Omega}`, because the outgoing
direction is fixed first and the compatible incident energy is solved for
afterwards. This section collects those relations. They hold for a target at
rest with mass ratio :math:`A > 1`, which is what OpenMC samples whenever
``sample_target_velocity()`` in ``src/physics.cpp`` returns a zero target velocity (see
:ref:`freegas`).

.. _methods_rmc_kinematics_lab_cm:

Laboratory and center-of-mass cosines
-------------------------------------

With :math:`\mu_{cm}` the center-of-mass scattering cosine, the laboratory
cosine is

.. math::
    :label: rmc-mu-lab

    \mu_0 = \frac{1 + A\mu_{cm}}{\sqrt{A^2 + 2A\mu_{cm} + 1}}.

For :math:`A > 1` this is monotonic on :math:`[-1, 1]`, so it has a single
inverse. Writing :math:`s = \sqrt{A^2 - 1 + \mu_0^2}`,

.. math::
    :label: rmc-mu-cm

    \mu_{cm}(\mu_0) = \frac{\mu_0 s - (1 - \mu_0^2)}{A},

and its derivative is

.. math::
    :label: rmc-dmu

    \frac{d\mu_{cm}}{d\mu_0} = \frac{1}{A}\left( 2\mu_0 + s + \frac{\mu_0^2}{s} \right) > 0.

Both endpoints map to themselves: :math:`\mu_{cm}(\pm 1) = \pm 1`.

.. _methods_rmc_kinematics_energy:

Outgoing energy ratio
---------------------

The ratio of outgoing to incident energy is

.. math::
    :label: rmc-gamma

    \gamma_A(\mu_0) = \frac{E}{E'} =
    \left[ \frac{\mu_0 + \sqrt{A^2 - 1 + \mu_0^2}}{A + 1} \right]^2,

with :math:`\gamma_A(1) = 1` and
:math:`\gamma_A(-1) = \alpha = \left(\frac{A-1}{A+1}\right)^2`.

.. _methods_rmc_kinematics_density:

Laboratory angular density
--------------------------

Elastic angular distributions in the nuclear data are given in the
center-of-mass frame as :math:`p_{cm}(\mu_{cm} \mid E')`. The corresponding
laboratory density is

.. math::
    :label: rmc-p-lab

    p_{lab}(\mu_0 \mid E') = p_{cm}\big(\mu_{cm}(\mu_0) \mid E'\big)\,
    \frac{d\mu_{cm}}{d\mu_0}.

**Check.** For isotropic center-of-mass scattering, :math:`p_{cm} = 1/2`, and
the outgoing energy is uniform on :math:`[\alpha E', E']`. This is the
property tested by ``tests/cpp_unit_tests/test_rmc_kinematics.cpp``.

**Limitations.** These relations do not describe hydrogen (:math:`A < 1`),
target motion (free gas), or bound scattering. Those require separate
treatments.
