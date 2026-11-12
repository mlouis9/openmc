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

.. _methods_rmc_field:

----------------
The Frozen Field
----------------

The field is the multigroup emission source of a converged random ray
eigenvalue solution, frozen and lifted to continuous energy. For source region
:math:`R` with material :math:`m`, group :math:`g`, and tag
:math:`t \in \{s, f\}` (scattering-born or fission-born),

.. math::
    :label: rmc-field

    Q^{MG}_t(\mathbf{r}, E, \mathbf{\Omega}) = \frac{1}{4\pi}\,
    Q_{t,R(\mathbf{r}),g(E)}\, h_{t,m,g}(E),
    \qquad \int_g h_{t,m,g}(E)\, dE = 1.

The group emission densities :math:`Q_{t,R,g}` (per unit volume, integrated over
angle) are built from the active-averaged scalar flux
:math:`\bar\phi` with the cross sections random ray iterated with, including
the region's density multiplier :math:`d_R`:

.. math::
    :label: rmc-field-emission

    Q_{s,R,g} = d_R \sum_{g'} \nu\Sigma_{s,m}(g' \rightarrow g)\, \bar\phi_{R,g'},
    \qquad
    Q_{f,R,g} = d_R\, \frac{\chi_{m,g}}{k^{MG}} \sum_{g'} \nu\Sigma_{f,m,g'}\, \bar\phi_{R,g'}.

In OpenMC's multigroup format the scattering matrix is already a production
matrix (it includes multiplicity), which is what the scattering source of
random ray uses.

**Lifts.** :math:`h_{t,m,g}` is piecewise constant on a fine energy grid that
contains every interior group edge, normalized within each group over the part
of the group the grid covers. It is tallied in a continuous-energy run of the
same model: ``nu-scatter`` and ``nu-fission`` with an outgoing-energy filter,
per material. Groups with no tallied content use a :math:`1/E` shape.

**Location.** A point belongs to the region keyed by its lowest-level cell
ID, cell instance, and source-region mesh bin, as in random ray. Points in
regions the random ray solve never discovered, and energies outside the fine
grid, have zero field. Energy bin :math:`i` is :math:`[e_i, e_{i+1})`.

**Normalization.** Region volumes :math:`V_R` are estimated once, by locating
points sampled uniformly in each mesh voxel, and the field is scaled so that

.. math::
    :label: rmc-field-norm

    \langle 1, Q^{MG}_f \rangle = \sum_R V_R \sum_g Q_{f,R,g} = \frac{1}{k^{MG}},

which is unit multigroup fission production. A second, independent volume
estimate measures the frozen error this introduces.

The field is a definition: the same lifts, location rule, and volumes are used
wherever it is sampled, evaluated, or integrated. Errors of the random ray
solution change the field, not the correctness of anything built on it.

.. _methods_rmc_field_source:

Sampling the field
------------------

Reference Monte Carlo runs use the field :math:`a_s Q^{MG}_s + a_f Q^{MG}_f` as a fixed
source. Positions are sampled uniformly in a box of volume :math:`V`,
directions isotropically, and energies from a known density :math:`p(E)`, and
each site carries the single-sample weight

.. math::
    :label: rmc-field-source-weight

    w = \frac{4\pi\, V\, \big(a_s Q^{MG}_s + a_f Q^{MG}_f\big)(\mathbf{r}, E, \mathbf{\Omega})}{p(E)}.

Each site is then an unbiased estimate of the whole source, so OpenMC's
per-particle tally normalization gives absolute responses without knowledge
of region volumes.

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
