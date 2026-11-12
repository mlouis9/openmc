.. _methods_rmc:

===============================
Residual Monte Carlo Correction
===============================

.. note::

    This chapter documents an experimental method under development.
    Sections are added as the corresponding code is implemented, so that
    every equation referenced from the source code is described here.

.. _methods_rmc_overview:

--------
Overview
--------

Residual Monte Carlo (RMC) correction solves the
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
production alone is evaluated, which is inexpensive. Structurally this is a two-level Monte Carlo estimator, with the multigroup
model as the coarse level.

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
ID, cell instance, and source-region mesh bin, as in random ray. Random ray
applies at most one mesh to each cell instance, and different instances may
use different meshes or none, so a point is located in two steps: its cell
instance determines the mesh (if any), and the point's bin in that mesh
completes the key (bin 0 without a mesh). Points in regions the random ray
solve never discovered, and energies outside the fine grid, have zero field.
Energy bin :math:`i` is :math:`[e_i, e_{i+1})`.

**Normalization.** Region volumes :math:`V_R` are estimated once, by locating
points sampled uniformly in a known volume that contains the region: every
voxel of the region's mesh, or the ray source box for a region without a mesh
(or outside its mesh). A point is counted only toward regions whose sampling
domain it was drawn in, so each estimate is a binomial fraction of one known
volume. The field is scaled so that

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

.. _methods_rmc_sweep:

-------------------------------------------------
The Multigroup Sweep in the Continuous-Energy Run
-------------------------------------------------

The residual correction compares continuous-energy production with the
multigroup prediction along the same rays, so the multigroup sweep runs in the
same process as the continuous-energy transport. That process has loaded
continuous-energy data and has no MGXS library. OpenMC's random ray solver
needs multigroup data only as a few flat tables per material (total,
nu-scattering, nu-fission, fission, kappa-fission, and :math:`\chi`), the group
edges, and a temperature lookup. In the continuous-energy run these tables are
read from the field file, which stores them exactly as the multigroup run
used them, so the sweep solves the same multigroup problem in both runs. With
the same seed and ray settings on one thread, the two runs give bitwise
identical eigenvalues and fluxes.

.. _methods_rmc_volumes:

Region volumes
--------------

Random ray estimates the volume of each source region from the track length
its rays deposit there. The multigroup flux iterate divides a track-length sum
by that volume, and the residual weights sweep segments crossing region
:math:`R` by

.. math::
    :label: rmc-sweep-weight

    \omega_j = \frac{4\pi\,\ell_j}{\rho\, V_R},

where :math:`\ell_j` is the segment's length and :math:`\rho` the active track
length per unit volume. The sum :math:`\sum_j \omega_j f(\mathbf{\Omega}_j)` is an
unbiased estimate of :math:`\int_{4\pi} f\, d\Omega` averaged over :math:`R` only
when :math:`V_R` is the true volume. When the volumes are known, for example
when the source-region mesh is aligned with the material boundaries or the
boundaries have closed-form intersections with the mesh voxels, random ray's
prescribed volume estimator uses them directly, and no volume error enters
either the sweep or the residual weights.

.. _methods_rmc_tracer:

---------------------------------
The Backward CE Characteristic
---------------------------------

At a destination :math:`\mathbf{r}^\star`, with incident direction
:math:`\mathbf{\Omega}'` and energy :math:`E'`, the angular flux due to the
field source :math:`Q^F` is an integral along the backward path
:math:`\mathbf{r}(s) = \mathbf{r}^\star - s\,\mathbf{\Omega}'`, folded
through reflective boundaries and ended at a vacuum boundary. With the path
split into segments :math:`k` of length :math:`\ell_k`, each inside one cell
instance and one source-region mesh bin, the field and the cross section are
constant on each segment, and

.. math::
    :label: rmc-backward-psi

    \psi(\mathbf{r}^\star, E', \mathbf{\Omega}') = \sum_k e^{-\tau_k(E')}\,
    Q^F_k(E')\,\Lambda\big(\Sigma_{t,k}(E'), \ell_k\big), \qquad
    \Lambda(\Sigma, \ell) = \frac{1 - e^{-\Sigma\ell}}{\Sigma},

where :math:`\tau_k` is the optical depth from :math:`\mathbf{r}^\star` to
the near end of segment :math:`k`, and :math:`\Lambda = \ell` when
:math:`\Sigma = 0`. Reflections change the path's direction but not the
integrand, because the field is isotropic, so the segment list needs only
lengths and what each segment lies in: material, temperature, density
multiplier, and field region. One trace serves every energy :math:`E'` at
which the integral is wanted.

.. _methods_rmc_production:

-----------------------------------
Production at a Common Output State
-----------------------------------

The residual compares continuous-energy production with the field at the
same outgoing state :math:`(E, \mathbf{\Omega})`. Each production channel is
evaluated as the density of what OpenMC's sampler for that channel does, so
that the evaluator and the forward transport are the same physics.

**Fission.** One fission neutron of reaction :math:`x` at incident energy
:math:`E'` is prompt with probability :math:`1 - \nu_d/\nu_t`, or from delayed
group :math:`d` with probability :math:`\nu_d^{(d)}/\nu_t`, and its energy is
drawn from that product's law, repeated until it is below the maximum
transport energy :math:`E_{max}`:

.. math::
    :label: rmc-fission-chi

    \chi_x(E \mid E') = \Big(1 - \frac{\nu_d}{\nu_t}\Big)
    \frac{f_p(E \mid E')}{F_p(E_{max} \mid E')}
    + \sum_d \frac{\nu_d^{(d)}}{\nu_t}
    \frac{f_d(E \mid E')}{F_d(E_{max} \mid E')}, \qquad E < E_{max}.

For a tabulated law (ACE law 4), the sampler chooses the incident table
:math:`\ell` or :math:`\ell+1` with the interpolation weight :math:`r`,
samples it, and maps the result linearly from the table's outgoing range onto
the interpolated range :math:`[E_1, E_K]`. Its density is therefore

.. math::
    :label: rmc-tabular-density

    f(E \mid E') = (1-r)\, f_\ell\big(E_\ell(E)\big)\, \frac{dE_\ell}{dE}
    + r\, f_{\ell+1}\big(E_{\ell+1}(E)\big)\, \frac{dE_{\ell+1}}{dE},

with :math:`E_\ell(E)` the inverse of that map. In OpenMC's fission data
every table spans the same outgoing range, so the map is the identity; the
general form matters for laws whose ranges differ.

**Elastic scattering off a stationary target.** For an elastic collision rate
:math:`s(E')` per unit incident energy, scattering at laboratory cosine
:math:`\mu_0` sends :math:`E'` to :math:`E = \gamma_A(\mu_0) E'`, so the joint
density of the outgoing :math:`(E, \mu_0)` is

.. math::
    :label: rmc-elastic-image

    s\big(E/\gamma_A(\mu_0)\big)\,
    \frac{p_{lab}\big(\mu_0 \mid E/\gamma_A(\mu_0)\big)}{\gamma_A(\mu_0)},

with :math:`p_{lab}` from eq. :eq:`rmc-p-lab`. The azimuth adds a factor
:math:`1/(2\pi)`. The kernel applies where OpenMC uses a stationary target:
:math:`A > 1`, no resonance scattering, and :math:`E'` at or above
``free_gas_threshold`` times :math:`kT`.

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
