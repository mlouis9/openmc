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

.. _methods_rmc_tails:

Ending the backward integral
----------------------------

The sum :eq:`rmc-backward-psi` runs to the end of the path. Three endings are
available:

- **Vacuum.** Follow the path to a vacuum boundary. The integral is exact for
  the path, and adds no noise. A fully reflective problem has no vacuum
  boundary, so this ending is refused there.
- **Roulette.** When the transmission :math:`T = e^{-\tau}` falls below a
  cutoff :math:`T_c`, continue with probability :math:`T/T_c` and set
  :math:`T = T_c`, or stop. The expected remaining tail is unchanged, so the
  estimate is unbiased for any :math:`T_c`; :math:`T_c` trades variance
  against cost. Each energy has its own transmission and its own decisions.
- **Fixed length.** Stop after a distance :math:`L` (a dead length). The
  result is :math:`\psi` without the attenuated contribution beyond
  :math:`L`, which is non-negative, so the approximation is biased low by at
  most :math:`e^{-\tau(L)}\,\max Q^F/\Sigma_t`. It adds no noise and costs a
  bounded amount, which can make it useful where :math:`\tau(L)` is large at
  every energy that matters.

Within a segment the attenuation integral
:math:`\Lambda(\Sigma, \ell) = (1-e^{-\Sigma\ell})/\Sigma` is computed with
``expm1``, and as a series for :math:`\Sigma\ell < 10^{-6}`.

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

.. _methods_rmc_channels:

---------------------
Channels and Routes
---------------------

Every reaction with a neutron product is a *channel* of one nuclide in one
material, with a tag: elastic scattering (without its
S(:math:`\alpha,\beta`) part), the S(:math:`\alpha,\beta`) part of elastic
scattering, every other non-fission reaction with a neutron product
(inelastic levels and continuum, :math:`(n,xn)`), with its yield
:math:`\bar\nu_x`, all tag :math:`s`; and fission, tag :math:`f`. Capture
belongs to no channel but stays in :math:`\Sigma_t`. The production of a
channel is :math:`N_i\,\bar\nu_x(E')\,\sigma_x(E')`; for fission it is
OpenMC's tabulated :math:`\nu\sigma_f`, which is interpolated as a product
on the energy grid and is what ``create_fission_sites`` uses.

Each channel's production at a common output is estimated on one of two
*routes*, decided per incident energy:

- **Evaluator route:** its density at :math:`(E, \mathbf{\Omega})` is
  evaluated (eqs. :eq:`rmc-fission-chi` and :eq:`rmc-elastic-image`).
  Fission always; elastic scattering where OpenMC uses a stationary target
  (:math:`A > 1`, not a resonant scatterer,
  :math:`E' \ge` ``free_gas_threshold`` :math:`kT`, inside the data range).
- **Sampler route:** at an incident energy :math:`E'` drawn with density
  :math:`g(E')`, one sampler-route channel is chosen with probability
  proportional to its production, OpenMC's own routine for it
  (``elastic_scatter`` with a moving target, ``sab_scatter``,
  ``inelastic_scatter``) is applied to a scratch particle, and the
  outgoing state is banked as a positive particle with weight

  .. math::
      :label: rmc-sampler-weight

      \omega_j\, \frac{\Sigma^{samp}_{\mathcal{P}}(E')\,
      \psi(\mathbf{r}, E', \mathbf{\Omega}'_j)}{g(E')},

  where :math:`\Sigma^{samp}_{\mathcal{P}}` is the summed production of the
  sampler-route channels. The yields are in :math:`\Sigma^{samp}_{\mathcal{P}}`,
  so the routine's own weight change and secondaries are discarded.

The routes partition the production exactly: at every energy, evaluator plus
sampler production equals the tag's total, and their sum over tags equals
the material's neutron production.

.. _methods_rmc_mg_control:

------------------------
The Multigroup Control
------------------------

Along the same backward path as the CE integral, the multigroup angular flux
of each group follows from the MG total cross sections of the segments and
the field's group emission:

.. math::
    :label: rmc-mg-psi

    \psi^{MG}_{g'} = \sum_k e^{-\tau_{k,g'}}\, \frac{Q_{k,g'}}{4\pi}\,
    \Lambda\big(\Sigma^{MG}_{t,k,g'}, \ell_k\big),

with each group carrying its own transmission and tail decisions. The MG
production at the common output, for the destination's material :math:`m`
and region :math:`R`, is

.. math::
    :label: rmc-mg-production

    q^{MG}_t(E) = \frac{h_{t,R}(E)}{4\pi} \sum_{g'} K^t_m\big(g' \to g(E)\big)\,
    \psi^{MG}_{g'}, \qquad
    K^s = \nu\Sigma_s, \quad K^f = \chi\,\nu\Sigma_f,

the same production the field was computed with, so that at
:math:`a = (1,1)` and :math:`k = k_{MG}` a converged field reproduces itself
on average. For fission there is a second form that shares the CE estimate's
incident energy: with :math:`E'` drawn with density :math:`g(E')`,

.. math::
    :label: rmc-mg-same-energy

    q^{MG}_f(E) \approx \frac{h_{f,R}(E)}{4\pi}\, \chi_{g(E)}\,
    \nu\Sigma_{f,g(E')}\, \psi^{MG}_{g(E')}\, \frac{h^{\phi}_R(E')}{g(E')},

whose mean over :math:`E'` is eq. :eq:`rmc-mg-production`, with
:math:`h^\phi` the within-group flux shape. The CE and MG fission estimates
of one direction then vary together with :math:`E'`, and their difference is
small where the CE and MG cross sections agree.

.. _methods_rmc_residual:

-----------------
Residual Samples
-----------------

The residual of the field for tag :math:`t \in \{s, f\}` is the CE production
of that tag from the field's flux, minus the field candidate:

.. math::
    :label: rmc-residual

    R_t(\mathbf{r}, E, \mathbf{\Omega}) = \frac{1}{k^{[t=f]}}\,
    C_t(\mathbf{r}, E, \mathbf{\Omega}) - F_t(\mathbf{r}, E, \mathbf{\Omega}),
    \qquad
    C_t = \int_{4\pi} d\mathbf{\Omega}' \int dE'\;
    K_t(E', \mathbf{\Omega}' \to E, \mathbf{\Omega})\,
    \psi(\mathbf{r}, E', \mathbf{\Omega}'),

where :math:`K_s` is nonfission production, :math:`K_f` is fission
production without the :math:`1/k`, and :math:`\psi` is
eq. :eq:`rmc-backward-psi` for the field
:math:`a_s Q^{MG}_s + a_f Q^{MG}_f`. Because :math:`\psi` is linear in the
field, :math:`C_t = a_s C_{t,s} + a_f C_{t,f}`, with :math:`C_{t,s}` and
:math:`C_{t,f}` the production from :math:`\psi` of :math:`Q^{MG}_s` and of
:math:`Q^{MG}_f` alone. The frozen field candidate is
:math:`F_t = b_t\,Q^{MG}_t`; the swept candidates are below. The amplitudes
:math:`a, b` and the eigenvalue :math:`k` are scalars that change between
generations.

.. _methods_rmc_residual_proposals:

Proposals
---------

A sample draws a destination, an outgoing state, and, on each incident
direction, the incident energies it needs.

- **Destination.** Uniform in the field's ray source box (density
  :math:`1/V_{box}`), or a region :math:`R` drawn with probability
  :math:`\pi_R` and then a point uniform in it (density
  :math:`\pi_R/V_R`). :math:`\pi_R` is proportional to the region volume, to
  the volume of fissionable regions only, or to given weights; a point
  uniform in a region is drawn uniformly in its mesh element (or the box)
  and kept if it is located in the region.
- **Outgoing state.** An isotropic direction, and an energy from the mixture

  .. math::
      :label: rmc-residual-outgoing

      p^t(E \mid R) = (1-\beta)\,\frac{Q_{t,R,g(E)}\,h_{t,R}(E)}
      {\sum_g Q_{t,R,g}} + \beta\,\frac{1}{E\ln(E_{max}/E_{min})},

  the region's emission spectrum and a log-uniform component. The residual
  contains :math:`-F_t` and the CE emission, so :math:`\beta > 0` and a range
  covering both lifts and CE emission keep the proposal positive wherever the
  residual can be non-zero.
- **Incident energy** (fission evaluator and sampler route). Log-uniform, or
  a mixture led by the direction's own MG flux: a group :math:`g'` with
  probability proportional to :math:`\nu\Sigma_{f,g'}\psi^{MG}_{g'}`
  (fission) or :math:`\psi^{MG}_{g'}` (sampler route), :math:`E'` from the
  flux shape :math:`h^\phi_R` in it, and a log-uniform component of weight
  :math:`\beta'`.

No proposal uses :math:`a`, :math:`b`, or :math:`k` (the flux-led ones use
:math:`\psi^{MG}` of :math:`Q^{MG}_s + Q^{MG}_f`), which keeps the weights
deferrable. Every proposal returns its value with the density it was drawn
from, and weights use only that density.

.. _methods_rmc_residual_family:

Directions and the estimator family
-----------------------------------

The direction integral in :math:`C_t` uses one of two direction sources.

- **Isotropic.** :math:`J` spherical Fibonacci points (cosines stratified on
  :math:`[-1, 1]`, azimuths at the golden angle) under a uniformly random
  rotation, with :math:`\omega_j = 4\pi/J`. Each rotated point is marginally
  uniform on the sphere, so the estimate is unbiased for any integrand.
- **Sweep.** One random ray transport sweep from :math:`Q^{MG}_s` and one
  from :math:`Q^{MG}_f`, with the same rays, give the segments crossing each
  region, with direction :math:`\mathbf{\Omega}'_j`, chord length
  :math:`\ell_j`, and chord-averaged MG angular fluxes, and weights
  :math:`\omega_j` of eq. :eq:`rmc-sweep-weight` with the field's region
  volumes. The CE
  integral uses every segment of the destination's region, or :math:`J` of
  them drawn with probability proportional to :math:`\omega_j`, each with
  weight :math:`\sum_j\omega_j/J`. The CE trace follows a new characteristic
  parallel to the segment, from the destination.

On each direction the MG control :math:`q^{MG}_j` is evaluated on the same
footing: from the MG flux on the same backward path (isotropic directions),
or from the segment's chord-averaged flux (sweep directions). With control
coefficient :math:`c` and mixing :math:`\lambda`,

.. math::
    :label: rmc-residual-family

    \Delta_t = \sum_j \omega_j \Big[\hat q^{CE}_j - c\, q^{MG}_j\Big]
    + c\,\hat R_t - b_t\Big[(1-\lambda)\, Q^{MG}_t + \lambda\, \hat Q_t\Big],

where :math:`\hat R_t` restores the control's mean: for isotropic directions,
:math:`L` further MG-only directions with the same MG estimator,
:math:`\hat R_t = \sum_{l=1}^{L} (4\pi/L)\, q^{MG}_l`; for sweep directions,
the sweep's region emission

.. math::
    :label: rmc-sweep-emission

    \hat Q_t(E) = \sum_{j \in R} \omega_j\, q^{MG}_j(E),

the multigroup update of the field from this sweep (with :math:`1/k` for
fission). :math:`\lambda` is used only with sweep directions. Every member is
unbiased for :math:`R_t` with :math:`F_t = b_t[(1-\lambda)Q^{MG}_t +
\lambda\hat Q_t]`: the control term and its restored mean have the same
expectation, so :math:`c` and :math:`\lambda` move noise between the field
and the residual but never the mean. Named members:

- *frozen, uncontrolled*: :math:`c = 0`, :math:`\lambda = 0`;
- *frozen, same-path control*: isotropic directions, :math:`c = b`,
  :math:`L \ge 1`;
- *frozen, sweep control*: sweep directions, :math:`c = b`,
  :math:`\lambda = 0`;
- *tandem*: sweep directions, :math:`c = b`, :math:`\lambda = 1`. With
  every segment of the region as a direction, :math:`\sum_j\omega_j
  q^{MG}_j = \hat Q_t` exactly and the bracket vanishes record by record:
  the sweep's noise stays in the field and only the CE minus MG difference
  per direction remains;
- *partial*: :math:`c = \lambda b`.

**Records with deferred weights.** Each sample stores its state, the density
:math:`\varrho` of its whole draw, and per source tag :math:`y \in \{s, f\}`
the CE term :math:`C_y`, the control on its directions :math:`M_y`, and the
restored mean :math:`R_y` (each from :math:`\psi` of :math:`Q^{MG}_y` alone),
and :math:`U = Q^{MG}_t` at its state. The weight, for :math:`N` samples of
its tag, is

.. math::
    :label: rmc-residual-weight

    w = \frac{1}{N\varrho}\Big\{\frac{\sum_y a_y\,(C_y - c\,M_y + c\,R_y)}
    {k^{[t=f]}} - b_t\Big[(1-\lambda)\,U +
    \lambda\,\frac{\sum_y a_y\,R_y}{k^{[t=f]}}\Big]\Big\},

and a sampler-route particle (density :math:`\varrho`, the destination's)
has :math:`w = \sum_y a_y C_y / (k^{[t=f]} N \varrho)` with :math:`C_y` from
eq. :eq:`rmc-sampler-weight`. The records do not depend on :math:`a, b, k,
c, \lambda`, so one set serves every value of them, and the members of the
family can be compared on the same samples.

.. _methods_rmc_residual_ce:

CE production on one direction
------------------------------

**Elastic production** (tag :math:`s`, evaluator route). For each nuclide
:math:`i` with :math:`A_i > 1`, the incident energy that scatters into
:math:`E` at :math:`\mu_0 = \mathbf{\Omega}'\cdot\mathbf{\Omega}` is
:math:`E'_i = E/\gamma_{A_i}(\mu_0)`, so the energy integral collapses
(eq. :eq:`rmc-elastic-image`):

.. math::
    :label: rmc-residual-elastic

    \int dE'\, K^{el}\,\psi = \sum_i N_i\, \sigma_{el,i}(E'_i)\,
    \frac{p_{lab,i}(\mu_0 \mid E'_i)}{2\pi\,\gamma_{A_i}(\mu_0)}\,
    \psi(\mathbf{r}, E'_i, \mathbf{\Omega}').

The :math:`E'_i` of all nuclides, and the sampler route's :math:`E'`, are
evaluated on one backward path.

**Fission production** (tag :math:`f`). Emission is isotropic, and one
incident energy :math:`E'` is drawn with density :math:`g(E')`:

.. math::
    :label: rmc-residual-fission

    \int dE'\, K^{f}\,\psi \approx \frac{1}{4\pi\, g(E')}
    \sum_i N_i\, \nu\sigma_{f,i}(E') \sum_x \frac{\sigma_{x,i}(E')}
    {\sigma_{f,i}(E')}\, \chi_{x,i}(E \mid E')\,
    \psi(\mathbf{r}, E', \mathbf{\Omega}'),

with :math:`\chi_{x,i}` from eq. :eq:`rmc-fission-chi` and :math:`x` the
partial fission reactions where a nuclide has them (otherwise one term).

**Means in an infinite medium.** With a flat field in a homogeneous,
reflective medium, :math:`\psi = Q^{MG}(E')/\Sigma_t(E')`, and the means of
the record terms reduce to energy integrals:

.. math::
    :label: rmc-residual-means

    \mathrm{E}\Big[\frac{C_{s,u}}{\varrho}\Big]
    = 4\pi V \int \Sigma_{el}\, \frac{Q^{MG}_u}{\Sigma_t}\, dE', \quad
    \mathrm{E}\Big[\frac{C_{f,u}}{\varrho}\Big]
    = 4\pi V \int \nu\Sigma_f\, \frac{Q^{MG}_u}{\Sigma_t}\, dE', \quad
    \mathrm{E}\Big[\frac{U}{\varrho}\Big] = 4\pi V \int Q^{MG}_t\, dE,

for :math:`u \in \{s, f\}`, and the sampler-route particles of a sample sum
on average to :math:`4\pi V\int\Sigma^{samp}_{\mathcal{P}}\,Q^{MG}_s/\Sigma_t\,dE'`,
provided the proposal ranges cover the integrands.

.. _methods_rmc_first_collision:

The first-collision reference
-----------------------------

The residual's high term is the first-collision neutron production of the
field source. It can be computed independently with OpenMC's own transport:
the field (with amplitudes :math:`a`) is the source of an analog run, an
energy cutoff above every source energy ends each history at its first
collision, and ``nu-scatter`` and ``nu-fission`` are tallied with
``CollisionFilter([1])`` by material and outgoing energy (the fission part
divided by :math:`k`). The run is one batch of an eigenvalue calculation, so
that fission neutrons are banked rather than transported: OpenMC starts
fission secondaries with no collisions, so in a fixed-source run their own
first collisions would be counted too. The low term is the field (or a
sweep's emission) integrated exactly over the same bins.

.. _methods_rmc_flights:

---------------------------
Forward Correction Flights
---------------------------

The bank of signed sites :math:`(\mathbf{r}_i, \mathbf{\Omega}_i, E_i, w_i)`
is a source whose next-generation production is estimated by analog flights.
Each of :math:`N` flights chooses a parent :math:`i` with probability
:math:`|w_i|/W_{abs}`, :math:`W_{abs} = \sum_i |w_i|`, flies it to its next
collision with OpenMC's geometry and cross sections (or out through a vacuum
boundary), and returns at most one child:

- a nuclide :math:`m` is chosen with probability
  :math:`\Sigma_{t,m}/\Sigma_t`;
- with probability :math:`\sigma_{s,m}/\sigma_{t,m}` the neutron scatters
  with OpenMC's ``scatter()``; the child, tag :math:`s`, has weight factor
  equal to the realized yield (the particle's weight after the call times
  one plus the integer-yield copies it banked, which share its outgoing
  state);
- otherwise, with probability :math:`\sigma_{f,m}/\sigma_{a,m}`, it
  fissions; the child, tag :math:`f`, comes from
  ``sample_fission_neutron()`` with weight factor
  :math:`\nu_m(E_i)/k`, with :math:`\nu_m = \nu\sigma_{f,m}/\sigma_{f,m}`
  from OpenMC's tabulated nu-fission;
- otherwise the neutron is captured and leaves no child.

The child's weight is :math:`\operatorname{sgn}(w_i)\, W_{abs}/N` times its
factor, so the expected children of parent :math:`i` are

.. math::
    :label: rmc-flight-expectation

    w_i\, \frac{\Sigma^{yield}_s(E_i)}{\Sigma_t(E_i)}\, p_s(\cdot \mid E_i), \qquad
    w_i\, \frac{\nu\Sigma_f(E_i)}{k\,\Sigma_t(E_i)}\, p_f(\cdot \mid E_i)

at the collision density along the parent's ray: the first-collision
production of the bank. No OpenMC tally, roulette, or fission bank is
touched. A track-length tally of the flights,
:math:`\operatorname{sgn}(w_i) W_{abs}/N` times the path length, estimates
the bank's uncollided flux.

.. _methods_rmc_iteration:

----------------
The Iteration
----------------

The state of generation :math:`n` is the eigenvalue :math:`k_n`, the field
amplitudes :math:`a_n = (a_s, a_f)` of the frozen field
:math:`a_s Q^{MG}_s + a_f Q^{MG}_f`, and the signed bank :math:`\delta Q_n`
of what the field does not carry. A generation:

1. **Amplitudes.** The field candidate is :math:`F = b\,Q^{MG}`, with
   :math:`b` chosen from earlier generations only: fixed (default
   :math:`b = (1,1)`), or per-tag mass matching,
   :math:`b_t \leftarrow b_t + d\,(a_t + \langle 1, \delta Q_t\rangle /
   \langle 1, Q^{MG}_t\rangle - b_t)` with damping :math:`d`.
2. **Candidate.** The residual of the field (records weighted with
   :math:`a_n, b, k_n`, eq. :eq:`rmc-residual-weight`) plus the children of
   :math:`\delta Q_n`'s forward flights:
   :math:`\delta Q^* = \hat R_n + \hat{\mathcal{B}}_n \delta Q_n`.
3. **Production and eigenvalue.**

   .. math::
       :label: rmc-iteration-f

       f = k_n\big(b_f \langle 1, Q^{MG}_f\rangle + \langle 1,
       \delta Q^*_f\rangle\big), \qquad k_{n+1} = k_n f.

   A non-positive :math:`f` stops the run; it is never clipped.
4. **Normalization.** Tag :math:`s` is scaled by :math:`1/f` and tag
   :math:`f` by :math:`k_n/(k_{n+1} f)`, in the bank and in the field:
   :math:`a_{n+1} = (b_s/f,\; b_f k_n/(k_{n+1} f))`.
5. **Resampling.** The bank is compressed to at most :math:`N_B` sites:
   :math:`N_B` sites chosen systematically by :math:`|w|`, each with weight
   :math:`\operatorname{sgn}(w)\,W_{abs}/N_B`, which keeps the bank's
   expectation.

With a frozen field only the two amplitudes change between generations.
Without cancellation the bank's absolute weight grows roughly linearly with
the generation, since every generation adds the residual's absolute weight
and flights in a multiplying medium carry most of it on.

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
