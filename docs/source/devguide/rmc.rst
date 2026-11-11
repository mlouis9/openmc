.. _devguide_rmc:

=========================================
Multilevel RMC: Development Log
=========================================

This page records the development of multilevel residual Monte Carlo (RMC),
the experimental method described in :ref:`methods_rmc`. It is a working
document for developers: for each stage it lists what was added, where, how
it was verified, and what is known to be missing. Entries are grouped by
stage and, within a stage, in the order the work was committed. Commit
subjects are quoted so that an entry can be found with ``git log --grep``.

The stages follow one rule: every layer is checked against an independent
reference before the next layer is built on it, so that a disagreement in the
eigenvalue at the end can be traced to one operator.

.. contents::
    :local:
    :depth: 2

Conventions
-----------

- **Code.** C++ lives in ``include/openmc/rmc/`` and ``src/rmc/`` (namespace
  ``openmc::rmc``), with C API functions prefixed ``openmc_rmc_`` and ctypes
  bindings in ``openmc/lib/rmc.py``. Changes to the random ray solver are in
  ``src/random_ray/``.
- **Drivers.** Python that builds problems and files is in ``examples/rmc/``.
  It is plain modules, imported by the tests through ``sys.path``.
- **Tests.** Catch2 tests of pure math are ``tests/cpp_unit_tests/test_rmc_*``.
  Tests that need nuclear data are in ``tests/rmc/`` and run through
  ``openmc.lib`` or OpenMC subprocesses. Settings round trips are in
  ``tests/unit_tests/test_rmc_settings.py``. Slow tests carry
  ``@pytest.mark.slow``.
- **Verification problems.** Defined in ``examples/rmc/problems.py``:

  - *infinite medium*: the graphite pincell's materials homogenized, in a
    reflective box;
  - *graphite pincell*: a UO\ :sub:`2` pin in graphite (without
    S(:math:`\alpha,\beta`)), reflective on every face, 2 cm thick.

  Both use matched physics: ``free_gas_threshold = 1e-6`` (stationary
  targets for every nuclide with :math:`A>1`), ``ptables = False``, and
  temperatures by nearest point at 294 K.

Running the tests::

    pytest tests/unit_tests/test_rmc_settings.py
    pytest tests/rmc                    # needs OPENMC_CROSS_SECTIONS
    ctest --test-dir build -R rmc       # Catch2 tests

The tests in ``tests/rmc/`` share a session fixture that runs random ray and
the lift tallies on the graphite pincell once (a few minutes on two cores).

-----------------------------------------
Stage 0: Toolchain and kinematics
-----------------------------------------

**Commit:** the initial ``rmc`` work (kinematics module and method
documentation).

Added
    - ``rmc/kinematics.{h,cpp}``: stationary-target elastic relations as
      functions of the laboratory cosine :math:`\mu_0`: the energy ratio
      :math:`\gamma_A`, the center-of-mass cosine :math:`\mu_{cm}(\mu_0)`, and
      its Jacobian :math:`d\mu_{cm}/d\mu_0`, written without cancellation near
      :math:`\mu_0=-1`.
    - ``docs/source/methods/rmc.rst``: overview and the kinematics sections
      cited by the source.

Verified by ``tests/cpp_unit_tests/test_rmc_kinematics.cpp``
    - The inverse cosine relation inverts the forward one.
    - The Jacobian is the reciprocal of the forward derivative (an earlier
      finite-difference test lost accuracy near :math:`A=1`,
      :math:`\mu_0=-1`, and was replaced).
    - :math:`\gamma_A` endpoints, :math:`\gamma_A(1)=1` and
      :math:`\gamma_A(-1)=\alpha`.
    - Isotropic center-of-mass scattering gives a flat laboratory energy
      distribution.

-----------------------------------------
Stage 1: The frozen field
-----------------------------------------

**Commits:** "rmc: frozen field export, C++ field, C API, bindings, plugin,
tests (stubs for exercises)" and "rmc: emission notation Q_s, Q_f".

Added
    - **Field export** (``FlatSourceDomain::export_rmc_field``, setting
      ``random_ray['rmc_field_file']``). After an eigenvalue random ray solve,
      writes per source region its key (cell ID, instance, mesh ID, mesh
      bin), material, density multiplier, volume estimate, active-averaged
      flux, and the multigroup emission :math:`Q^{MG}_s`, :math:`Q^{MG}_f` by
      group; per material the MGXS used. Arrays are flat and groups ascend in
      energy. Refused for fixed-source, adjoint, multi-rank, and
      multi-temperature runs.
    - **Lifts** (``examples/rmc/field.py``, ``build_lifts``): within-group
      emission shapes per material from CE tallies on a fine energy grid.
    - **C++ field** (``rmc/field.{h,cpp}``): ``FieldSource`` interface and
      ``FrozenField``, which loads the field, lift, and volume files, locates
      a point's source region, and evaluates
      :math:`Q^{MG}_t(\mathbf r,E)/(4\pi)`.
    - **C API and bindings** (``rmc/capi.{h,cpp}``, ``openmc/lib/rmc.py``):
      load, free, locate, evaluate, mass, and region volumes by point
      location.
    - **Field source plugin** (``examples/rmc/field_source/``): a compiled
      source that samples the frozen field, for reference CE runs.
    - **Python reference implementation** (``examples/rmc/field.py``,
      ``ReferenceField``): the same locate, evaluate, and volume operations,
      used to test the C++ ones.
    - **Pipeline** (``examples/rmc/pipeline.py``): random ray, lifts, and
      volumes in order; ``ce_session`` initializes a CE model in-process and
      adds the source-region mesh it lacks.

Verified by ``tests/rmc/test_frozen_field.py``
    - IDs survive conversion to multigroup.
    - Exported fission production matches the random ray tally; exported
      emission matches an independent numpy recomputation from ``mgxs.h5``
      to :math:`10^{-12}`.
    - Lifts integrate to one over each group.
    - The Python locator agrees with OpenMC's cell lookup; Python volumes
      are consistent between seeds.
    - C++ masses, locate, evaluate, and volumes agree with the Python
      reference (the last three are the exercises below).

Verified by ``tests/rmc/test_field_source.py``
    - The field source plugin reproduces the field's mass.

Exercises left as stubs
    ``FrozenField::locate``, ``FrozenField::evaluate``,
    ``openmc_rmc_field_region_volumes``, and ``FieldSampler::sample``. The
    four tests that exercise them fail until they are written.

Known limitations
    One temperature per material; one source-region mesh; one MPI rank.

-----------------------------------------
Stage 2: Random ray in the CE process
-----------------------------------------

Stage 2 makes OpenMC's random ray solver usable inside the CE process as the
multigroup sweep, and adds the backward CE characteristic. Entries are added
as the work is committed.

.. _devguide_rmc_prescribed_volumes:

Prescribed source-region volumes
--------------------------------

**Commit:** "random ray: prescribed source-region volumes".

Why
    The residual's sweep quadrature weights
    :math:`\omega_j=4\pi\ell_j/(\rho V_R)` and the field masses need true
    region volumes. Random ray estimates volumes from track lengths, which
    adds a volume error to every flux iterate. When the source-region mesh is
    aligned with the residual voxels, or the geometry is simple, the volumes
    are known exactly and should be used.

Added
    - Volume estimator ``'prescribed'`` and setting
      ``random_ray['source_region_volumes']`` (C++ and Python, with
      documentation in the user's guide and the settings file format). The
      file holds ``cell_id``, ``instance``, ``mesh_bin``, and ``volume``
      [cm\ :sup:`3`].
    - ``FlatSourceDomain::load_prescribed_volumes()`` reads and validates the
      file once the meshes are applied (unknown cells, cells not filled with
      a material, out-of-range instances and mesh bins, negative or
      non-finite volumes, and duplicates are fatal).
    - Discovered source regions take their prescribed volume in
      ``finalize_discovered_source_regions()``. A region that rays reach but
      the file omits, or whose prescribed volume is zero, is fatal: there is
      no silent fallback to an estimate.
    - ``normalize_scalar_flux_and_volumes()`` sets the volume used by the flux
      update, the eigenvalue, and the tallies to the prescribed volume
      divided by the ray source box volume. Track-length volumes are still
      accumulated, to mark hits and compute centroids.
    - ``examples/rmc/volumes.py``: the exact area of a disk intersected with
      a rectangle, the graphite pincell's source-region volumes under a
      regular mesh, and the file writer.
    - ``examples/rmc/mg.py``: ``run_random_ray`` accepts an existing MGXS
      library and extra random ray settings.
    - ``openmc_finalize_random_ray()`` now also clears ``rmc_field_file_``
      (a Stage 1 omission: the export path survived ``openmc.lib.finalize()``)
      and the volume file path.

Verified by ``tests/rmc/test_prescribed_volumes.py``
    - Closed forms: exact limits (full disk, quadrant, rectangle inside the
      disk, half-plane), and agreement with hit-or-miss sampling for random
      rectangles.
    - The pincell's volumes partition the box, and the fuel volumes sum to
      the cylinder, to :math:`10^{-13}`.
    - The closed forms agree with Stage 1's point-location volumes for both
      seeds, region by region, within binomial statistics (maximum
      :math:`|z|<5` and a :math:`\chi^2` bound). This cross-checks the
      closed forms, the mesh bin numbering, and OpenMC's cell lookup.
    - The volumes random ray exports equal the prescribed volumes to
      :math:`10^{-12}`: the whole path from file to flux update is exact.
    - :math:`k` agrees with the estimated-volume run of the same MGXS and
      settings, within four block-mean standard errors.
    - A volume file missing the fuel region at the pin centre stops the run
      with an error naming the region.
    - Settings round trip and type check
      (``tests/unit_tests/test_rmc_settings.py``).

Known limitations
    Flat sources only (the linear source domain overrides the volume
    normalization); one MPI rank (load balancing moves source regions
    between ranks).

.. _devguide_rmc_mg_data:

Multigroup data provider
------------------------

**Commit:** "random ray: read multigroup data through RandomRayMGData".

Why
    OpenMC loads either continuous-energy or multigroup data in a process,
    never both. Random ray read the MGXS library (``data::mg``) directly, so
    it could only run in a multigroup process.

Added
    - ``random_ray/mg_data.{h,cpp}``: ``RandomRayMGData``, the flat tables
      random ray reads (group count, edges, group-average energies; per
      material and temperature :math:`\Sigma_t`, :math:`\nu\Sigma_f`,
      :math:`\Sigma_f`, :math:`\chi`, :math:`\kappa\Sigma_f`, and the
      nu-scattering matrix; the temperature points), with a global instance
      ``data::random_ray_mg``. ``fill_from_mgxs()`` builds them from
      ``data::mg`` exactly as ``flatten_xs()`` did.
    - Every random ray read of ``data::mg`` now reads the provider: the group
      count in the ``FlatSourceDomain``, ``RandomRay``,
      ``RandomRaySimulation``, ``RayBank``, and ``DecompositionMap``
      constructors; ``flatten_xs()`` (now a copy, since adjoint solves modify
      the domain's tables in place); the temperature index of a discovered
      region; group-average energies for tallies; the group of a discrete
      external source energy; the MGXS checks of
      ``validate_random_ray_inputs()``; and the group edges of the field
      export.
    - ``Mgxs::kT(t)``, a read accessor for a temperature point.

Verified
    The random ray regression tests that run with the available nuclear data
    (53 cases) produce bitwise identical ``results_true.dat`` files before
    and after the change. Procedure, for each build: run
    ``pytest tests/regression_tests -k random_ray --update --exe <openmc>``,
    copy every ``results_true.dat`` aside, and restore the committed files
    with ``git checkout -- tests/regression_tests``; then compare the two
    copies with ``diff -r``. Cases that need nuclear data absent from the
    test library (for example Zr90 and N14) are skipped by this procedure
    and should be checked with a full library.

.. _devguide_rmc_ce_random_ray:

Random ray in the continuous-energy process
-------------------------------------------

**Commit:** "rmc: run random ray in the continuous-energy process".

Added
    - **Field export version 2.** ``materials/`` also holds ``in_model``,
      ``sigma_f``, and ``kappa_fission``, so the field file carries every
      table random ray uses.
    - ``rmc/mg_tables.{h,cpp}``: ``rmc::load_mg_tables()`` fills
      ``data::random_ray_mg`` from a field file, matching materials by ID and
      reversing the groups, copying every value exactly.
    - **Settings.** An ``<rmc>`` element (``Settings.rmc`` in Python, keys
      ``field_file``, ``lift_file``, ``volume_file``) declares a multilevel
      RMC model. In a continuous-energy model with ``<rmc>``, the
      ``<random_ray>`` element is read instead of rejected, and the solver
      type stays Monte Carlo.
    - ``openmc_run_random_ray()`` fills the provider from the field file in a
      continuous-energy process and from the MGXS library otherwise.
    - ``openmc_rmc_run_random_ray()`` (``src/rmc/random_ray_capi.cpp``) and
      ``openmc.lib.rmc.run_random_ray()``: run a whole random ray solve in an
      initialized continuous-energy session. The solver type is set to random
      ray for the solve (OpenMC's batch, eigenvalue, and tally code branch on
      it) and restored afterwards, together with ``source_write``, which
      random ray turns off.
    - ``examples/rmc/pipeline.py``: ``random_ray_ce_model()`` and
      ``run_random_ray_in_ce()`` copy the random ray settings, ray count,
      batches, and seed from the multigroup model.
    - ``docs/source/methods/rmc.rst``: sections on the multigroup sweep in
      the continuous-energy run and on region volumes.

Verified by ``tests/rmc/test_random_ray_ce.py`` (gate G2.5)
    - The tables random ray uses in the continuous-energy process equal the
      multigroup run's entry by entry.
    - On one thread, :math:`k`, every region flux, emission, and volume
      estimate are bitwise equal to the multigroup run's.
    - On two threads they agree to :math:`10^{-10}` relative; OpenMP
      reductions sum in a different order from run to run, which changes
      round-off only.
    - A continuous-energy Monte Carlo run in the same session afterwards
      gives a finite, plausible :math:`k`.
    - Settings round trip and validation of ``Settings.rmc``.

Known limitations
    One temperature per material (as the field export). No per-generation
    entry point from Python yet.
