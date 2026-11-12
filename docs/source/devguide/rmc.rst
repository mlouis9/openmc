.. _devguide_rmc:

===============================================
Residual Monte Carlo (RMC): Development Log
===============================================

This page records the development of residual Monte Carlo (RMC),
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
tests (stubs for exercises)", "rmc: emission notation Q_s, Q_f", and "rmc:
reference implementations of the Stage 1 exercises". (In the main
repository, Stage 1 may be a single commit containing all three.)

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
      reference (locate, evaluate, and volumes were the exercises below).

Verified by ``tests/rmc/test_field_source.py``
    - The field source plugin reproduces the field's mass.

Exercises
    ``FrozenField::locate``, ``FrozenField::evaluate``,
    ``openmc_rmc_field_region_volumes``, and ``FieldSampler::sample`` were
    first committed as stubs, as learning exercises. ``locate`` and
    ``evaluate`` were written independently, then all four were replaced by
    the reference implementations, with which every Stage 1 test passes
    (``test_field_source.py`` needs the plugin in
    ``examples/rmc/field_source`` built against an installed OpenMC).

Known limitations
    One temperature per material; one MPI rank. (One source-region mesh per
    field was lifted in Stage 2; see :ref:`devguide_rmc_multiple_meshes`.)

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
      ``field_file``, ``lift_file``, ``volume_file``) declares an RMC
      model. In a continuous-energy model with ``<rmc>``, the
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
    entry point from Python yet; see the next entry.

.. _devguide_rmc_sweep_split:

Per-iteration sweep
-------------------

**Commit:** "random ray: split the transport sweep out of simulate()".

Added
    ``RandomRaySimulation::sweep()``: one random ray iteration from the
    current region sources (reset the iteration tallies, transport every
    ray, add discovered regions, normalize, form the new flux iterate).
    ``simulate()`` calls it between the source update and the eigenvalue
    update. RMC will call it after setting the sources itself.

Verified
    The random ray regression tests produce bitwise identical results
    before and after the split.

Not yet done
    Setting region sources from an RMC field before ``sweep()``; see
    :ref:`devguide_rmc_segment_records` for the batch-by-batch entry point.

.. _devguide_rmc_multiple_meshes:

Several source-region meshes in the frozen field
------------------------------------------------

**Commit:** "rmc: frozen fields with several source-region meshes".

Why
    Random ray applies at most one mesh to each cell instance, but different
    instances may use different meshes or none (for example, a fine mesh on
    the fuel and a coarse one on the moderator). The Stage 1 field refused
    any field with more than one mesh.

Added
    - **Field export version 3** adds the ray source box
      (``domain_lower_left``, ``domain_upper_right``), which contains every
      source region. ``FrozenField`` refuses older files with a message to
      rerun the random ray solve.
    - ``FrozenField`` keeps the mesh of each cell instance
      (``mesh_of_base_``, built from the per-region ``mesh_id`` and checked
      for consistency) instead of one mesh index. ``locate`` mirrors random
      ray's lookup: cell instance, then its mesh, then the bin. New
      accessors: ``mesh_indices()``, ``region_mesh(r)``, and the box.
    - ``openmc_rmc_field_region_volumes`` takes ``n_box`` and returns hit
      counts. Regions on a mesh are sampled voxel by voxel in their own mesh;
      regions without a mesh, or outside it, in the ray source box. A point
      is counted only toward regions of the domain it was drawn in.
      ``openmc.lib.rmc.field_region_volumes(n_per_voxel, seed, n_box=0)``
      now returns ``(volumes, counts)``.
    - The Python reference (``ReferenceField``, ``region_volumes``), the
      pipeline (``ce_session``, ``build_volumes``), and ``mg.run_random_ray``
      accept one mesh or a list of ``(mesh, domains)`` pairs
      (``mg.source_region_meshes()``, ``mg.meshes_of()``). The volume file
      (version 2) records ``n_box``.
    - ``problems.graphite_pincell_split_meshes()``: the graphite pincell with
      the moderator split into an unmeshed ring and an outer region, a fine
      mesh on the fuel and a coarse mesh on the outer graphite.

Verified by ``tests/rmc/test_multiple_meshes.py``
    - The export assigns each cell's regions to its own mesh (the ring is a
      single region with no mesh, bin 0) and records the box.
    - C++ ``locate`` agrees exactly with the Python reference at 20,000
      points, and every region larger than the sampling resolution is found.
    - C++ and Python volumes agree with the closed-form volumes region by
      region (disk-rectangle intersections for the fuel and outer regions,
      an annulus for the ring), within binomial statistics; sampling without
      box points is refused when a region needs them.
    - The single-mesh tests of ``test_frozen_field.py`` and
      ``test_field_source.py`` pass unchanged in substance.

.. _devguide_rmc_segment_records:

Random ray batch by batch, and segment records
----------------------------------------------

**Commit:** "rmc: random ray batch by batch, and segment records".

Why
    The residual estimator uses the sweep's segments crossing a destination's
    region as a quadrature of incident directions, with weights
    :math:`\omega_j = 4\pi\ell_j/(\rho V_R)` (:ref:`methods_rmc_volumes`),
    so the CE process must run random ray one batch at a time and read those
    segments.

Added
    - ``RandomRaySimulation::power_iteration()``: one batch of the power
      iteration, the loop body of ``simulate()``.
    - ``resolve_random_ray_volume_estimator()`` and
      ``initialize_random_ray_solve()``, split out of
      ``openmc_run_random_ray()`` so the batch-by-batch path initializes the
      same way.
    - ``FlatSourceDomain::source_region_keys()``, split out of the field
      export, and accessors ``n_groups()`` and ``simulation_volume()``.
    - **Segment records** (``random_ray/segment_records.h``): per thread,
      for flagged regions, each active segment's region, ray ID, position
      along the ray, direction, length, and chord-averaged angular flux
      :math:`\bar\psi_g = q_g + \Delta\psi_g/(\Sigma_{t,g}\ell)`, in
      random ray's units (:math:`4\pi\psi`). The hook is in
      ``RandomRay::attenuate_flux_inner()`` for flat, non-void regions. It
      costs one branch when recording is off. ``gather_segment_records()``
      merges the threads in (ray, position) order.
    - C API ``openmc_rmc_sweep_{init, iterate, set_recording, n_records,
      records, n_regions, regions, finalize}`` and the bindings
      ``openmc.lib.rmc.sweep_*``.

Verified by ``tests/rmc/test_segment_records.py`` (gate G2.6)
    - Driven one batch at a time in the CE process, random ray reproduces
      the multigroup run's :math:`k` in every batch, bitwise, on one thread.
    - With the naive volume estimator, every region's flux iterate equals
      the length-weighted mean of its records' :math:`\bar\psi` to
      :math:`10^{-6}` (single-precision attenuation). This ties the records
      to the solver's own update.
    - The recorded lengths sum to rays times the active distance, to
      :math:`10^{-6}` (random ray trims ``TINY_BIT`` at mesh crossings).
    - Only flagged regions are recorded; an empty list stops recording.
    - The random ray regression tests are bitwise unchanged.

Not yet done
    Setting region sources from an RMC field (needed once the field changes
    between generations), and the direction bundle built from the records
    (Stage 4).
