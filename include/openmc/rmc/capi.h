//! \file capi.h
//! \brief C API for residual Monte Carlo (RMC), called from Python through
//! openmc.lib.rmc.
//!
//! Every function returns 0 on success or a negative OpenMC error code, with
//! the message available from openmc_get_err_msg(). Only C types cross this
//! boundary.

#ifndef OPENMC_RMC_CAPI_H
#define OPENMC_RMC_CAPI_H

#include <cstdint>

extern "C" {

//! Load the frozen field. Requires an initialized CE model (openmc_init).
//! An empty volume_file loads it unnormalized, for computing volumes.
int openmc_rmc_field_load(
  const char* field_file, const char* lift_file, const char* volume_file);

//! Release the loaded field.
int openmc_rmc_field_free();

//! Number of regions in the loaded field.
int openmc_rmc_field_n_regions(int64_t* n);

//! Locate n points; region[i] = -1 where the field is zero.
//! \param xyz  [n][3] global coordinates
int openmc_rmc_field_locate(int64_t n, const double* xyz, int64_t* region);

//! Evaluate Q_t for n (region, energy) pairs; tag 0 = s, 1 = f.
int openmc_rmc_field_evaluate(
  int64_t n, const int64_t* region, const double* E, int tag, double* out);

//! Field mass <1, Q_t>; tag 0 = s, 1 = f.
int openmc_rmc_field_mass(int tag, double* mass);

//! Region volumes [cm^3] by locating uniform points. Regions on a
//! source-region mesh are sampled with n_per_voxel points in every voxel of
//! their mesh; regions without a mesh (or outside it) with n_box points in the
//! ray source box (n_box may be 0 if there are none). volume and count (hits)
//! have openmc_rmc_field_n_regions() entries.
int openmc_rmc_field_region_volumes(int64_t n_per_voxel, int64_t n_box,
  uint64_t seed, double* volume, int64_t* count);

//! Run OpenMC's random ray solver (a full power iteration with the
//! <random_ray> settings) inside this continuous-energy process, with the
//! multigroup tables of the field file named in <rmc>. Requires an
//! initialized continuous-energy model with an <rmc> element and a
//! <random_ray> element, and no simulation in progress. Defined in
//! src/rmc/random_ray_capi.cpp.
int openmc_rmc_run_random_ray();

//! Random ray one batch at a time in a CE process
//! (src/rmc/random_ray_capi.cpp). The same requirements as
//! openmc_rmc_run_random_ray(), forward eigenvalue only. Random ray stays the
//! solver type until openmc_rmc_sweep_finalize().
int openmc_rmc_sweep_init();
//! Run n batches of the power iteration, exactly as the whole solve does.
int openmc_rmc_sweep_iterate(int n);
//! Record active segments in the given source regions (by index) in later
//! batches; n = 0 stops recording, n < 0 records every existing region.
int openmc_rmc_sweep_set_recording(int64_t n, const int64_t* regions);
//! Number of segments recorded in the last batch.
int openmc_rmc_sweep_n_records(int64_t* n);
//! The last batch's records, ordered by ray and position along it:
//! region [n], u [n][3], length [n] (cm), psi_bar [n][G] (4 pi psi, groups
//! in descending energy as random ray stores them).
int openmc_rmc_sweep_records(
  int64_t* region, double* u, double* length, double* psi_bar);
//! Number of source regions and groups.
int openmc_rmc_sweep_n_regions(int64_t* n, int* n_groups);
//! Per region: key, the volume [cm^3] the solver used, and the latest scalar
//! flux iterate [R][G] (descending energy); and the latest k.
int openmc_rmc_sweep_regions(int32_t* cell_id, int32_t* instance,
  int32_t* mesh_id, int64_t* mesh_bin, double* volume, double* phi, double* k);
//! End the solve: finalize the simulation and restore the solver type.
int openmc_rmc_sweep_finalize();

//! Trace one backward path with rmc::CharacteristicTracer on the loaded
//! field (src/rmc/tracer_capi.cpp). Output arrays hold max_segments entries;
//! n receives the number used. material is an index into model::materials.
int openmc_rmc_trace(const double* r, const double* u, int max_segments, int* n,
  double* length, int32_t* material, int64_t* region, int* ends_in_vacuum);

//! Production evaluators and OpenMC's samplers on a scratch particle
//! (src/rmc/production_capi.cpp). i_nuclide indexes data::nuclides.
//! Fission (first fission reaction): outgoing-energy density and CDF.
int openmc_rmc_fission_density(int i_nuclide, double E_in, int64_t n,
  const double* E_out, double* density, double* cdf);
//! n outgoing energies from sample_fission_neutron().
int openmc_rmc_sample_fission_energy(
  int i_nuclide, double E_in, int64_t n, uint64_t seed, double* E_out);
//! Stationary-target elastic: incident energy and density for each
//! outgoing (E, mu0) (rmc::elastic_stationary).
int openmc_rmc_elastic_image(int i_nuclide, double kT, int64_t n,
  const double* E, const double* mu0, double* E_in, double* density);
//! Outgoing (E, mu0) from elastic_scatter() for each incident energy.
int openmc_rmc_sample_elastic(int i_nuclide, double kT, int64_t n,
  const double* E_in, uint64_t seed, double* E_out, double* mu0);
//! A ContinuousTabular law from group "law" of an HDF5 file (for tests):
//! density and CDF at n outgoing energies, and n_samples samples.
int openmc_rmc_tabular_law(const char* path, double E_in, int64_t n,
  const double* E_out, double* density, double* cdf, int64_t n_samples,
  uint64_t seed, double* samples);
}

#endif // OPENMC_RMC_CAPI_H
