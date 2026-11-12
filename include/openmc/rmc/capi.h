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
}

#endif // OPENMC_RMC_CAPI_H
