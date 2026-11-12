//! \file capi.h
//! \brief C API for multilevel RMC, called from Python through openmc.lib.rmc.
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

//! Region volumes [cm^3] from n_per_voxel uniform points in every voxel of
//! the field's source-region mesh. volume has openmc_rmc_field_n_regions()
//! entries.
int openmc_rmc_field_region_volumes(
  int64_t n_per_voxel, uint64_t seed, double* volume);
}

#endif // OPENMC_RMC_CAPI_H
