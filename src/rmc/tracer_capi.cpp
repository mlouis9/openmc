//! \file tracer_capi.cpp
//! \brief C API for the backward characteristic tracer.

#include <algorithm>

#include "openmc/capi.h"
#include "openmc/error.h"
#include "openmc/geometry.h"
#include "openmc/rmc/capi.h"
#include "openmc/rmc/tracer.h"

using namespace openmc;

extern "C" int openmc_rmc_trace(const double* r, const double* u,
  int max_segments, int* n, double* length, int32_t* material, int64_t* region,
  int* ends_in_vacuum)
{
  if (!rmc::loaded_field) {
    set_errmsg("No RMC field is loaded; call openmc_rmc_field_load first.");
    return OPENMC_E_ALLOCATE;
  }
  if (max_segments <= 0) {
    set_errmsg("max_segments must be positive.");
    return OPENMC_E_INVALID_ARGUMENT;
  }
  // The tracer stops the program for a start point outside the geometry;
  // check here so that the C API can return an error instead
  GeometryState probe;
  probe.init_from_r_u({r[0], r[1], r[2]}, {0.0, 0.0, 1.0});
  if (!exhaustive_find_cell(probe)) {
    set_errmsg("Start point is outside the geometry.");
    return OPENMC_E_GEOMETRY;
  }

  rmc::CharacteristicTracer tracer(*rmc::loaded_field);
  rmc::SegmentList seg;
  Direction dir {u[0], u[1], u[2]};
  tracer.trace({r[0], r[1], r[2]}, dir / dir.norm(), max_segments, seg);
  *n = seg.size();
  std::copy(seg.length.begin(), seg.length.end(), length);
  std::copy(seg.material.begin(), seg.material.end(), material);
  std::copy(seg.region.begin(), seg.region.end(), region);
  *ends_in_vacuum = seg.ends_in_vacuum;
  return 0;
}
