#include "openmc/rmc/capi.h"

#include <exception>

#include "openmc/capi.h"
#include "openmc/error.h"
#include "openmc/geometry.h"
#include "openmc/memory.h"
#include "openmc/mesh.h"
#include "openmc/random_lcg.h"
#include "openmc/rmc/field.h"

namespace openmc {
namespace rmc {

//! The loaded field. Owned here; everything else borrows it.
unique_ptr<FrozenField> loaded_field;

} // namespace rmc
} // namespace openmc

using namespace openmc;

namespace {

int no_field()
{
  set_errmsg("No RMC field is loaded; call openmc_rmc_field_load first.");
  return OPENMC_E_ALLOCATE;
}

//! Place a scratch geometry state at a point and find its cell.
bool locate_point(const double* x, GeometryState& p)
{
  p.init_from_r_u({x[0], x[1], x[2]}, {0.0, 0.0, 1.0});
  return exhaustive_find_cell(p);
}

} // namespace

extern "C" int openmc_rmc_field_load(
  const char* field_file, const char* lift_file, const char* volume_file)
{
  // fatal_error() inside the constructor ends the process; that is accepted
  // for malformed input files.
  rmc::loaded_field =
    make_unique<rmc::FrozenField>(field_file, lift_file, volume_file);
  return 0;
}

extern "C" int openmc_rmc_field_free()
{
  rmc::loaded_field.reset();
  return 0;
}

extern "C" int openmc_rmc_field_n_regions(int64_t* n)
{
  if (!rmc::loaded_field)
    return no_field();
  *n = rmc::loaded_field->n_regions();
  return 0;
}

extern "C" int openmc_rmc_field_locate(
  int64_t n, const double* xyz, int64_t* region)
{
  if (!rmc::loaded_field)
    return no_field();
  GeometryState p;
  for (int64_t i = 0; i < n; ++i) {
    region[i] =
      locate_point(xyz + 3 * i, p) ? rmc::loaded_field->locate(p) : -1;
  }
  return 0;
}

extern "C" int openmc_rmc_field_evaluate(
  int64_t n, const int64_t* region, const double* E, int tag, double* out)
{
  if (!rmc::loaded_field)
    return no_field();
  if (tag != 0 && tag != 1) {
    set_errmsg("RMC field: tag must be 0 (s) or 1 (f).");
    return OPENMC_E_INVALID_ARGUMENT;
  }
  const auto t = static_cast<rmc::Tag>(tag);
  for (int64_t i = 0; i < n; ++i)
    out[i] = rmc::loaded_field->evaluate(region[i], t, E[i]);
  return 0;
}

extern "C" int openmc_rmc_field_mass(int tag, double* mass)
{
  if (!rmc::loaded_field)
    return no_field();
  if (tag != 0 && tag != 1) {
    set_errmsg("RMC field: tag must be 0 (s) or 1 (f).");
    return OPENMC_E_INVALID_ARGUMENT;
  }
  *mass = rmc::loaded_field->mass(static_cast<rmc::Tag>(tag));
  return 0;
}

extern "C" int openmc_rmc_field_region_volumes(
  int64_t n_per_voxel, uint64_t seed, double* volume)
{
  if (!rmc::loaded_field)
    return no_field();
  const auto& f = *rmc::loaded_field;
  if (f.mesh_index() < 0) {
    set_errmsg("RMC field: no source-region mesh to sample voxels from.");
    return OPENMC_E_INVALID_ARGUMENT;
  }
  auto* mesh = dynamic_cast<RegularMesh*>(model::meshes[f.mesh_index()].get());
  if (!mesh) {
    set_errmsg("RMC field: region volumes require a RegularMesh.");
    return OPENMC_E_INVALID_TYPE;
  }
  if (n_per_voxel <= 0) {
    set_errmsg("RMC field: n_per_voxel must be positive.");
    return OPENMC_E_INVALID_ARGUMENT;
  }

  const int64_t R = f.n_regions();
  vector<int64_t> counts(R, 0);
  const Position ll = mesh->lower_left();
  const Position w {mesh->width_[0], mesh->width_[1], mesh->width_[2]};
  const double V_voxel = w.x * w.y * w.z;

  GeometryState p;
  uint64_t s = seed;
  for (int k = 0; k < mesh->shape_[2]; ++k) {
    for (int j = 0; j < mesh->shape_[1]; ++j) {
      for (int i = 0; i < mesh->shape_[0]; ++i) {
        for (int64_t n = 0; n < n_per_voxel; ++n) {
          double x[3] = {ll.x + w.x * (i + prn(&s)), ll.y + w.y * (j + prn(&s)),
            ll.z + w.z * (k + prn(&s))};
          if (!locate_point(x, p))
            continue;
          int64_t r = f.locate(p);
          if (r >= 0)
            ++counts[r];
        }
      }
    }
  }
  for (int64_t r = 0; r < R; ++r)
    volume[r] = V_voxel * static_cast<double>(counts[r]) / n_per_voxel;
  return 0;
}
