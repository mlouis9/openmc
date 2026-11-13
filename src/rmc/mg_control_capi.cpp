//! \file mg_control_capi.cpp
//! \brief C API for the field's groups and shapes and the multigroup
//! control.

#include <algorithm>

#include "openmc/capi.h"
#include "openmc/cell.h"
#include "openmc/error.h"
#include "openmc/geometry.h"
#include "openmc/rmc/backward.h"
#include "openmc/rmc/capi.h"
#include "openmc/rmc/mg_control.h"

using namespace openmc;

namespace {
int no_field()
{
  set_errmsg("No RMC field is loaded; call openmc_rmc_field_load first.");
  return OPENMC_E_ALLOCATE;
}
} // namespace

extern "C" int openmc_rmc_field_groups(int* n_groups, double* edges)
{
  if (!rmc::loaded_field)
    return no_field();
  const auto& e = rmc::loaded_field->group_edges();
  *n_groups = rmc::loaded_field->n_groups();
  if (edges)
    std::copy(e.begin(), e.end(), edges);
  return 0;
}

extern "C" int openmc_rmc_field_emission(int64_t region, int tag, double* Q)
{
  if (!rmc::loaded_field)
    return no_field();
  const auto& f = *rmc::loaded_field;
  if (region < 0 || region >= f.n_regions()) {
    set_errmsg("Region index out of range.");
    return OPENMC_E_OUT_OF_BOUNDS;
  }
  for (int g = 0; g < f.n_groups(); ++g)
    Q[g] = f.emission(region, static_cast<rmc::Tag>(tag), g);
  return 0;
}

extern "C" int openmc_rmc_field_shape(int64_t n, const int64_t* region,
  const double* E, int which, double* h)
{
  if (!rmc::loaded_field)
    return no_field();
  auto k = static_cast<rmc::FrozenField::Shape>(which);
  if (!rmc::loaded_field->has_shape(k)) {
    set_errmsg("The lift file has no flux shape.");
    return OPENMC_E_INVALID_ARGUMENT;
  }
  for (int64_t i = 0; i < n; ++i)
    h[i] = rmc::loaded_field->shape(k, region[i], E[i]);
  return 0;
}

extern "C" int openmc_rmc_mg_control(const double* r, const double* u_in,
  int64_t n_E, const double* E, int tag, int tail, double cutoff,
  double length, int64_t max_segments, uint64_t seed, double* q_s,
  double* q_f, double* mg_s, double* mg_f)
{
  if (!rmc::loaded_field)
    return no_field();
  const auto& field = *rmc::loaded_field;
  const auto& tables = rmc::loaded_mg_tables();
  // Destination state
  GeometryState gs;
  gs.init_from_r_u({r[0], r[1], r[2]}, {0.0, 0.0, 1.0});
  if (!exhaustive_find_cell(gs)) {
    set_errmsg("Point is outside the geometry.");
    return OPENMC_E_GEOMETRY;
  }
  const Cell& cell = *model::cells[gs.lowest_coord().cell()];
  int32_t inst = gs.cell_instance();
  int32_t m = cell.material(inst);
  double sqrtkT = cell.sqrtkT(inst);
  double dm = cell.density_mult(inst);
  int64_t region = field.locate(gs);

  rmc::TailOptions opt;
  opt.mode = static_cast<rmc::Tail>(tail);
  opt.cutoff = cutoff;
  opt.length = length;
  opt.max_segments = max_segments;
  rmc::BackwardFlux bf(field);
  vector<double> none, ps, pf, ms, mf;
  uint64_t s = seed;
  rmc::PathStats stats;
  Direction u_back {-u_in[0], -u_in[1], -u_in[2]};
  bf.evaluate_all(gs.r(), u_back, none, &tables, opt, &s, ps, pf, ms, mf, &stats);
  if (stats.hit_limit && opt.mode == rmc::Tail::vacuum) {
    set_errmsg("The backward path did not reach a vacuum boundary.");
    return OPENMC_E_INVALID_ARGUMENT;
  }
  rmc::MGControl ctl(field, tables);
  auto t = static_cast<rmc::Tag>(tag);
  for (int64_t i = 0; i < n_E; ++i) {
    q_s[i] = dm * ctl.production(t, region, m, sqrtkT, E[i], ms);
    q_f[i] = dm * ctl.production(t, region, m, sqrtkT, E[i], mf);
  }
  std::copy(ms.begin(), ms.end(), mg_s);
  std::copy(mf.begin(), mf.end(), mg_f);
  return 0;
}
