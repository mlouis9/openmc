//! \file correction_capi.cpp
//! \brief C API for the correction tallies and exact cancellation kernels,
//! and flights with per-child and per-region outputs.

#include <algorithm>

#include "openmc/capi.h"
#include "openmc/error.h"
#include "openmc/material.h"
#include "openmc/nuclide.h"
#include "openmc/rmc/capi.h"
#include "openmc/rmc/correction.h"
#include "openmc/rmc/mg_control.h"
#include "openmc/rmc/flight.h"
#include "openmc/rmc/residual.h"

using namespace openmc;

namespace {

rmc::FlightAux last_aux;

int prepare()
{
  if (!rmc::loaded_field) {
    set_errmsg("No RMC field is loaded.");
    return OPENMC_E_ALLOCATE;
  }
  double dummy, E1 = 1.0;
  return openmc_rmc_xs_total(model::materials[0]->id(), 1, &E1, 0.16, 1.0, &dummy);
}

} // namespace

extern "C" int openmc_rmc_field_tally(int64_t n_points, int n_E, int tail,
  double cutoff, uint64_t seed, double* fission, double* nu_fission,
  double* fission_sq, int use_mg, double* nu_fission_mg, double* diff_sq)
{
  if (int err = prepare())
    return err;
  rmc::TailOptions opt;
  opt.mode = static_cast<rmc::Tail>(tail);
  opt.cutoff = cutoff;
  rmc::FieldTallyResult res;
  const rmc::MGTables* tables = use_mg ? &rmc::loaded_mg_tables() : nullptr;
  rmc::field_reaction_tally(
    *rmc::loaded_field, n_points, n_E, opt, seed, res, tables);
  std::copy(res.fission.begin(), res.fission.end(), fission);
  std::copy(res.nu_fission_mg.begin(), res.nu_fission_mg.end(), nu_fission_mg);
  std::copy(res.diff_sq.begin(), res.diff_sq.end(), diff_sq);
  std::copy(res.nu_fission.begin(), res.nu_fission.end(), nu_fission);
  std::copy(res.fission_sq.begin(), res.fission_sq.end(), fission_sq);
  return 0;
}

extern "C" int openmc_rmc_field_production(int64_t n_points, int n_E,
  int tail, double cutoff, uint64_t seed, double* X, double* var, int score,
  double du, double sigma_e, int classes)
{
  if (int err = prepare())
    return err;
  rmc::TailOptions opt;
  opt.mode = static_cast<rmc::Tail>(tail);
  opt.cutoff = cutoff;
  rmc::field_fission_production(*rmc::loaded_field, rmc::loaded_mg_tables(),
    n_points, n_E, opt, seed, X, var, score, du, sigma_e, classes);
  return 0;
}

extern "C" int openmc_rmc_fission_chi(int32_t material_id, int i_nuclide,
  double E_in, int64_t n, const double* E_out, double* chi)
{
  auto it = model::material_map.find(material_id);
  if (it == model::material_map.end()) {
    set_errmsg("No material with that ID.");
    return OPENMC_E_INVALID_ID;
  }
  if (int err = prepare())
    return err;
  rmc::XsProbe probe;
  probe.total(it->second, E_in, std::sqrt(294.0 * K_BOLTZMANN), 1.0);
  for (int64_t i = 0; i < n; ++i)
    chi[i] = rmc::fission_chi(i_nuclide, probe, E_out[i]);
  return 0;
}

extern "C" int openmc_rmc_near_kernel(int64_t n, const double* x,
  const double* xp, const double* E, double d_max, double* out)
{
  if (int err = prepare())
    return err;
  const auto& field = *rmc::loaded_field;
  rmc::KernelScratch tracer;
  rmc::XsProbe probe;
  rmc::SegmentList seg;
  for (int64_t i = 0; i < n; ++i) {
    Position a {x[3 * i], x[3 * i + 1], x[3 * i + 2]};
    Position b {xp[3 * i], xp[3 * i + 1], xp[3 * i + 2]};
    out[i] = rmc::near_kernel(a, b, E[i], d_max, field.box_lower_left(),
      field.box_upper_right(), tracer, probe, seg);
  }
  return 0;
}

extern "C" int openmc_rmc_propagate_aux(int32_t in, int64_t n, double k,
  uint64_t seed, int32_t out, int children, int region_tally, int history,
  int64_t* n_children, double* region_fission, double* region_nu_fission,
  double* stats);

extern "C" int openmc_rmc_flight_children_incident(
  double* E_in, double* u_in, int* mt)
{
  std::copy(last_aux.E_in.begin(), last_aux.E_in.end(), E_in);
  std::copy(last_aux.u_in.begin(), last_aux.u_in.end(), u_in);
  std::copy(last_aux.mt.begin(), last_aux.mt.end(), mt);
  return 0;
}

extern "C" int openmc_rmc_grouped_fission_weights(int64_t n_cand,
  const double* x, const int* nuclide, const double* E_f,
  const int64_t* offset, const int64_t* member, int64_t n_rec,
  const double* rec_r, const int* rec_kind, const double* rec_E,
  const double* rec_u, const int* rec_nuclide, const double* rec_w,
  const double* rec_m, double d_max, double k, double* W, int* ok)
{
  if (int err = prepare())
    return err;
  rmc::KernelScratch tracer;
  rmc::XsProbe probe;
  vector<rmc::EmissionRecord> group;
  for (int64_t c = 0; c < n_cand; ++c) {
    group.clear();
    for (int64_t q = offset[c]; q < offset[c + 1]; ++q) {
      int64_t j = member[q];
      if (j < 0 || j >= n_rec)
        return OPENMC_E_INVALID_ARGUMENT;
      rmc::EmissionRecord r;
      r.r = {rec_r[3 * j], rec_r[3 * j + 1], rec_r[3 * j + 2]};
      r.kind = rec_kind[j];
      r.E = rec_E[j];
      r.u = {rec_u[3 * j], rec_u[3 * j + 1], rec_u[3 * j + 2]};
      r.nuclide = rec_nuclide[j];
      r.w = rec_w[j];
      r.m = rec_m[j];
      group.push_back(r);
    }
    double w = 0.0;
    ok[c] = rmc::grouped_fission_weight(*rmc::loaded_field, group,
      {x[3 * c], x[3 * c + 1], x[3 * c + 2]}, nuclide[c], E_f[c], d_max, k,
      tracer, probe, w);
    W[c] = w;
  }
  return 0;
}

extern "C" int openmc_rmc_flight_children(
  int64_t* parent, double* path, int* nuclide)
{
  std::copy(last_aux.parent.begin(), last_aux.parent.end(), parent);
  std::copy(last_aux.path.begin(), last_aux.path.end(), path);
  std::copy(last_aux.nuclide.begin(), last_aux.nuclide.end(), nuclide);
  return 0;
}

namespace openmc {
namespace rmc {
// Defined in bank_capi.cpp: the bank behind a handle
SignedBank* bank_of(int32_t handle);
} // namespace rmc
} // namespace openmc

extern "C" int openmc_rmc_propagate_aux(int32_t in, int64_t n, double k,
  uint64_t seed, int32_t out, int children, int region_tally, int history,
  int64_t* n_children, double* region_fission, double* region_nu_fission,
  double* stats)
{
  auto* bi = rmc::bank_of(in);
  auto* bo = rmc::bank_of(out);
  if (!bi || !bo)
    return OPENMC_E_INVALID_ID;
  if (int err = prepare())
    return err;
  last_aux = rmc::FlightAux();
  last_aux.children = children != 0;
  const int64_t R = rmc::loaded_field->n_regions();
  if (region_tally) {
    last_aux.region_fission.assign(R, 0.0);
    last_aux.region_nu_fission.assign(R, 0.0);
  }
  rmc::FlightPropagator prop;
  auto st =
    prop.propagate(*bi, n, k, seed, *bo, nullptr, &last_aux, history != 0);
  *n_children = static_cast<int64_t>(last_aux.parent.size());
  if (region_tally) {
    std::copy(last_aux.region_fission.begin(), last_aux.region_fission.end(),
      region_fission);
    std::copy(last_aux.region_nu_fission.begin(),
      last_aux.region_nu_fission.end(), region_nu_fission);
  }
  stats[0] = st.flights;
  stats[1] = st.leaked;
  stats[2] = st.captured;
  stats[3] = st.scattered;
  stats[4] = st.fissions;
  stats[5] = st.W_abs;
  stats[6] = st.collisions;
  stats[7] = st.absorption;
  stats[8] = st.multiplied;
  stats[9] = st.nu_fission;
  return 0;
}

extern "C" int openmc_rmc_residual_records_aux(double* E_in, int* nuclide);
