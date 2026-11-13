//! \file backward_capi.cpp
//! \brief C API for the backward characteristic and the cross-section probe.

#include <algorithm>

#include <fmt/format.h>

#include "openmc/capi.h"
#include "openmc/error.h"
#include "openmc/geometry.h"
#include "openmc/material.h"
#include "openmc/nuclide.h"
#include "openmc/rmc/backward.h"
#include "openmc/rmc/capi.h"
#include "openmc/simulation.h"

using namespace openmc;

namespace {

//! Cross-section lookups need the logarithmic energy grid of every nuclide,
//! which OpenMC sets up in openmc_simulation_init(); set it up here if this
//! session has not run a simulation. (log_spacing alone is not a test: it
//! survives openmc_finalize() while the nuclides are reloaded.)
void ensure_xs_grid()
{
  bool ready = simulation::log_spacing > 0.0;
  for (const auto& nuc : data::nuclides) {
    if (nuc->grid_.empty() || nuc->grid_[0].grid_index.empty())
      ready = false;
  }
  if (!ready)
    initialize_data();
}

} // namespace

extern "C" int openmc_rmc_xs_total(int32_t material_id, int64_t n,
  const double* E, double sqrtkT, double density_mult, double* sigma_t)
{
  auto it = model::material_map.find(material_id);
  if (it == model::material_map.end()) {
    set_errmsg(fmt::format("No material with ID {}.", material_id));
    return OPENMC_E_INVALID_ID;
  }
  ensure_xs_grid();
  rmc::XsProbe probe;
  for (int64_t i = 0; i < n; ++i)
    sigma_t[i] = probe.total(it->second, E[i], sqrtkT, density_mult);
  return 0;
}

extern "C" int openmc_rmc_backward_psi(const double* r, const double* u,
  int n_E, const double* E, double a_s, double a_f, int tail, double cutoff,
  double length, int64_t max_segments, uint64_t seed, double* psi,
  int64_t* n_segments)
{
  if (!rmc::loaded_field) {
    set_errmsg("No RMC field is loaded; call openmc_rmc_field_load first.");
    return OPENMC_E_ALLOCATE;
  }
  if (tail < 0 || tail > 2) {
    set_errmsg("tail must be 0 (vacuum), 1 (roulette), or 2 (fixed length).");
    return OPENMC_E_INVALID_ARGUMENT;
  }
  rmc::TailOptions opt;
  opt.mode = static_cast<rmc::Tail>(tail);
  if (opt.mode == rmc::Tail::roulette && !(cutoff > 0.0 && cutoff < 1.0)) {
    set_errmsg("The roulette cutoff must be in (0, 1).");
    return OPENMC_E_INVALID_ARGUMENT;
  }
  if (opt.mode == rmc::Tail::fixed_length && !(length > 0.0)) {
    set_errmsg("The fixed-length tail needs a positive length.");
    return OPENMC_E_INVALID_ARGUMENT;
  }
  opt.cutoff = cutoff;
  opt.length = length;
  opt.max_segments = max_segments;

  GeometryState probe;
  probe.init_from_r_u({r[0], r[1], r[2]}, {0.0, 0.0, 1.0});
  if (!exhaustive_find_cell(probe)) {
    set_errmsg("Start point is outside the geometry.");
    return OPENMC_E_GEOMETRY;
  }

  ensure_xs_grid();
  rmc::BackwardFlux bf(*rmc::loaded_field);
  vector<double> energies(E, E + n_E), out;
  rmc::PathStats stats;
  Direction dir {u[0], u[1], u[2]};
  uint64_t s = seed;
  bf.evaluate({r[0], r[1], r[2]}, dir / dir.norm(), energies, a_s, a_f, opt, &s,
    out, &stats);
  if (stats.hit_limit) {
    set_errmsg(fmt::format(
      "The backward path reached {} segments without ending{}.", max_segments,
      opt.mode == rmc::Tail::vacuum
        ? "; the vacuum tail needs a path that reaches a vacuum boundary, so "
          "use the roulette or fixed-length tail in reflective problems"
        : ""));
    return OPENMC_E_OUT_OF_BOUNDS;
  }
  std::copy(out.begin(), out.end(), psi);
  *n_segments = stats.segments;
  return 0;
}
