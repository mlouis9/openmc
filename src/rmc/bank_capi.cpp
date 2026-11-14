//! \file bank_capi.cpp
//! \brief C API for signed banks (by integer handle) and forward flights.
//! Banks stay in C++ memory; Python copies arrays in and out.

#include <algorithm>

#include "openmc/capi.h"
#include "openmc/error.h"
#include "openmc/material.h"
#include "openmc/memory.h"
#include "openmc/rmc/capi.h"
#include "openmc/rmc/flight.h"

using namespace openmc;

namespace {

vector<unique_ptr<rmc::SignedBank>> banks;

rmc::SignedBank* get(int32_t h)
{
  if (h < 0 || h >= static_cast<int32_t>(banks.size()) || !banks[h]) {
    set_errmsg("Invalid RMC bank handle.");
    return nullptr;
  }
  return banks[h].get();
}

} // namespace

namespace openmc {
namespace rmc {
SignedBank* bank_of(int32_t handle) { return get(handle); }
} // namespace rmc
} // namespace openmc

extern "C" int openmc_rmc_bank_create(int64_t n, const double* r,
  const double* u, const double* E, const double* w, const int* tag,
  int32_t* handle)
{
  auto b = make_unique<rmc::SignedBank>();
  for (int64_t i = 0; i < n; ++i) {
    if (tag[i] < 0 || tag[i] > 1) {
      set_errmsg("Bank tags must be 0 (s) or 1 (f).");
      return OPENMC_E_INVALID_ARGUMENT;
    }
    b->push_back({{r[3 * i], r[3 * i + 1], r[3 * i + 2]},
      {u[3 * i], u[3 * i + 1], u[3 * i + 2]}, E[i], w[i],
      static_cast<rmc::Tag>(tag[i])});
  }
  // Reuse a freed slot
  for (size_t h = 0; h < banks.size(); ++h) {
    if (!banks[h]) {
      banks[h] = std::move(b);
      *handle = static_cast<int32_t>(h);
      return 0;
    }
  }
  banks.push_back(std::move(b));
  *handle = static_cast<int32_t>(banks.size() - 1);
  return 0;
}

extern "C" int openmc_rmc_bank_size(int32_t handle, int64_t* n)
{
  auto* b = get(handle);
  if (!b)
    return OPENMC_E_INVALID_ID;
  *n = b->size();
  return 0;
}

extern "C" int openmc_rmc_bank_get(int32_t handle, double* r, double* u,
  double* E, double* w, int* tag)
{
  auto* b = get(handle);
  if (!b)
    return OPENMC_E_INVALID_ID;
  for (int64_t i = 0; i < b->size(); ++i) {
    r[3 * i] = b->x[i];
    r[3 * i + 1] = b->y[i];
    r[3 * i + 2] = b->z[i];
    u[3 * i] = b->ux[i];
    u[3 * i + 1] = b->uy[i];
    u[3 * i + 2] = b->uz[i];
  }
  std::copy(b->E.begin(), b->E.end(), E);
  std::copy(b->w.begin(), b->w.end(), w);
  std::copy(b->tag.begin(), b->tag.end(), tag);
  return 0;
}

extern "C" int openmc_rmc_bank_free(int32_t handle)
{
  if (!get(handle))
    return OPENMC_E_INVALID_ID;
  banks[handle].reset();
  return 0;
}

extern "C" int openmc_rmc_propagate(int32_t in, int64_t n, double k,
  uint64_t seed, int32_t out, int n_materials, const int32_t* material_ids,
  int n_edges, const double* E_edges, double* flux, double* stats)
{
  auto* bi = get(in);
  auto* bo = get(out);
  if (!bi || !bo)
    return OPENMC_E_INVALID_ID;
  if (in == out) {
    set_errmsg("Flights need different input and output banks.");
    return OPENMC_E_INVALID_ARGUMENT;
  }
  // Lookups need the energy grid (see backward_capi.cpp)
  double dummy, E1 = 1.0;
  if (int err = openmc_rmc_xs_total(
        model::materials[0]->id(), 1, &E1, 0.16, 1.0, &dummy))
    return err;

  rmc::FlightTally tally;
  if (n_materials > 0 && n_edges > 1) {
    for (int i = 0; i < n_materials; ++i) {
      auto it = model::material_map.find(material_ids[i]);
      if (it == model::material_map.end()) {
        set_errmsg("No material with that ID.");
        return OPENMC_E_INVALID_ID;
      }
      tally.materials.push_back(it->second);
    }
    tally.E_edges.assign(E_edges, E_edges + n_edges);
    tally.reset();
  }
  rmc::FlightPropagator prop;
  auto st = prop.propagate(*bi, n, k, seed, *bo, tally.active() ? &tally : nullptr);
  if (tally.active() && flux)
    std::copy(tally.flux.begin(), tally.flux.end(), flux);
  if (stats) {
    stats[0] = st.flights;
    stats[1] = st.leaked;
    stats[2] = st.captured;
    stats[3] = st.scattered;
    stats[4] = st.fissions;
    stats[5] = st.W_abs;
  }
  return 0;
}
