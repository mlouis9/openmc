//! \file channels_capi.cpp
//! \brief C API for the channel registry and the sampler route.

#include <memory>

#include "openmc/capi.h"
#include "openmc/error.h"
#include "openmc/material.h"
#include "openmc/rmc/capi.h"
#include "openmc/rmc/channels.h"
#include "openmc/rmc/xs_probe.h"

using namespace openmc;

namespace {

int find_material(int32_t id, int32_t& index)
{
  auto it = model::materials.empty() ? model::material_map.end()
                                     : model::material_map.find(id);
  if (it == model::material_map.end()) {
    set_errmsg("No material with that ID.");
    return OPENMC_E_INVALID_ID;
  }
  index = it->second;
  // Lookups need the energy grid (see backward_capi.cpp)
  double E = 1.0, dummy;
  return openmc_rmc_xs_total(id, 1, &E, 0.16, 1.0, &dummy);
}

} // namespace

extern "C" int openmc_rmc_channel_production(int32_t material_id, int64_t n,
  const double* E, double sqrtkT, double density_mult, double* by_kind,
  double* by_route)
{
  int32_t m;
  if (int err = find_material(material_id, m))
    return err;
  rmc::ChannelRegistry reg;
  rmc::XsProbe probe;
  for (int64_t i = 0; i < n; ++i) {
    probe.total(m, E[i], sqrtkT, density_mult);
    double* k = by_kind + 4 * i;
    double* r = by_route + 4 * i;
    for (int j = 0; j < 4; ++j)
      k[j] = r[j] = 0.0;
    for (const auto& c : reg.channels(m)) {
      double x = rmc::ChannelRegistry::production(c, density_mult, probe);
      k[static_cast<int>(c.kind)] += x;
      int route = static_cast<int>(reg.route(c, E[i], probe));
      r[2 * static_cast<int>(c.tag) + route] += x;
    }
  }
  return 0;
}

extern "C" int openmc_rmc_sample_sampler_route(int32_t material_id, double E_in,
  const double* u_in, double sqrtkT, double density_mult, int64_t n,
  uint64_t seed, double* production, double* E, double* u, int* kind)
{
  int32_t m;
  if (int err = find_material(material_id, m))
    return err;
  rmc::ChannelRegistry reg;
  rmc::XsProbe probe;
  Direction d {u_in[0], u_in[1], u_in[2]};
  uint64_t s = seed;
  for (int64_t i = 0; i < n; ++i) {
    probe.total(m, E_in, sqrtkT, density_mult);
    auto out = rmc::sample_sampler_route(reg, rmc::Tag::s, d, probe, &s);
    production[i] = out.production;
    E[i] = out.E;
    u[3 * i] = out.u.x;
    u[3 * i + 1] = out.u.y;
    u[3 * i + 2] = out.u.z;
    kind[i] = static_cast<int>(out.kind);
  }
  return 0;
}
