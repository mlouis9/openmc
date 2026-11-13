#include "openmc/rmc/channels.h"

#include <cmath>

#include "openmc/constants.h"
#include "openmc/error.h"
#include "openmc/material.h"
#include "openmc/math_functions.h"
#include "openmc/nuclide.h"
#include "openmc/physics.h"
#include "openmc/random_lcg.h"
#include "openmc/reaction.h"
#include "openmc/settings.h"

namespace openmc {
namespace rmc {

//==============================================================================
// ChannelRegistry
//==============================================================================

ChannelRegistry::ChannelRegistry()
{
  channels_.resize(model::materials.size());
  for (int m = 0; m < static_cast<int>(model::materials.size()); ++m) {
    const Material& mat = *model::materials[m];
    if (!mat.p0_.empty()) {
      for (bool p0 : mat.p0_)
        if (p0)
          fatal_error("RMC channels: isotropic-in-lab scattering (p0) is not "
                      "supported.");
    }
    for (int j = 0; j < static_cast<int>(mat.nuclide_.size()); ++j) {
      int i_nuc = mat.nuclide_[j];
      const Nuclide& nuc = *data::nuclides[i_nuc];
      auto& out = channels_[m];
      out.push_back({j, i_nuc, 0, ChannelKind::elastic, Tag::s});
      out.push_back({j, i_nuc, 0, ChannelKind::thermal_elastic, Tag::s});
      for (int i_rx : nuc.index_inelastic_scatter_)
        out.push_back({j, i_nuc, i_rx, ChannelKind::inelastic, Tag::s});
      if (nuc.fissionable_)
        out.push_back({j, i_nuc, -1, ChannelKind::fission, Tag::f});
    }
  }
}

double ChannelRegistry::elastic_kT(int i_nuclide, const XsProbe& probe)
{
  // As scatter() in physics.cpp
  const Nuclide& nuc = *data::nuclides[i_nuclide];
  const Particle& p = probe.particle();
  if (nuc.multipole_)
    return p.sqrtkT() * p.sqrtkT();
  int i_temp = p.neutron_xs(i_nuclide).index_temp;
  return nuc.kTs_[std::max(i_temp, 0)];
}

Route ChannelRegistry::route(
  const Channel& c, double E_in, const XsProbe& probe)
{
  switch (c.kind) {
  case ChannelKind::elastic: {
    // The condition of sample_target_velocity() for a target at rest, and
    // of elastic_stationary() (src/rmc/production.cpp)
    const Nuclide& nuc = *data::nuclides[c.i_nuclide];
    double kT = elastic_kT(c.i_nuclide, probe);
    const auto& grid = nuc.grid_[0].energy;
    bool stationary = nuc.awr_ > 1.0 && !nuc.resonant_ &&
                      E_in >= settings::free_gas_threshold * kT &&
                      E_in >= grid.front() && E_in <= grid.back();
    return stationary ? Route::evaluator : Route::sampler;
  }
  case ChannelKind::fission:
    return Route::evaluator;
  default:
    return Route::sampler;
  }
}

double ChannelRegistry::production(
  const Channel& c, double density_mult, XsProbe& probe)
{
  const Material& mat = *model::materials[probe.material()];
  double N = mat.atom_density(c.i_mat, density_mult);
  const Particle& p = probe.particle();
  const auto& micro = p.neutron_xs(c.i_nuclide);
  const Nuclide& nuc = *data::nuclides[c.i_nuclide];
  switch (c.kind) {
  case ChannelKind::elastic:
    // micro.elastic includes the S(alpha,beta) part when a table applies
    return N * (probe.micro_elastic(c.i_nuclide) - micro.thermal);
  case ChannelKind::thermal_elastic:
    return N * micro.thermal;
  case ChannelKind::inelastic: {
    const Reaction& rx = *nuc.reactions_[c.i_reaction];
    double xs = rx.xs(micro);
    if (xs <= 0.0)
      return 0.0;
    return N * (*rx.products_[0].yield_)(probe.E()) * xs;
  }
  case ChannelKind::fission:
    // OpenMC tabulates nu * sigma_f on the energy grid and interpolates the
    // product (Nuclide::calculate_xs), and create_fission_sites() uses that
    return N * micro.nu_fission;
  }
  return 0.0;
}

double ChannelRegistry::production(Tag t, Route r, XsProbe& probe) const
{
  if (probe.material() == MATERIAL_VOID)
    return 0.0;
  double sum = 0.0;
  for (const Channel& c : channels_[probe.material()]) {
    if (c.tag == t && route(c, probe.E(), probe) == r)
      sum += production(c, probe.density_mult(), probe);
  }
  return sum;
}

//==============================================================================
// Sampler route
//==============================================================================

SamplerOutcome sample_sampler_route(const ChannelRegistry& reg, Tag t,
  Direction u_in, XsProbe& probe, uint64_t* seed)
{
  SamplerOutcome out;
  if (probe.material() == MATERIAL_VOID)
    return out;
  const double E_in = probe.E();

  // Production of each sampler-route channel; choose one in proportion
  const auto& ch = reg.channels(probe.material());
  double total = 0.0;
  for (const Channel& c : ch)
    if (c.tag == t && ChannelRegistry::route(c, E_in, probe) == Route::sampler)
      total += ChannelRegistry::production(c, probe.density_mult(), probe);
  out.production = total;
  if (total <= 0.0)
    return out;
  double cutoff = prn(seed) * total;
  const Channel* chosen = nullptr;
  double cum = 0.0;
  for (const Channel& c : ch) {
    if (c.tag != t || ChannelRegistry::route(c, E_in, probe) != Route::sampler)
      continue;
    double x = ChannelRegistry::production(c, probe.density_mult(), probe);
    if (x <= 0.0)
      continue;
    chosen = &c;
    cum += x;
    if (cum > cutoff)
      break;
  }

  if (!chosen)
    return out; // round-off: every positive channel was below the cutoff
  // Apply OpenMC's routine to the probe's particle, on our stream
  Particle& p = probe.scratch();
  p.E() = E_in;
  p.u() = u_in;
  p.wgt() = 1.0;
  // A fresh particle's stream index is not initialized
  p.stream() = STREAM_TRACKING;
  const int stream = STREAM_TRACKING;
  p.seeds(stream) = *seed;
  const Nuclide& nuc = *data::nuclides[chosen->i_nuclide];
  switch (chosen->kind) {
  case ChannelKind::elastic:
    elastic_scatter(chosen->i_nuclide, *nuc.reactions_[0],
      ChannelRegistry::elastic_kT(chosen->i_nuclide, probe), p);
    break;
  case ChannelKind::thermal_elastic:
    sab_scatter(chosen->i_nuclide, p.neutron_xs(chosen->i_nuclide).index_sab, p);
    break;
  case ChannelKind::inelastic:
    inelastic_scatter(nuc, *nuc.reactions_[chosen->i_reaction], p);
    break;
  case ChannelKind::fission:
    fatal_error("RMC sampler route: fission is on the evaluator route.");
  }
  *seed = p.seeds(stream);
  p.local_secondary_bank().clear();
  p.n_secondaries() = 0;

  out.E = p.E();
  out.u = p.u();
  out.i_nuclide = chosen->i_nuclide;
  out.kind = chosen->kind;
  // Restore the probe's incident state for later lookups
  p.E() = E_in;
  return out;
}

} // namespace rmc
} // namespace openmc
