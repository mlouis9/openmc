#include "openmc/rmc/xs_probe.h"

#include <algorithm>
#include <cmath>

#include "openmc/constants.h"
#include "openmc/nuclide.h"
#include "openmc/material.h"
#include "openmc/nuclide.h"
#include "openmc/reaction.h"

namespace openmc {
namespace rmc {

double XsProbe::total(
  int32_t material, double E, double sqrtkT, double density_mult)
{
  if (material == MATERIAL_VOID)
    return 0.0;
  // OpenMC's log grid covers [energy_min, energy_max); transport never
  // reaches energy_max, but an energy proposed by RMC can. Clamp into the
  // grid, so data at the edges extends flat.
  const int n = ParticleType::neutron().transport_index();
  E = std::clamp(E, data::energy_min[n],
    data::energy_max[n] * (1.0 - 1e-10)); // below the top log-grid bin edge,
                                          // allowing for round-off
  p_.E() = E;
  p_.sqrtkT() = sqrtkT;
  p_.density_mult() = density_mult;
  p_.material() = material;
  model::materials[material]->calculate_xs(p_);
  return p_.macro_xs().total;
}

double XsProbe::micro_elastic(int i_nuclide)
{
  // OpenMC computes the elastic cross section only when a collision needs
  // it; do the same
  auto& micro = p_.neutron_xs(i_nuclide);
  if (micro.elastic == CACHE_INVALID)
    data::nuclides[i_nuclide]->calculate_elastic_xs(p_);
  return micro.elastic;
}

double XsProbe::micro_fission(int i_nuclide) const
{
  return p_.neutron_xs(i_nuclide).fission;
}

double XsProbe::micro_reaction(int i_nuclide, const Reaction& rx) const
{
  return rx.xs(p_.neutron_xs(i_nuclide));
}

} // namespace rmc
} // namespace openmc
