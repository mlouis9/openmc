#include "openmc/rmc/xs_probe.h"

#include "openmc/constants.h"
#include "openmc/material.h"

namespace openmc {
namespace rmc {

double XsProbe::total(
  int32_t material, double E, double sqrtkT, double density_mult)
{
  if (material == MATERIAL_VOID)
    return 0.0;
  p_.E() = E;
  p_.sqrtkT() = sqrtkT;
  p_.density_mult() = density_mult;
  p_.material() = material;
  model::materials[material]->calculate_xs(p_);
  return p_.macro_xs().total;
}

} // namespace rmc
} // namespace openmc
