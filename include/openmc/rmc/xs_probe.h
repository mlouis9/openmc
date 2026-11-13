//! \file xs_probe.h
//! \brief Continuous-energy cross sections at an arbitrary energy.

#ifndef OPENMC_RMC_XS_PROBE_H
#define OPENMC_RMC_XS_PROBE_H

#include <cstdint>

#include "openmc/particle.h"

namespace openmc {
namespace rmc {

//==============================================================================
//! Looks up continuous-energy cross sections through OpenMC's own routine,
//! Material::calculate_xs(), on a scratch particle that is never transported.
//!
//! The probe hides OpenMC's particle-centred interface behind the question
//! the backward characteristic asks: what is Sigma_t in this material, at
//! this energy and temperature? OpenMC caches microscopic cross sections in
//! the particle by energy and temperature, so a probe is per-thread state:
//! never share one between threads. Lookups need OpenMC's logarithmic
//! energy grid, which initialize_data() (called by openmc_simulation_init())
//! sets up.
//==============================================================================

class XsProbe {
public:
  //! Macroscopic total cross section [1/cm]
  //! \param material      Index into model::materials, or MATERIAL_VOID
  //! \param E             Energy [eV]
  //! \param sqrtkT        sqrt(kT) [sqrt(eV)] of the cell instance
  //! \param density_mult  Density multiplier of the cell instance
  double total(int32_t material, double E, double sqrtkT, double density_mult);

  //! The scratch particle after the last total() call: its macro_xs() and
  //! neutron_xs(i_nuclide) hold every cross section at that state.
  const Particle& particle() const { return p_; }

private:
  Particle p_;
};

} // namespace rmc
} // namespace openmc

#endif // OPENMC_RMC_XS_PROBE_H
