//! \file channels.h
//! \brief The channel registry: every neutron-producing reaction of every
//! material, with its tag and the route that estimates its production.
//!
//! Theory: docs/source/methods/rmc.rst, methods_rmc_channels.

#ifndef OPENMC_RMC_CHANNELS_H
#define OPENMC_RMC_CHANNELS_H

#include <cstdint>

#include "openmc/position.h"
#include "openmc/rmc/field.h"
#include "openmc/rmc/xs_probe.h"
#include "openmc/vector.h"

namespace openmc {
namespace rmc {

//! How a channel's production at a common output is estimated: by evaluating
//! its density there (Stage 3 evaluators), or by sampling OpenMC's own
//! collision routine and banking the outgoing state.
enum class Route { evaluator, sampler };

enum class ChannelKind {
  elastic,         //!< MT 2 without the S(alpha,beta) part
  thermal_elastic, //!< the S(alpha,beta) part of elastic scattering
  inelastic,       //!< every other non-fission reaction with a neutron
                   //!< product ((n,n'), (n,xn), ...), with its yield
  fission          //!< all fission, nu-total times sigma_f
};

//! One channel of one nuclide in one material
struct Channel {
  int i_mat;          //!< index of the nuclide within the material
  int i_nuclide;      //!< index into data::nuclides
  int i_reaction {-1}; //!< index into Nuclide::reactions_ (inelastic only)
  ChannelKind kind;
  Tag tag;
};

//==============================================================================
//! The registry partitions each material's neutron production into channels,
//! so that every contribution is assigned to exactly one tag and, at each
//! incident energy, to exactly one route:
//!
//! - elastic: evaluator where OpenMC uses a stationary target (A > 1, not a
//!   resonant scatterer, E' >= free_gas_threshold * kT, and inside the data
//!   range); otherwise sampler (free gas, resonance scattering);
//! - thermal elastic and inelastic: sampler;
//! - fission: evaluator.
//!
//! Production values need an XsProbe whose last total() call was at the
//! incident state, in the same material.
//==============================================================================

class ChannelRegistry {
public:
  //! Builds the channels of every material in model::materials
  ChannelRegistry();

  const vector<Channel>& channels(int32_t material) const
  {
    return channels_[material];
  }

  //! Route of a channel at incident energy E_in, with the probe at that state
  static Route route(const Channel& c, double E_in, const XsProbe& probe);

  //! Production nu_bar * Sigma_x [1/cm] of one channel at the probe's state
  static double production(const Channel& c, double density_mult,
    XsProbe& probe);

  //! Sum of production over the channels of a tag on a route, at the probe's
  //! state (material and incident energy of the last total() call)
  double production(Tag t, Route r, XsProbe& probe) const;

  //! Temperature kT [eV] OpenMC's scatter() uses for a nuclide's elastic
  //! scattering at the probe's state
  static double elastic_kT(int i_nuclide, const XsProbe& probe);

private:
  vector<vector<Channel>> channels_; //!< [material]
};

//==============================================================================
//! The sampler route: at an incident state, choose a sampler-route channel of
//! a tag with probability proportional to its production, and apply OpenMC's
//! own collision routine for it to the probe's scratch particle. Yields are
//! carried by the production, so the routine's own weight change and
//! secondary particles are discarded.
//==============================================================================

struct SamplerOutcome {
  double production {0.0}; //!< sum of sampler-route production [1/cm]
  double E {0.0};          //!< outgoing energy [eV] (if production > 0)
  Direction u;             //!< outgoing direction
  int i_nuclide {-1};
  ChannelKind kind {ChannelKind::elastic};
};

//! \param reg    Channel registry
//! \param t      Tag (only tag s has sampler-route channels)
//! \param u_in   Incident direction of travel
//! \param probe  At the incident state (total() called at E_in)
//! \param seed   Random number stream
SamplerOutcome sample_sampler_route(const ChannelRegistry& reg, Tag t,
  Direction u_in, XsProbe& probe, uint64_t* seed);

} // namespace rmc
} // namespace openmc

#endif // OPENMC_RMC_CHANNELS_H
