#ifndef OPENMC_CMFD_SOLVER_H
#define OPENMC_CMFD_SOLVER_H

#include "xtensor/xtensor.hpp"

namespace openmc {

//==============================================================================
// Constants
//==============================================================================

// For non-accelerated regions on coarse mesh overlay
constexpr int CMFD_NOACCEL {-1};

//! External source distribution projected onto CMFD mesh (if subcritical mode)
extern xt::xtensor<double, 4> external_src_cmfd;

//! Whether external source projection is active
extern bool external_src_projection_on;

//==============================================================================
// Non-member functions
//==============================================================================

void free_memory_cmfd();

} // namespace openmc

#endif // OPENMC_CMFD_SOLVER_H
