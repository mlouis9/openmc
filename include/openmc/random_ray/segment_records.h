#ifndef OPENMC_RANDOM_RAY_SEGMENT_RECORDS_H
#define OPENMC_RANDOM_RAY_SEGMENT_RECORDS_H

#include <cstdint>

#include "openmc/position.h"
#include "openmc/vector.h"

namespace openmc {

//==============================================================================
//! Active ray segments recorded during one random ray sweep, for the source
//! regions flagged with FlatSourceDomain::set_segment_recording().
//!
//! Residual Monte Carlo uses the segments crossing a region as a quadrature
//! of incident directions there (docs/source/methods/rmc.rst,
//! methods_rmc_volumes). For each segment it needs the direction, the chord
//! length, and the chord-averaged multigroup angular flux. For a flat source
//! the latter follows from what random ray already computes,
//!
//!   psi_bar_g = q_g + delta_psi_g / (Sigma_t,g * length),
//!
//! with q_g the region's reduced source and delta_psi_g the attenuation term.
//! Like the ray's angular flux, psi_bar carries 4 pi psi (an isotropic source
//! q contributes q directly).
//!
//! One buffer per thread; each record has a ray ID and a sequence number
//! within the ray, so that records can be put in a deterministic order.
//==============================================================================

struct SegmentRecords {
  int n_groups {0};
  vector<int64_t> region; //!< index into the domain's source regions
  vector<int64_t> ray;    //!< ray ID
  vector<int64_t> seq;    //!< order of the segment along its ray
  vector<double> u;       //!< [3 * n] direction
  vector<double> length;  //!< [n] chord length [cm]
  vector<double> psi_bar; //!< [n * G] chord-averaged 4 pi psi, by group

  int64_t size() const { return static_cast<int64_t>(region.size()); }

  //! Empty the records, keeping capacity
  void clear()
  {
    region.clear();
    ray.clear();
    seq.clear();
    u.clear();
    length.clear();
    psi_bar.clear();
  }
};

} // namespace openmc

#endif // OPENMC_RANDOM_RAY_SEGMENT_RECORDS_H
