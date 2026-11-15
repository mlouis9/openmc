//! \file flight.h
//! \brief Forward correction flights (Stage 5): each flight carries a signed
//! parent to its next collision and returns the neutrons that collision
//! produces, with signed weights.
//!
//! Theory: docs/source/methods/rmc.rst, methods_rmc_flights.

#ifndef OPENMC_RMC_FLIGHT_H
#define OPENMC_RMC_FLIGHT_H

#include <cstdint>

#include "openmc/particle.h"
#include "openmc/rmc/bank.h"
#include "openmc/rmc/xs_probe.h"
#include "openmc/vector.h"

namespace openmc {
namespace rmc {

//! Optional track-length tally of the flights: signed flux by material and
//! energy of the flight (the uncollided flux of the parents).
struct FlightTally {
  vector<int32_t> materials; //!< indices into model::materials (bins)
  vector<double> E_edges;    //!< ascending [eV]
  vector<double> flux;       //!< [material][energy bin], signed track length

  bool active() const { return !materials.empty() && E_edges.size() > 1; }
  void reset();
};

//! Optional per-child and per-region outputs of propagate(), for exact
//! cancellation (which needs each child's parent, path length, and
//! nuclide) and for the correction tally (a collision estimator of the
//! fission rate by field region)
struct FlightAux {
  bool children {false};       //!< record the per-child arrays
  vector<int64_t> parent;      //!< [children] index of the parent in the bank
  vector<double> path;         //!< [children] flight path length [cm]
  vector<int> nuclide;         //!< [children] index into data::nuclides
  vector<double> region_fission; //!< [R] signed sum of w Sigma_f / Sigma_t
                                 //!< at collisions (if sized to R)
  vector<double> region_nu_fission; //!< [R] the same with nu Sigma_f
};

//! Diagnostics of one propagate() call
struct FlightStats {
  int64_t flights {0};
  int64_t leaked {0};
  int64_t captured {0};
  int64_t scattered {0};
  int64_t fissions {0};
  int64_t collisions {0}; //!< all collisions (more than flights with history)
  double W_abs {0.0}; //!< sum |w| of the parents
};

//==============================================================================
//! Propagates a signed bank by N analog flights. Flight j selects parent i
//! with probability |w_i| / W_abs, flies it from (r_i, Omega_i) at E_i to its
//! next collision (or out through a vacuum boundary), and at the collision:
//!
//! - chooses a nuclide m with probability Sigma_t,m / Sigma_t;
//! - with probability sigma_s,m / sigma_t,m, scatters with OpenMC's
//!   scatter(): one child, tag s, weight factor its realized yield (the
//!   particle's weight after the call times one plus the integer-yield
//!   copies it banked, which share its outgoing state);
//! - otherwise, with probability sigma_f,m / sigma_a,m, fissions: one child
//!   from sample_fission_neutron(), tag f, weight factor nu_m(E_i) / k;
//! - otherwise captures: no child.
//!
//! The child's weight is sgn(w_i) W_abs / N times its factor. The expected
//! children of parent i are w_i Sigma_P(E_i) / Sigma_t(E_i) times the
//! normalized production law, by tag. No OpenMC tallies, roulette, or fission
//! bank are touched.
//==============================================================================

class FlightPropagator {
public:
  FlightPropagator();

  //! \param in       Parents (not modified)
  //! \param n        Number of flights
  //! \param k        Eigenvalue dividing fission production
  //! \param seed     Base seed; flight j uses its own stream
  //! \param out      Children are appended
  //! \param tally    Optional track-length tally (accumulated)
  //! \param aux      Optional per-child records and region tallies
  //! \param history  If true, a flight continues through its scatterings
  //!                 until it is captured, causes fission, or leaks, so
  //!                 that only fission children are banked; otherwise it
  //!                 ends at its first collision and banks the emission
  //!                 there (scattering or fission)
  FlightStats propagate(const SignedBank& in, int64_t n, double k,
    uint64_t seed, SignedBank& out, FlightTally* tally = nullptr,
    FlightAux* aux = nullptr, bool history = false);

private:
  //! Outcome of the collision at the probe's state, for direction u
  struct Collision {
    int kind {0}; //!< 0 capture, 1 scatter, 2 fission
    int nuclide {-1};
    double factor {0.0};
    double E {0.0};
    Direction u;
  };
  Collision collide(Direction u, double k, uint64_t* seed);

  Particle g_; //!< geometry particle
  XsProbe probe_;
};

} // namespace rmc
} // namespace openmc

#endif // OPENMC_RMC_FLIGHT_H
