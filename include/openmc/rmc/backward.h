//! \file backward.h
//! \brief The backward CE characteristic: angular flux at a destination due
//! to the frozen field, eq. rmc-backward-psi.

#ifndef OPENMC_RMC_BACKWARD_H
#define OPENMC_RMC_BACKWARD_H

#include <cstdint>

#include "openmc/position.h"
#include "openmc/rmc/field.h"
#include "openmc/rmc/tracer.h"
#include "openmc/rmc/xs_probe.h"
#include "openmc/vector.h"

namespace openmc {
namespace rmc {

//! How the integral along the backward path ends. Theory:
//! docs/source/methods/rmc.rst, methods_rmc_tails.
enum class Tail {
  vacuum,      //!< to the vacuum boundary: exact, no added noise; the path
               //!< must reach vacuum (refused in fully reflective problems)
  roulette,    //!< unbiased Russian roulette once exp(-tau) < cutoff
  fixed_length //!< stop after a fixed distance (the dead length): biased by
               //!< the attenuated contribution beyond it
};

struct TailOptions {
  Tail mode {Tail::roulette};
  double cutoff {3.3546262790251185e-04}; //!< roulette threshold, e^-8
  double length {0.0};                    //!< fixed_length: distance [cm]
  int chunk {32};                         //!< segments traced at a time
  int64_t max_segments {10'000'000};      //!< hard limit per path
};

//! Diagnostics of one backward path
struct PathStats {
  int64_t segments {0};   //!< segments traced
  int64_t lookups {0};    //!< cross-section lookups
  bool hit_limit {false}; //!< stopped at max_segments (result incomplete)
};

//==============================================================================
//! Evaluates psi(r*, E', Omega') for several energies along one backward
//! path. Holds the tracer and the cross-section probe, so it is per-thread
//! state.
//==============================================================================

class BackwardFlux {
public:
  explicit BackwardFlux(const FrozenField& field)
    : field_(field), tracer_(field)
  {}

  //! CE angular flux [per unit energy, solid angle, area, time, in the
  //! field's normalization] at r for incident direction Omega' = -u_back, at
  //! each energy in E, due to the field a_s Q_s + a_f Q_f.
  //! \param seed  Random number seed, used only by the roulette tail
  void evaluate(Position r, Direction u_back, const vector<double>& E,
    double a_s, double a_f, const TailOptions& opt, uint64_t* seed,
    vector<double>& psi, PathStats* stats = nullptr);

  //! The same, with the angular fluxes due to Q_s and to Q_f returned
  //! separately (one path, shared transmissions and roulette decisions), so
  //! that the field amplitudes can be applied later (deferred weights).
  void evaluate_tags(Position r, Direction u_back, const vector<double>& E,
    const TailOptions& opt, uint64_t* seed, vector<double>& psi_s,
    vector<double>& psi_f, PathStats* stats = nullptr);

private:
  const FrozenField& field_;
  CharacteristicTracer tracer_;
  XsProbe probe_;
  SegmentList chunk_;
  vector<double> T_; //!< transmission exp(-tau) per energy
  vector<char> alive_;
  vector<double> psi_s_, psi_f_;
};

//! Lambda(Sigma, l) = (1 - exp(-Sigma l)) / Sigma, and l for Sigma l -> 0,
//! without cancellation for thin segments.
double attenuation_integral(double sigma, double length);

} // namespace rmc
} // namespace openmc

#endif // OPENMC_RMC_BACKWARD_H
