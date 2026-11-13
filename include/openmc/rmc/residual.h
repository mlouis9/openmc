//! \file residual.h
//! \brief Residual samples (Stage 4): the CE production of the field minus
//! the field, at sampled outgoing states, as records with deferred weights.
//!
//! Theory: docs/source/methods/rmc.rst, methods_rmc_residual.

#ifndef OPENMC_RMC_RESIDUAL_H
#define OPENMC_RMC_RESIDUAL_H

#include <cstdint>

#include "openmc/position.h"
#include "openmc/rmc/backward.h"
#include "openmc/rmc/field.h"
#include "openmc/vector.h"

namespace openmc {
namespace rmc {

//! A value drawn from a proposal, with the density it was drawn from.
//! Weights are formed only from this density, never recomputed elsewhere.
template<typename T>
struct Draw {
  T value;
  double density;
};

//! Outgoing state of a residual particle
struct OutgoingState {
  double E;    //!< energy [eV]
  Direction u; //!< direction
};

//==============================================================================
//! Outgoing-state proposal: energy log-uniform on [E_min, E_max], direction
//! isotropic. Positive wherever the residual can be non-zero, as long as the
//! range covers the field's lifts and the CE emission.
//==============================================================================

class OutgoingProposal {
public:
  OutgoingProposal(double E_min, double E_max);
  Draw<OutgoingState> sample(uint64_t* seed) const;
  double density(const OutgoingState& s) const;

private:
  double E_min_, E_max_, log_ratio_;
};

//==============================================================================
//! Residual records: one per sample, SoA. A record holds what the weight
//! needs except the scalars (a, b, k), which enter only in weight_records():
//!
//!   w = ((a_s C_s + a_f C_f) / k^[tag f] - b_tag U) / (N density)
//!
//! where C_s and C_f are the CE production of the record's tag from the
//! field's Q_s and Q_f (the 1/k of fission production removed), U is the
//! field Q_tag at the record's state, and density is that of the whole draw
//! (position, energy, direction).
//==============================================================================

struct ResidualRecords {
  vector<double> r;       //!< [3n] position
  vector<double> u;       //!< [3n] direction
  vector<double> E;       //!< [n] energy [eV]
  vector<double> density; //!< [n] density of (r, E, u)
  vector<int> tag;        //!< [n] 0 = s, 1 = f
  vector<double> C_s;     //!< [n]
  vector<double> C_f;     //!< [n]
  vector<double> U;       //!< [n]

  int64_t size() const { return static_cast<int64_t>(E.size()); }
  void clear();
};

struct ResidualOptions {
  int n_directions {8};   //!< J stratified isotropic incident directions
  TailOptions tail;       //!< ending of each backward integral
  double E_min {1e-6};    //!< outgoing energy proposal range [eV]
  double E_max {2e7};
  double Ein_min {1e-5};  //!< incident energy proposal range for fission [eV]
  double Ein_max {2e7};
};

//==============================================================================
//! Draws residual samples of one tag with destinations uniform in the field's
//! ray source box, J stratified isotropic directions with omega_j = 4 pi / J,
//! and the evaluator-route channels of Stage 3: stationary-target elastic
//! scattering for tag s, fission for tag f. Other scattering channels
//! (inelastic, (n,xn), hydrogen) belong to the sampler route, which is not
//! yet implemented, so tag-s records omit their production.
//==============================================================================

class ResidualSampler {
public:
  ResidualSampler(const FrozenField& field, const ResidualOptions& opt);

  //! Append n records of tag t, drawn with random numbers derived from seed
  //! (sample i uses its own stream, so results do not depend on order).
  void sample(Tag t, int64_t n, uint64_t seed, ResidualRecords& out);

private:
  //! CE production of tag t at (r, E, Omega) in the located material, from
  //! Q_s and from Q_f, for one incident direction
  void production(Tag t, Position r, Direction omega_in, const OutgoingState& s,
    uint64_t* seed, double& c_s, double& c_f);

  const FrozenField& field_;
  ResidualOptions opt_;
  OutgoingProposal proposal_;
  BackwardFlux backward_;
  XsProbe probe_;
  // State of the current destination
  int32_t material_ {-1};
  double sqrtkT_ {0.0};
  double density_mult_ {1.0};
  // Scratch, reused between samples (no allocation in steady state)
  vector<double> energies_, psi_s_, psi_f_, image_density_;
  vector<int> image_nuclide_;
};

//! Weights of the records for amplitudes (a_s, a_f), (b_s, b_f), eigenvalue
//! k, and N samples of each record's tag.
void weight_records(const ResidualRecords& rec, int64_t N, double a_s,
  double a_f, double b_s, double b_f, double k, vector<double>& w);

//! J directions of a spherical Fibonacci set under a uniformly random
//! rotation: stratified, and each one marginally uniform on the sphere.
void stratified_directions(int J, uint64_t* seed, vector<Direction>& out);

} // namespace rmc
} // namespace openmc

#endif // OPENMC_RMC_RESIDUAL_H
