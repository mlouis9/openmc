//! \file residual.h
//! \brief Residual samples (Stage 4): the CE production of the field minus
//! the field, at sampled outgoing states, as records with deferred weights,
//! for the whole family of estimators (direction source, multigroup control,
//! mixing).
//!
//! Theory: docs/source/methods/rmc.rst, methods_rmc_residual.

#ifndef OPENMC_RMC_RESIDUAL_H
#define OPENMC_RMC_RESIDUAL_H

#include <cstdint>

#include "openmc/memory.h"
#include "openmc/position.h"
#include "openmc/rmc/backward.h"
#include "openmc/rmc/channels.h"
#include "openmc/rmc/field.h"
#include "openmc/rmc/mg_control.h"
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
// Proposals
//==============================================================================

//! How destinations are drawn. Every mode except box draws a region R with
//! probability pi_R, then a point uniform in R (needs region volumes).
enum class VoxelMode {
  box,         //!< uniform in the field's ray source box
  volume,      //!< pi_R proportional to V_R
  fissionable, //!< pi_R proportional to V_R in fissionable materials
  tabulated    //!< pi_R proportional to given weights w_R (V_R w_R if
               //!< weights_per_volume)
};

struct VoxelOptions {
  VoxelMode mode {VoxelMode::box};
  vector<double> weights; //!< [R] for tabulated
  bool weights_per_volume {false};
  //! Box mode only: a fraction skin_fraction of the destinations is drawn
  //! near the inside of a z-cylinder (centre skin_x, skin_y, radius
  //! skin_radius), with depth d below its surface log-uniform on
  //! [skin_dmin, skin_radius], uniform in angle and in z over the box. For
  //! features that live in a thin layer at an absorber's surface (resonance
  //! scattering of the moderator's unshielded flux). The density is the
  //! mixture with the box, so the estimate stays unbiased.
  double skin_fraction {0.0};
  double skin_x {0.0};
  double skin_y {0.0};
  double skin_radius {0.0};
  double skin_dmin {1e-5};
};

//! Destination proposal; density is per unit volume
class VoxelProposal {
public:
  VoxelProposal(const FrozenField& field, const VoxelOptions& opt);
  //! A destination, its region, and its density [1/cm^3]
  Draw<Position> sample(uint64_t* seed, int64_t& region) const;
  //! Density at a point in region R (box mode: at any point r)
  double density(int64_t region) const;
  double density_at(Position r, int64_t region) const;
  double pi(int64_t region) const { return pi_.empty() ? 0.0 : pi_[region]; }

private:
  const FrozenField& field_;
  VoxelOptions opt_;
  double V_box_;
  vector<double> pi_;  //!< [R] region probabilities (not for box)
  vector<double> cdf_; //!< [R]
};

struct OutgoingOptions {
  double beta {1.0};   //!< weight of the log-uniform component; 1 gives a
                       //!< pure log-uniform proposal
  double E_min {1e-5}; //!< log-uniform range [eV]
  double E_max {2e7};
  //! Optional tabulated energy proposal, replacing the emission spectrum in
  //! the (1 - beta) part: per region R, bin e of table_edges has probability
  //! table[R * n_bins + e] (normalized per region), log-uniform within the
  //! bin. A region whose row sums to zero uses the log-uniform part only.
  vector<double> table_edges;
  vector<double> table;
};

//! Outgoing-state proposal: direction isotropic; energy from (1 - beta)
//! times the region's emission spectrum of the tag (group by Q_{t,R,g},
//! then the lift within the group) plus beta times log-uniform. Positive
//! wherever the residual can be non-zero when beta > 0 and the range covers
//! the field's lifts and the CE emission.
class OutgoingProposal {
public:
  OutgoingProposal(const FrozenField& field, const OutgoingOptions& opt);
  Draw<OutgoingState> sample(Tag t, int64_t region, uint64_t* seed) const;
  double density(Tag t, int64_t region, const OutgoingState& s) const;

private:
  double energy_density(Tag t, int64_t region, double E) const;
  const FrozenField& field_;
  OutgoingOptions opt_;
  double log_ratio_;
  int n_tab_ {0};             //!< bins of the tabulated proposal (0: none)
  vector<double> tab_cdf_;    //!< [R * n_tab] per-region cumulative
  vector<double> tab_sum_;    //!< [R] row sums (before normalization)
};

//! Incident-energy proposal of the fission evaluator and the sampler route
enum class IncidentMode {
  log_uniform, //!< log-uniform on [Ein_min, Ein_max]
  flux //!< (1 - beta_in) times a group drawn by w_g' (below), then the flux
       //!< shape h_phi within it, plus beta_in times log-uniform. Weights:
       //!< nu Sigma_f,g' psi^MG_g' (fission) or psi^MG_g' (sampler route),
       //!< from the direction's own MG flux (its path, or its sweep segment)
};

struct IncidentOptions {
  IncidentMode mode {IncidentMode::log_uniform};
  double beta {0.1};
  double E_min {1e-5};
  double E_max {2e7};
};

//==============================================================================
// Field sweep: one random ray transport sweep from the field
//==============================================================================

//! Segment records of one transport sweep from Q_s and one from Q_f with the
//! same rays, indexed by field region (CSR). psi_s, psi_f are chord-averaged
//! group-integrated angular fluxes per unit solid angle (ascending groups).
//! Quadrature weights are omega_j = 4 pi length_j / (rho V_R), with rho the
//! sweep's active track length per unit volume and V_R the field's volume.
struct SweepRecords {
  int n_groups {0};
  double rho {0.0};
  vector<int64_t> offset; //!< [R + 1]
  vector<Direction> u;    //!< direction of travel of each segment
  vector<double> length;
  vector<double> psi_s, psi_f; //!< [n * G]

  int64_t size() const { return static_cast<int64_t>(length.size()); }
  int64_t count(int64_t region) const
  {
    return offset[region + 1] - offset[region];
  }
};

//! The records of the last field sweep through the C API (src/rmc/capi.cpp)
extern unique_ptr<SweepRecords> loaded_sweep;

//==============================================================================
// Records
//==============================================================================

//! Residual records: one per sample (density branch) plus the sampler-route
//! particles, SoA. Every term is stored per source tag y (from Q_s and Q_f),
//! so the scalars a, b, k, c, lambda enter only in weight_records():
//!
//!   density branch, tag t:
//!     w = { sum_y a_y (C_y - c M_y + c R_y) / k^[t=f]
//!           - b_t [ (1 - lambda) U + lambda sum_y a_y R_y / k^[t=f] ] }
//!         / (N density)
//!   sampler branch:  w = (sum_y a_y C_y + Z) / k^[t=f] / (N density)
//!
//! C_y: CE production (the channels of the record's branch); M_y: MG control
//! on the same directions; R_y: the control's mean, restored (L MG-only
//! directions, or the sweep's region emission Q-hat); U: the field.
struct ResidualRecords {
  vector<double> r;       //!< [3n] position
  vector<double> u;       //!< [3n] direction
  vector<double> E;       //!< [n] energy [eV]
  vector<double> density; //!< [n] density of the draw (per unit volume for
                          //!< sampler records, of (r, E, u) otherwise)
  vector<int> tag;        //!< [n] 0 = s, 1 = f
  vector<int> sampler;    //!< [n] 1 for sampler-route particles
  vector<int> kind;       //!< [n] 0 density branch, 1 sampler route on a
                          //!< backward path, 2 region term of the flux-
                          //!< difference control
  vector<int> channel;    //!< [n] ChannelKind of a sampled particle, or -1
  vector<double> E_in;    //!< [n] incident energy of a sampled particle's
                          //!< collision [eV], or 0
  vector<int> nuclide;    //!< [n] its nuclide (data::nuclides), or -1
  vector<double> C_s, C_f, M_s, M_f, R_s, R_f, U;
  vector<double> Z; //!< [n] amplitude-free part of a sampled
                    //!< particle's weight (the flux-difference control)

  int64_t size() const { return static_cast<int64_t>(E.size()); }
  void clear();
};

enum class DirectionMode {
  isotropic, //!< J stratified isotropic directions, omega_j = 4 pi / J
  sweep      //!< the sweep segments in the destination's region
};

//! How the MG fission control handles the incident energy
enum class ControlEnergy {
  exact,      //!< exact sum over incident groups
  same_energy //!< estimated at the CE estimate's sampled E' (shares its noise)
};

//! Options of the sampler route (channels without a density evaluator, or
//! every channel with all_sampler). Theory: docs/source/methods/rmc.rst,
//! methods_rmc_sampler_control.
enum class SamplerControl {
  none,  //!< weight from the full CE flux psi
  region //!< flux-difference split: weight from psi - phi_R h_phi / 4 pi on
         //!< the backward path, plus a region term sampled forward from
         //!< phi_R h_phi / 4 pi (no backtrace, no density)
};

//! Options of the sampler route (theory: docs/source/methods/rmc.rst,
//! methods_rmc_sampler_control)
struct SamplerOptions {
  bool all_sampler {false};    //!< every channel on the sampler route
  int energies {1};            //!< K incident energies per direction
  bool common_numbers {false}; //!< the K outgoing states share random numbers
  SamplerControl control {SamplerControl::none};
  int region_samples {1}; //!< forward samples of the region term per sample
  bool net {false};       //!< study only, biased: one outgoing state from
                          //!< the K, chosen by |weight|, carries their net
};

struct ResidualOptions {
  DirectionMode directions {DirectionMode::isotropic};
  int n_directions {8}; //!< J; for sweep directions, the number of segments
                        //!< subsampled with probability proportional to
                        //!< omega_j (0 = every segment in the region)
  bool control {false}; //!< compute M and R (needs the MG tables)
  int n_control_directions {0}; //!< L MG-only directions (isotropic)
  ControlEnergy control_energy {ControlEnergy::exact};
  bool sampler_route {true}; //!< tag s: sampler-route particles
  TailOptions tail;
  VoxelOptions voxel;
  OutgoingOptions outgoing;
  IncidentOptions incident;
  SamplerOptions sampler;
  //! Forward-cone directions (isotropic direction source only): a fraction
  //! of the J incident directions is drawn from a von Mises-Fisher density
  //! about the outgoing direction, with concentration kappa(E) rising
  //! log-linearly from 0 at cone_E0 to cone_kappa at cone_E1 (outgoing
  //! energy E), for kernels that are forward-peaked at high energy. Every
  //! direction is weighted by the balance heuristic over both components,
  //! 1 / (J_iso / 4 pi + J_cone f(Omega' . Omega)), so the estimate stays
  //! unbiased for any kappa.
  double cone_fraction {0.0};
  double cone_kappa {0.0};
  double cone_E0 {1e4};
  double cone_E1 {1e7};
};

//==============================================================================
//! Draws residual samples of one tag. Each sample: a destination from the
//! voxel proposal, an outgoing state, the field there (U), and over the
//! directions of the direction source the CE production of the evaluator
//! channels (C), the MG control (M) and its restored mean (R); and, for the
//! sampler-route channels of the tag, K particles per direction (term I) and,
//! with the flux-difference control, M_II region-term particles (term II).
//==============================================================================

class ResidualSampler {
public:
  ResidualSampler(const FrozenField& field, const ResidualOptions& opt,
    const MGTables* tables = nullptr, const SweepRecords* sweep = nullptr);

  //! Append the records of n samples of tag t (sample i uses its own
  //! stream, so results do not depend on order).
  void sample(Tag t, int64_t n, uint64_t seed, ResidualRecords& out);

  //! Diagnostics: per direction of the last sample() call, the CE and MG
  //! estimates of q_j (from Q_s + Q_f) at the sample's outgoing state, for
  //! the same-path correlation study (G3.7). Only kept if requested.
  void keep_directions(bool on) { keep_ = on; }
  const vector<double>& kept_ce() const { return kept_ce_; }
  const vector<double>& kept_mg() const { return kept_mg_; }

private:
  //! Destination state
  struct Destination {
    Position r;
    int64_t region;
    int32_t material;
    double sqrtkT;
    double density_mult;
  };

  //! A sampled particle of one direction: C per source tag and Z, before
  //! the direction weight
  struct SampledParticle {
    OutgoingState state;
    double C[2] {0.0, 0.0};
    double Z {0.0};
    int channel {-1};
    double E_in {0.0}; //!< incident energy of the sampled collision [eV]
    int nuclide {-1};  //!< nuclide of the sampled collision
  };

  //! Terms of one direction (per source tag y = s, f)
  struct DirectionTerms {
    double C[2] {0.0, 0.0};
    double M[2] {0.0, 0.0};
    vector<SampledParticle> sampled;
  };

  //! MG scalar flux of the region at E': phi_{R,g(E')} h_phi(E') [per eV]
  double region_flux(int64_t region, double E) const;

  //! Forward samples of the region term (flux-difference control)
  void region_term(
    Tag t, const Destination& d, uint64_t* seed, vector<SampledParticle>& out);

  void direction(Tag t, const Destination& d, Direction omega_in,
    const OutgoingState& s, const double* psi_seg_s, const double* psi_seg_f,
    uint64_t* seed, DirectionTerms& out);

  //! Restored control mean (isotropic): L MG-only directions
  void control_mean(Tag t, const Destination& d, const OutgoingState& s,
    uint64_t* seed, double R[2]);

  //! MG control of one direction from its MG fluxes; E_in, p_in are the
  //! fission estimate's incident energy and density (same-energy form)
  void mg_terms(Tag t, const Destination& d, const OutgoingState& s,
    const vector<double>& mg_s, const vector<double>& mg_f, double E_in,
    double p_in, double M[2]) const;

  //! Incident-energy draw with weights from MG fluxes (mg_s + mg_f)
  Draw<double> incident(Tag t, const Destination& d, const vector<double>& mg_s,
    const vector<double>& mg_f, uint64_t* seed);
  double incident_density(Tag t, const Destination& d,
    const vector<double>& mg_s, const vector<double>& mg_f, double E) const;

  bool needs_mg_flux(Tag t) const;

  static void push_sampled(ResidualRecords& out, Position r,
    const SampledParticle& q, double omega, double density, Tag t, int kind);

  const FrozenField& field_;
  ResidualOptions opt_;
  const MGTables* tables_;
  const SweepRecords* sweep_;
  VoxelProposal voxel_;
  OutgoingProposal outgoing_;
  ChannelRegistry registry_;
  BackwardFlux backward_;
  XsProbe probe_;

  bool keep_ {false};
  vector<double> kept_ce_, kept_mg_;

  // Scratch, reused between samples (no allocation in steady state)
  vector<double> energies_, psi_s_, psi_f_, mg_s_, mg_f_, image_density_;
  vector<int> image_nuclide_;
  vector<double> seg_s_, seg_f_, w_g_;
  vector<int64_t> picks_;
  vector<Draw<double>> sampler_E_;
  vector<SampledParticle> candidates_, region_particles_;
};

//! Weights of records for amplitudes (a_s, a_f), (b_s, b_f), eigenvalue k,
//! control coefficient c, mixing lambda, and N samples of each tag.
struct WeightScalars {
  double a[2] {1.0, 1.0};
  double b[2] {1.0, 1.0};
  double k {1.0};
  double c {0.0};
  double lambda {0.0};
};
void weight_records(const ResidualRecords& rec, int64_t N,
  const WeightScalars& x, vector<double>& w);

//! J spherical Fibonacci points under a uniformly random rotation
void stratified_directions(int J, uint64_t* seed, vector<Direction>& out);

} // namespace rmc
} // namespace openmc

#endif // OPENMC_RMC_RESIDUAL_H
