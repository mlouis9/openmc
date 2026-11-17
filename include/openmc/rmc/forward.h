//! Continuous correlated forward residual for a homogeneous reflective box.
#ifndef OPENMC_RMC_FORWARD_H
#define OPENMC_RMC_FORWARD_H

#include <map>

#include "openmc/rmc/bank.h"
#include "openmc/rmc/mg_control.h"

namespace openmc {
namespace rmc {

//! Sample a CE-minus-reconstructed-MG residual, then the analytic MG-field
//! remainder. Only one spatially uniform region/material in a reflective
//! box is supported. There is no source projection or cancellation.
//! part: 0 full residual, 1 CE production, 2 MG production, 3 CE-MG,
//! 4 analytic MG-field remainder. Diagnostics: sums CE, MG, |difference|,
//! |remainder|, number of null samples. Fission includes 1/k.
void forward_homogeneous(const FrozenField& field, const MGTables& mg,
  Tag tag, int64_t n, const double a[2], double b, double k, uint64_t seed,
  SignedBank& out, double diagnostics[5], int part = 0);

//! A preliminary RESOLVE energy representation (Mylotte and Forget): per
//! material index m and group g, an orthonormal basis of flux-per-lethargy
//! functions U (n fine bins x L lanes, piecewise constant on the bins given
//! by edges), with the eigenvalues D of the projected total cross section,
//! so that the uncollided flux in m attenuates lane by lane, exp(-D_k l).
//! Crossing matrices map lane coefficients between materials; source
//! vectors give the lane coefficients of a region's emission lift.
struct ResolveBlock {
  vector<double> edges; //!< n + 1 ascending energies [eV]
  int n {0}, L {0};
  vector<double> U, D;  //!< U row-major n x L
  //! Optional band structure (subgroup libraries): band of every bin, band
  //! lethargy widths, and per band its bins with the cumulative draw weights
  //! (lethargy width times a per-bin weight, the elastic cross section)
  vector<int> band;
  vector<double> band_mass;
  vector<vector<int>> band_bins;
  vector<vector<double>> band_cdf;
};
struct ResolveLibrary {
  int n_mat {0}, n_groups {0};
  vector<ResolveBlock> block; //!< index m * n_groups + g
  std::map<int64_t, vector<double>> cross;  //!< key (from, to, g): L_to x L_from
  std::map<int64_t, vector<double>> source; //!< key (region, tag, g): L of the region's material
  //! Cumulative band masses a_k of each source (band libraries), built lazily
  mutable std::map<int64_t, vector<double>> source_cdf;
  void clear()
  {
    block.clear();
    cross.clear();
    source.clear();
    source_cdf.clear();
    n_mat = n_groups = 0;
  }
  const ResolveBlock* get(int m, int g) const
  {
    if (m < 0 || m >= n_mat || g < 0 || g >= n_groups)
      return nullptr;
    const auto& b = block[static_cast<size_t>(m) * n_groups + g];
    return b.L > 0 ? &b : nullptr;
  }
  static int64_t cross_key(int from, int to, int g)
  {
    return (static_cast<int64_t>(from) * 1000 + to) * 100 + g;
  }
  static int64_t source_key(int64_t region, int tag, int g)
  {
    return (region * 10 + tag) * 100 + g;
  }
};
//! Library slots: 0 the partner, 1 the proposal bands (forward_points part
//! 8). With slot -1, the slot selected by resolve_select() (for loading).
ResolveLibrary& resolve_library(int slot = -1);
void resolve_select(int slot);

//! Points of the continuous correlated forward residual (forward_points)
struct ForwardPoints {
  vector<Position> r;
  vector<Direction> u;
  vector<double> E;
  vector<int64_t> region;
  vector<int32_t> material;
  vector<double> w_ce, w_mg; //!< the two terms of the signed weight
  vector<int> group_in;      //!< source group
  //! Sums of the weights by channel: key ((nuclide + 2) * 1000 + MT) * 100 +
  //! material * 10 + outgoing group. CE: nuclide index and MT (2 elastic, 18
  //! fission, others non-elastic). Multigroup (nuclide -1): MT 990 the
  //! elastic share, 991 the remainder, 992 fission.
  std::map<int64_t, double> channel;
  void add(int nuclide, int mt, int32_t material, int group, double w)
  {
    if (group < 0 || material < 0)
      return;
    channel[((static_cast<int64_t>(nuclide) + 2) * 1000 + mt) * 100 + material * 10 + group] += w;
  }
  void clear()
  {
    channel.clear();
    r.clear();
    u.clear();
    E.clear();
    region.clear();
    material.clear();
    w_ce.clear();
    w_mg.clear();
    group_in.clear();
  }
};

//! The field's first-collision emission of one tag, CE minus a multigroup
//! reconstruction, as signed points on any field (see forward.cpp). part 0:
//! the density part (CE stationary elastic or fission against the paired
//! multigroup emission); 1: CE non-elastic scattering, unpaired; 2: the
//! unpaired share of the multigroup scattering. lambda: probability of the CE
//! component of the mixture proposal. Diagnostics: sums of w_ce, w_mg,
//! |w_ce - w_mg|, |w_ce| + |w_mg|, null samples, paths without collision;
//! then (part 0, weighted like w) for alternative multigroup partners:
//! |CE - m1s|, |CE - m1|, |CE - p1|, |CE - p0|, |p0 - MG|, |p1 - p0|,
//! |m1 - p1|, m1, m1s, p0 - MG. m1s and m1 apply the CE kernel to the
//! multigroup-attenuated group flux, with the source region's emission lift
//! or the collision region's flux lift; p1 and p0 are the P0+P1 and P0
//! Legendre truncations of m1 in the incident-outgoing cosine (tag s); for
//! tag f, p0 is m1 averaged over the incident energy. Part 4 also gives
//! |CE - pn| (16): pn is the multigroup partner with a P1 elastic transfer,
//! (1 + 3 mu mubar(g -> g')), mubar from OpenMC's elastic sampling.
//! Part 5: the RESOLVE partner (the CE law applied to the uncollided flux
//! carried in RESOLVE lanes); diagnostics 17-19: |CE - partner| with the
//! exact angular law, its P1, P3 (and 20: P7) truncation (tag s). Part 6
//! banks the P3 partner, part 7 the exact one without the moment diagnostics.
//! Part 8 (tag s): as part 7 with a third proposal component drawn from the
//! band (subgroup) library of slot 1, with probability 0.45 (CE: lambda, MG:
//! the rest).
void forward_points(const FrozenField& field, const MGTables& mg, Tag tag, int part,
  int64_t n, const double a[2], double k, double lambda, uint64_t seed,
  ForwardPoints& out, double diagnostics[24]);

//! Band first-flight sweep (see forward.cpp): region-integrated uncollided
//! flux of the field's emission in every band of the band library in slot.
//! Phi is [regions][offset[G]], band k of group g at offset[g] + k.
void band_sweep(const FrozenField& field, int slot, int64_t n_rays, double dead,
  double active, uint64_t seed, vector<double>& Phi, vector<int>& offset, int n_sample = 0);

} // namespace rmc
} // namespace openmc
#endif
