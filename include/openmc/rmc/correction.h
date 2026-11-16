//! \file correction.h
//! \brief Kernels of the correction tallies and of exact cancellation:
//! the field's CE reaction rate by region from backward characteristics,
//! the fission emission density of a nuclide, and the near point kernel of
//! an isotropic parent's next collision.
//!
//! Theory: docs/source/methods/rmc.rst, methods_rmc_tallies and
//! methods_rmc_cancellation.

#ifndef OPENMC_RMC_CORRECTION_H
#define OPENMC_RMC_CORRECTION_H

#include <cstdint>

#include "openmc/particle.h"
#include "openmc/position.h"
#include "openmc/rmc/backward.h"
#include "openmc/rmc/field.h"
#include "openmc/rmc/mg_control.h"
#include "openmc/rmc/tracer.h"
#include "openmc/rmc/xs_probe.h"
#include "openmc/vector.h"

namespace openmc {
namespace rmc {

//! The field part of the correction tally: for each region R, unbiased
//! estimates of X_{y,R} = int_R int int Sigma_f psi_y dE dOmega dV and the
//! same with nu Sigma_f, where psi_y = G Q_y is the CE angular flux of
//! emission component y in {s, f} (a = 1, no 1/k). Points are uniform in
//! the field's box, one isotropic direction each with a backward CE trace,
//! and n_E incident energies drawn from the region's lifted MG flux
//! (log-uniform where the region has none).
//! With MG tables, the same points also score the MG nu-fission rate of the
//! MG flux on the same backward paths (a control variate whose mean can be
//! computed once to high precision): nu_fission_mg, and the sum of squares
//! of the per-point difference nu_fission - nu_fission_mg.
struct FieldTallyResult {
  vector<double> fission;       //!< [R][2]
  vector<double> nu_fission;    //!< [R][2]
  vector<double> fission_sq;    //!< [R][2] sum of squares of per-point scores
  vector<double> nu_fission_mg; //!< [R][2] (with MG tables)
  vector<double> diff_sq;       //!< [R][2] (with MG tables)
};

//! CE nu-fission production of the field's first-flight flux per source tag,
//! X_y = <nu Sigma_f, G Q_y>, by backward points drawn from the field's own
//! lifted MG fission rate (region, group, energy within the group), n_E
//! energies per point on one isotropic path. var: variance of each mean.
void field_fission_production(const FrozenField& field, const MGTables& mg,
  int64_t n_points, int n_E, const TailOptions& tail, uint64_t seed,
  double X[2], double var[2]);

void field_reaction_tally(const FrozenField& field, int64_t n_points,
  int n_E, const TailOptions& tail, uint64_t seed, FieldTallyResult& out,
  const MGTables* mg = nullptr);

//! Outgoing-energy density [1/eV] of one fission neutron of nuclide
//! i_nuclide at incident energy E_in, as sample_fission() and
//! sample_fission_neutron() produce it: partial fission reactions weighted
//! by their cross sections. The probe's last total() call must be at E_in
//! in a material containing the nuclide.
double fission_chi(int i_nuclide, const XsProbe& probe, double E_out);

//! Density per unit volume, at x, of the next-collision site of a parent at
//! xp flying isotropically at energy E, restricted to flight paths shorter
//! than d_max: sum over the images of xp under the problem's reflective
//! box faces within d_max of x of exp(-tau) / (4 pi d^2), tau the optical
//! depth at E along the reflected path from x toward the image. Multiply by
//! Sigma_t(x) for the collision density. Only reflective boxes are
//! supported (lo, hi its faces).
//! Scratch geometry particle for near_kernel()
struct KernelScratch {
  Particle p;
  Particle& scratch() { return p; }
};

double near_kernel(Position x, Position xp, double E, double d_max,
  Position lo, Position hi, KernelScratch& scratch, XsProbe& probe,
  SegmentList& seg);

} // namespace rmc
} // namespace openmc

#endif // OPENMC_RMC_CORRECTION_H
