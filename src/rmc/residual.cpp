#include "openmc/rmc/residual.h"

#include <cmath>

#include "openmc/cell.h"
#include "openmc/constants.h"
#include "openmc/error.h"
#include "openmc/geometry.h"
#include "openmc/material.h"
#include "openmc/nuclide.h"
#include "openmc/random_lcg.h"
#include "openmc/rmc/production.h"

namespace openmc {
namespace rmc {

//==============================================================================
// Proposals and directions
//==============================================================================

OutgoingProposal::OutgoingProposal(double E_min, double E_max)
  : E_min_(E_min), E_max_(E_max), log_ratio_(std::log(E_max / E_min))
{
  if (!(E_min > 0.0 && E_max > E_min))
    fatal_error("RMC residual: invalid outgoing energy range.");
}

Draw<OutgoingState> OutgoingProposal::sample(uint64_t* seed) const
{
  OutgoingState s;
  s.E = E_min_ * std::exp(log_ratio_ * prn(seed));
  double mu = 2.0 * prn(seed) - 1.0;
  double phi = 2.0 * PI * prn(seed);
  double st = std::sqrt(std::max(0.0, 1.0 - mu * mu));
  s.u = {st * std::cos(phi), st * std::sin(phi), mu};
  return {s, density(s)};
}

double OutgoingProposal::density(const OutgoingState& s) const
{
  if (s.E < E_min_ || s.E > E_max_)
    return 0.0;
  return 1.0 / (s.E * log_ratio_) / (4.0 * PI);
}

void stratified_directions(int J, uint64_t* seed, vector<Direction>& out)
{
  // Spherical Fibonacci points: z stratified on [-1, 1], golden-angle azimuth
  out.resize(J);
  const double golden = PI * (3.0 - std::sqrt(5.0));
  for (int j = 0; j < J; ++j) {
    double z = 1.0 - (2.0 * j + 1.0) / J;
    double rho = std::sqrt(std::max(0.0, 1.0 - z * z));
    out[j] = {rho * std::cos(golden * j), rho * std::sin(golden * j), z};
  }
  // Uniformly random rotation (Arvo's method), so that every direction is
  // marginally uniform and the set is unbiased for any integrand
  double x1 = prn(seed), x2 = prn(seed), x3 = prn(seed);
  double theta = 2.0 * PI * x1, phi = 2.0 * PI * x2, z = x3;
  double c = std::cos(theta), s = std::sin(theta);
  Direction v {
    std::cos(phi) * std::sqrt(z), std::sin(phi) * std::sqrt(z), std::sqrt(1 - z)};
  for (auto& d : out) {
    // Rotate about z by theta, then reflect with the Householder matrix of v
    Direction a {c * d.x - s * d.y, s * d.x + c * d.y, d.z};
    double vd = v.dot(a);
    d = {2 * vd * v.x - a.x, 2 * vd * v.y - a.y, 2 * vd * v.z - a.z};
  }
}

void ResidualRecords::clear()
{
  r.clear();
  u.clear();
  E.clear();
  density.clear();
  tag.clear();
  C_s.clear();
  C_f.clear();
  U.clear();
}

//==============================================================================
// ResidualSampler
//==============================================================================

ResidualSampler::ResidualSampler(
  const FrozenField& field, const ResidualOptions& opt)
  : field_(field), opt_(opt), proposal_(opt.E_min, opt.E_max),
    backward_(field)
{
  if (opt.n_directions < 1)
    fatal_error("RMC residual: at least one incident direction is needed.");
}

void ResidualSampler::production(Tag t, Position r, Direction omega_in,
  const OutgoingState& s, uint64_t* seed, double& c_s, double& c_f)
{
  c_s = c_f = 0.0;
  if (material_ == MATERIAL_VOID)
    return;
  const Material& mat = *model::materials[material_];
  const double kT = sqrtkT_ * sqrtkT_;
  const Direction u_back = -omega_in;

  if (t == Tag::s) {
    // Stationary-target elastic, every nuclide with A > 1: the incident
    // energy that scatters into E at mu0 differs by nuclide, so collect them
    // and evaluate psi at all of them along one path
    const double mu0 = omega_in.dot(s.u);
    energies_.clear();
    image_density_.clear();
    image_nuclide_.clear();
    for (int i = 0; i < static_cast<int>(mat.nuclide_.size()); ++i) {
      auto img = elastic_stationary(mat.nuclide_[i], kT, s.E, mu0);
      if (img.density > 0.0) {
        energies_.push_back(img.E_in);
        image_density_.push_back(img.density);
        image_nuclide_.push_back(i);
      }
    }
    if (energies_.empty())
      return;
    backward_.evaluate_tags(r, u_back, energies_, opt_.tail, seed, psi_s_, psi_f_);
    for (size_t k = 0; k < image_nuclide_.size(); ++k) {
      int i = image_nuclide_[k];
      probe_.total(material_, energies_[k], sqrtkT_, density_mult_);
      double sigma_el = mat.atom_density(i, density_mult_) *
                        probe_.micro_elastic(mat.nuclide_[i]);
      // Density per unit mu0 and azimuth: 1 / (2 pi)
      double f = sigma_el * image_density_[k] / (2.0 * PI);
      c_s += f * psi_s_[k];
      c_f += f * psi_f_[k];
    }
  } else {
    // Fission: one incident energy E' drawn log-uniformly; isotropic emission
    double log_ratio = std::log(opt_.Ein_max / opt_.Ein_min);
    double E_in = opt_.Ein_min * std::exp(log_ratio * prn(seed));
    double g = 1.0 / (E_in * log_ratio);
    energies_.assign(1, E_in);
    backward_.evaluate_tags(r, u_back, energies_, opt_.tail, seed, psi_s_, psi_f_);
    probe_.total(material_, E_in, sqrtkT_, density_mult_);
    double emission = 0.0; // sum_i N_i nu_i sigma_f,i chi_i(E | E')
    for (int i = 0; i < static_cast<int>(mat.nuclide_.size()); ++i) {
      int i_nuc = mat.nuclide_[i];
      const Nuclide& nuc = *data::nuclides[i_nuc];
      if (!nuc.fissionable_)
        continue;
      double N = mat.atom_density(i, density_mult_);
      double nu = nuc.nu(E_in, Nuclide::EmissionMode::total);
      if (nuc.has_partial_fission_) {
        // As sample_fission(): each partial reaction with its share
        for (const Reaction* rx : nuc.fission_rx_) {
          double xs = probe_.micro_reaction(i_nuc, *rx);
          if (xs > 0.0)
            emission +=
              N * nu * xs * fission_energy_density(i_nuc, *rx, E_in, s.E);
        }
      } else {
        double xs = probe_.micro_fission(i_nuc);
        if (xs > 0.0)
          emission += N * nu * xs *
                      fission_energy_density(i_nuc, *nuc.fission_rx_[0], E_in, s.E);
      }
    }
    double f = emission / (4.0 * PI * g);
    c_s = f * psi_s_[0];
    c_f = f * psi_f_[0];
  }
}

void ResidualSampler::sample(
  Tag t, int64_t n, uint64_t seed, ResidualRecords& out)
{
  const Position lo = field_.box_lower_left();
  const Position w = field_.box_upper_right() - lo;
  const double V = w.x * w.y * w.z;
  const int J = opt_.n_directions;
  const double omega = 4.0 * PI / J;
  vector<Direction> dirs;
  GeometryState gs;

  for (int64_t i = 0; i < n; ++i) {
    // Each sample has its own stream, so results do not depend on order
    uint64_t s = future_seed(static_cast<uint64_t>(i) * 1000003ULL, seed);

    // Destination uniform in the box
    Position r {lo.x + w.x * prn(&s), lo.y + w.y * prn(&s), lo.z + w.z * prn(&s)};
    gs.init_from_r_u(r, {0.0, 0.0, 1.0});
    if (!exhaustive_find_cell(gs))
      fatal_error("RMC residual: destination outside the geometry.");
    const Cell& cell = *model::cells[gs.lowest_coord().cell()];
    int32_t inst = gs.cell_instance();
    material_ = cell.material(inst);
    sqrtkT_ = cell.sqrtkT(inst);
    density_mult_ = cell.density_mult(inst);
    int64_t region = field_.locate(gs);

    // Outgoing state, and the field at it
    auto draw = proposal_.sample(&s);
    const OutgoingState& st = draw.value;
    double U = field_.evaluate(region, t, st.E);

    // CE production over J incident directions
    double C_s = 0.0, C_f = 0.0;
    stratified_directions(J, &s, dirs);
    for (const auto& omega_in : dirs) {
      double cs, cf;
      production(t, r, omega_in, st, &s, cs, cf);
      C_s += omega * cs;
      C_f += omega * cf;
    }

    out.r.insert(out.r.end(), {r.x, r.y, r.z});
    out.u.insert(out.u.end(), {st.u.x, st.u.y, st.u.z});
    out.E.push_back(st.E);
    out.density.push_back(draw.density / V);
    out.tag.push_back(static_cast<int>(t));
    out.C_s.push_back(C_s);
    out.C_f.push_back(C_f);
    out.U.push_back(U);
  }
}

void weight_records(const ResidualRecords& rec, int64_t N, double a_s,
  double a_f, double b_s, double b_f, double k, vector<double>& w)
{
  w.resize(rec.size());
  for (int64_t i = 0; i < rec.size(); ++i) {
    bool fission = rec.tag[i] == static_cast<int>(Tag::f);
    double high = a_s * rec.C_s[i] + a_f * rec.C_f[i];
    if (fission)
      high /= k;
    double low = (fission ? b_f : b_s) * rec.U[i];
    w[i] = (high - low) / (N * rec.density[i]);
  }
}

} // namespace rmc
} // namespace openmc
