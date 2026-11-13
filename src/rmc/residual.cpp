#include "openmc/rmc/residual.h"

#include <algorithm>
#include <cmath>

#include "openmc/cell.h"
#include "openmc/constants.h"
#include "openmc/error.h"
#include "openmc/geometry.h"
#include "openmc/material.h"
#include "openmc/mesh.h"
#include "openmc/nuclide.h"
#include "openmc/random_lcg.h"
#include "openmc/reaction.h"
#include "openmc/rmc/production.h"

namespace openmc {
namespace rmc {

namespace {

//! Locate a point: its region and the cell instance's material state
bool locate(const FrozenField& field, Position r, int64_t& region,
  int32_t& material, double& sqrtkT, double& density_mult)
{
  GeometryState gs;
  gs.init_from_r_u(r, {0.0, 0.0, 1.0});
  if (!exhaustive_find_cell(gs))
    return false;
  const Cell& cell = *model::cells[gs.lowest_coord().cell()];
  int32_t inst = gs.cell_instance();
  material = cell.material(inst);
  sqrtkT = cell.sqrtkT(inst);
  density_mult = cell.density_mult(inst);
  region = field.locate(gs);
  return true;
}

int64_t draw_index(const vector<double>& cdf, uint64_t* seed)
{
  double x = prn(seed) * cdf.back();
  auto it = std::upper_bound(cdf.begin(), cdf.end(), x);
  return std::min<int64_t>(it - cdf.begin(), cdf.size() - 1);
}

} // namespace

//==============================================================================
// Voxel proposal
//==============================================================================

VoxelProposal::VoxelProposal(const FrozenField& field, const VoxelOptions& opt)
  : field_(field), opt_(opt)
{
  Position w = field.box_upper_right() - field.box_lower_left();
  V_box_ = w.x * w.y * w.z;
  if (opt.mode == VoxelMode::box)
    return;
  if (!field.has_volumes())
    fatal_error("RMC residual: region-based destinations need region volumes "
                "(load the field with a volume file).");
  const int64_t R = field.n_regions();
  pi_.assign(R, 0.0);
  for (int64_t r = 0; r < R; ++r) {
    double V = field.volume(r);
    switch (opt.mode) {
    case VoxelMode::volume:
      pi_[r] = V;
      break;
    case VoxelMode::fissionable: {
      int32_t m = field.region_material(r);
      pi_[r] =
        (m != MATERIAL_VOID && model::materials[m]->fissionable()) ? V : 0.0;
      break;
    }
    case VoxelMode::tabulated:
      if (static_cast<int64_t>(opt.weights.size()) != R)
        fatal_error("RMC residual: tabulated voxel weights need one value per "
                    "region.");
      pi_[r] = opt.weights_per_volume ? V * opt.weights[r] : opt.weights[r];
      break;
    default:
      break;
    }
    if (!(pi_[r] >= 0.0) || (pi_[r] > 0.0 && !(V > 0.0)))
      fatal_error("RMC residual: invalid voxel weight or volume.");
  }
  double sum = 0.0;
  for (double p : pi_)
    sum += p;
  if (!(sum > 0.0))
    fatal_error("RMC residual: every voxel has zero probability.");
  cdf_.resize(R);
  double c = 0.0;
  for (int64_t r = 0; r < R; ++r) {
    pi_[r] /= sum;
    c += pi_[r];
    cdf_[r] = c;
  }
}

double VoxelProposal::density(int64_t region) const
{
  if (opt_.mode == VoxelMode::box)
    return 1.0 / V_box_;
  if (region < 0 || pi_[region] == 0.0)
    return 0.0;
  return pi_[region] / field_.volume(region);
}

Draw<Position> VoxelProposal::sample(uint64_t* seed, int64_t& region) const
{
  const Position lo = field_.box_lower_left();
  const Position w = field_.box_upper_right() - lo;
  int32_t m;
  double kT, dm;
  if (opt_.mode == VoxelMode::box) {
    Position r {
      lo.x + w.x * prn(seed), lo.y + w.y * prn(seed), lo.z + w.z * prn(seed)};
    if (!locate(field_, r, region, m, kT, dm))
      fatal_error("RMC residual: destination outside the geometry.");
    return {r, 1.0 / V_box_};
  }

  // A region, then a point uniform in it by rejection from its mesh element
  // (or the box, for a region without a mesh)
  int64_t target = draw_index(cdf_, seed);
  int32_t mesh = field_.region_mesh(target);
  int32_t cell_id, instance;
  int64_t bin;
  field_.region_key(target, cell_id, instance, bin);
  for (int tries = 0; tries < 10'000'000; ++tries) {
    Position r = (mesh >= 0)
                   ? model::meshes[mesh]->sample_element(bin, seed)
                   : Position {lo.x + w.x * prn(seed), lo.y + w.y * prn(seed),
                       lo.z + w.z * prn(seed)};
    if (locate(field_, r, region, m, kT, dm) && region == target)
      return {r, pi_[target] / field_.volume(target)};
  }
  fatal_error("RMC residual: could not place a point in a region.");
}

//==============================================================================
// Outgoing proposal
//==============================================================================

OutgoingProposal::OutgoingProposal(
  const FrozenField& field, const OutgoingOptions& opt)
  : field_(field), opt_(opt), log_ratio_(std::log(opt.E_max / opt.E_min))
{
  if (!(opt.E_min > 0.0 && opt.E_max > opt.E_min))
    fatal_error("RMC residual: invalid outgoing energy range.");
  if (!(opt.beta > 0.0 && opt.beta <= 1.0))
    fatal_error("RMC residual: the log-uniform weight beta must be in (0, 1] "
                "so that the proposal covers the residual's support.");
}

namespace {
double group_sum(const FrozenField& f, Tag t, int64_t region)
{
  double s = 0.0;
  if (region < 0)
    return 0.0;
  for (int g = 0; g < f.n_groups(); ++g)
    s += f.emission(region, t, g);
  return s;
}
} // namespace

Draw<OutgoingState> OutgoingProposal::sample(
  Tag t, int64_t region, uint64_t* seed) const
{
  OutgoingState s;
  double mass = (opt_.beta < 1.0) ? group_sum(field_, t, region) : 0.0;
  if (mass > 0.0 && prn(seed) >= opt_.beta) {
    // Group by emission, then the lift within the group
    double x = prn(seed) * mass, c = 0.0;
    int g = 0;
    for (; g < field_.n_groups() - 1; ++g) {
      c += field_.emission(region, t, g);
      if (c > x)
        break;
    }
    double dens;
    s.E = field_.sample_shape(
      static_cast<FrozenField::Shape>(t), region, g, seed, dens);
  } else {
    s.E = opt_.E_min * std::exp(log_ratio_ * prn(seed));
  }
  double mu = 2.0 * prn(seed) - 1.0;
  double phi = 2.0 * PI * prn(seed);
  double st = std::sqrt(std::max(0.0, 1.0 - mu * mu));
  s.u = {st * std::cos(phi), st * std::sin(phi), mu};
  return {s, density(t, region, s)};
}

double OutgoingProposal::energy_density(Tag t, int64_t region, double E) const
{
  double f = 0.0;
  double mass = (opt_.beta < 1.0) ? group_sum(field_, t, region) : 0.0;
  double beta = (mass > 0.0) ? opt_.beta : 1.0;
  if (E >= opt_.E_min && E <= opt_.E_max)
    f += beta / (E * log_ratio_);
  if (mass > 0.0) {
    int g = field_.group(E);
    if (g >= 0) {
      double Qg = field_.emission(region, t, g);
      if (Qg > 0.0)
        f += (1.0 - beta) * Qg / mass *
             field_.shape_draw_density(
               static_cast<FrozenField::Shape>(t), region, g, E);
    }
  }
  return f;
}

double OutgoingProposal::density(
  Tag t, int64_t region, const OutgoingState& s) const
{
  return energy_density(t, region, s.E) / (4.0 * PI);
}

//==============================================================================
// Directions and records
//==============================================================================

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
  Direction v {std::cos(phi) * std::sqrt(z), std::sin(phi) * std::sqrt(z),
    std::sqrt(1 - z)};
  for (auto& d : out) {
    // Rotate about z by theta, then reflect with the Householder matrix of v
    Direction a {c * d.x - s * d.y, s * d.x + c * d.y, d.z};
    double vd = v.dot(a);
    d = {2 * vd * v.x - a.x, 2 * vd * v.y - a.y, 2 * vd * v.z - a.z};
  }
}

void ResidualRecords::clear()
{
  for (auto* v : {&r, &u, &E, &density, &C_s, &C_f, &M_s, &M_f, &R_s, &R_f, &U})
    v->clear();
  tag.clear();
  sampler.clear();
  kind.clear();
  channel.clear();
  Z.clear();
}

void weight_records(const ResidualRecords& rec, int64_t N,
  const WeightScalars& x, vector<double>& w)
{
  w.resize(rec.size());
  for (int64_t i = 0; i < rec.size(); ++i) {
    const int t = rec.tag[i];
    const double inv_k = (t == static_cast<int>(Tag::f)) ? 1.0 / x.k : 1.0;
    double num;
    if (rec.sampler[i]) {
      double z = rec.Z.empty() ? 0.0 : rec.Z[i];
      num = (x.a[0] * rec.C_s[i] + x.a[1] * rec.C_f[i] + z) * inv_k;
    } else {
      double high =
        x.a[0] * (rec.C_s[i] - x.c * rec.M_s[i] + x.c * rec.R_s[i]) +
        x.a[1] * (rec.C_f[i] - x.c * rec.M_f[i] + x.c * rec.R_f[i]);
      double swept = (x.a[0] * rec.R_s[i] + x.a[1] * rec.R_f[i]) * inv_k;
      num = high * inv_k -
            x.b[t] * ((1.0 - x.lambda) * rec.U[i] + x.lambda * swept);
    }
    w[i] = num / (N * rec.density[i]);
  }
}

//==============================================================================
// ResidualSampler
//==============================================================================

ResidualSampler::ResidualSampler(const FrozenField& field,
  const ResidualOptions& opt, const MGTables* tables, const SweepRecords* sweep)
  : field_(field), opt_(opt), tables_(tables), sweep_(sweep),
    voxel_(field, opt.voxel), outgoing_(field, opt.outgoing),
    registry_(opt.sampler.all_sampler), backward_(field)
{
  if (opt.sampler.energies < 1)
    fatal_error("RMC residual: the sampler route needs K >= 1 energies.");
  if (opt.sampler.control == SamplerControl::region) {
    if (!field.has_flux())
      fatal_error("RMC residual: the flux-difference control needs the field's "
                  "scalar flux (field file version with regions/phi).");
    if (!field.has_shape(FrozenField::Shape::phi))
      fatal_error("RMC residual: the flux-difference control needs h_phi.");
    if (opt.sampler.region_samples < 1)
      fatal_error("RMC residual: the region term needs at least one sample.");
  }
  if (opt.directions == DirectionMode::isotropic && opt.n_directions < 1)
    fatal_error("RMC residual: at least one incident direction is needed.");
  if (opt.directions == DirectionMode::sweep && !sweep)
    fatal_error("RMC residual: sweep directions need a field sweep.");
  if (opt.directions == DirectionMode::sweep && !field.has_volumes())
    fatal_error("RMC residual: sweep weights need region volumes.");
  bool need_tables = opt.control || opt.incident.mode == IncidentMode::flux ||
                     opt.directions == DirectionMode::sweep;
  if (need_tables && !tables)
    fatal_error("RMC residual: the control, the flux proposal, and sweep "
                "directions need the MG tables.");
  if (opt.control && opt.directions == DirectionMode::isotropic &&
      opt.n_control_directions < 1)
    fatal_error("RMC residual: an isotropic control needs L >= 1 MG-only "
                "directions to restore its mean.");
  if (opt.incident.mode == IncidentMode::flux &&
      !field.has_shape(FrozenField::Shape::phi))
    fatal_error("RMC residual: the flux proposal needs the flux shape h_phi.");
  if (opt.control_energy == ControlEnergy::same_energy &&
      !field.has_shape(FrozenField::Shape::phi))
    fatal_error("RMC residual: the same-energy control needs the flux shape.");
}

bool ResidualSampler::needs_mg_flux(Tag t) const
{
  return opt_.control || (opt_.incident.mode == IncidentMode::flux &&
                           (t == Tag::f || opt_.sampler_route));
}

Draw<double> ResidualSampler::incident(Tag t, const Destination& d,
  const vector<double>& mg_s, const vector<double>& mg_f, uint64_t* seed)
{
  const auto& o = opt_.incident;
  const double log_ratio = std::log(o.E_max / o.E_min);
  if (o.mode == IncidentMode::flux) {
    const int G = field_.n_groups();
    w_g_.assign(G, 0.0);
    double sum = 0.0;
    for (int g = 0; g < G; ++g) {
      double psi = mg_s[g] + mg_f[g];
      double x = (t == Tag::f)
                   ? tables_->nu_sigma_f(d.material, d.sqrtkT, g) * psi
                   : psi;
      w_g_[g] = std::max(x, 0.0);
      sum += w_g_[g];
    }
    if (sum > 0.0 && prn(seed) >= o.beta) {
      double x = prn(seed) * sum, c = 0.0;
      int g = 0;
      for (; g < G - 1; ++g) {
        c += w_g_[g];
        if (c > x)
          break;
      }
      double dens;
      double E =
        field_.sample_shape(FrozenField::Shape::phi, d.region, g, seed, dens);
      return {E, incident_density(t, d, mg_s, mg_f, E)};
    }
    double E = o.E_min * std::exp(log_ratio * prn(seed));
    return {E, incident_density(t, d, mg_s, mg_f, E)};
  }
  double E = o.E_min * std::exp(log_ratio * prn(seed));
  return {E, 1.0 / (E * log_ratio)};
}

double ResidualSampler::incident_density(Tag t, const Destination& d,
  const vector<double>& mg_s, const vector<double>& mg_f, double E) const
{
  const auto& o = opt_.incident;
  const double log_ratio = std::log(o.E_max / o.E_min);
  double lu = (E >= o.E_min && E <= o.E_max) ? 1.0 / (E * log_ratio) : 0.0;
  if (o.mode != IncidentMode::flux)
    return lu;
  const int G = field_.n_groups();
  double sum = 0.0, wg = 0.0;
  int gE = field_.group(E);
  for (int g = 0; g < G; ++g) {
    double psi = mg_s[g] + mg_f[g];
    double x =
      (t == Tag::f) ? tables_->nu_sigma_f(d.material, d.sqrtkT, g) * psi : psi;
    x = std::max(x, 0.0);
    sum += x;
    if (g == gE)
      wg = x;
  }
  if (!(sum > 0.0))
    return lu;
  double f = o.beta * lu;
  if (gE >= 0 && wg > 0.0)
    f += (1.0 - o.beta) * wg / sum *
         field_.shape_draw_density(FrozenField::Shape::phi, d.region, gE, E);
  return f;
}

void ResidualSampler::mg_terms(Tag t, const Destination& d,
  const OutgoingState& s, const vector<double>& mg_s,
  const vector<double>& mg_f, double E_in, double p_in, double M[2]) const
{
  MGControl ctl(field_, *tables_);
  const vector<double>* mg[2] = {&mg_s, &mg_f};
  for (int y = 0; y < 2; ++y) {
    if (t == Tag::f && opt_.control_energy == ControlEnergy::same_energy) {
      M[y] = ctl.fission_same_energy(
        d.region, d.material, d.sqrtkT, s.E, E_in, p_in, *mg[y]);
    } else {
      M[y] = ctl.production(t, d.region, d.material, d.sqrtkT, s.E, *mg[y]);
    }
    // The tables are for unit density; the cell instance scales them
    M[y] *= d.density_mult;
  }
}

void ResidualSampler::direction(Tag t, const Destination& d, Direction omega_in,
  const OutgoingState& s, const double* psi_seg_s, const double* psi_seg_f,
  uint64_t* seed, DirectionTerms& out)
{
  out = DirectionTerms {};
  const Material& mat = *model::materials[d.material];
  const Direction u_back = -omega_in;
  const bool sweep_dir = psi_seg_s != nullptr;
  const int G = field_.n_groups();
  const bool want_mg = needs_mg_flux(t);

  // MG fluxes of this direction: its sweep segment, or its own path
  bool two_pass = false;
  if (sweep_dir && want_mg) {
    mg_s_.assign(psi_seg_s, psi_seg_s + G);
    mg_f_.assign(psi_seg_f, psi_seg_f + G);
  } else if (want_mg && opt_.incident.mode == IncidentMode::flux) {
    // The incident proposal needs the MG flux before the CE energies are
    // known: an MG-only pass first, then the CE pass on the same path
    two_pass = true;
    energies_.clear();
    backward_.evaluate_all(d.r, u_back, energies_, tables_, opt_.tail, seed,
      psi_s_, psi_f_, mg_s_, mg_f_);
  }

  // CE incident energies: elastic images of every nuclide (tag s, evaluator
  // route), one drawn energy for the fission evaluator (tag f), and K drawn
  // energies for the sampler route
  const bool all_sampler = registry_.all_sampler();
  const bool sampled = opt_.sampler_route && (t == Tag::s || all_sampler);
  const int K = sampled ? opt_.sampler.energies : 0;
  energies_.clear();
  image_density_.clear();
  image_nuclide_.clear();
  double mu0 = omega_in.dot(s.u);
  if (t == Tag::s && !all_sampler) {
    probe_.total(d.material, s.E, d.sqrtkT, d.density_mult);
    for (int i = 0; i < static_cast<int>(mat.nuclide_.size()); ++i) {
      int i_nuc = mat.nuclide_[i];
      auto img = elastic_stationary(
        i_nuc, ChannelRegistry::elastic_kT(i_nuc, probe_), s.E, mu0);
      if (img.density > 0.0) {
        energies_.push_back(img.E_in);
        image_density_.push_back(img.density);
        image_nuclide_.push_back(i);
      }
    }
  }
  Draw<double> Ein {0.0, 0.0};
  const bool fission_evaluator = (t == Tag::f) && !all_sampler;
  if (fission_evaluator) {
    Ein = incident(t, d, mg_s_, mg_f_, seed);
    energies_.push_back(Ein.value);
  }
  const size_t first_sampled = energies_.size();
  sampler_E_.clear();
  for (int k = 0; k < K; ++k) {
    auto e = incident(t, d, mg_s_, mg_f_, seed);
    sampler_E_.push_back(e);
    energies_.push_back(e.value);
  }

  // Backward pass(es)
  const bool combined = want_mg && !sweep_dir && !two_pass;
  backward_.evaluate_all(d.r, u_back, energies_, combined ? tables_ : nullptr,
    opt_.tail, seed, psi_s_, psi_f_, seg_s_, seg_f_);
  if (combined) {
    mg_s_ = seg_s_;
    mg_f_ = seg_f_;
  }

  // CE production of the evaluator channels
  if (t == Tag::s && !all_sampler) {
    for (size_t k = 0; k < image_nuclide_.size(); ++k) {
      int i = image_nuclide_[k];
      probe_.total(d.material, energies_[k], d.sqrtkT, d.density_mult);
      Channel el {i, mat.nuclide_[i], 0, ChannelKind::elastic, Tag::s};
      if (registry_.route(el, energies_[k], probe_) != Route::evaluator)
        continue;
      double sigma_el = ChannelRegistry::production(el, d.density_mult, probe_);
      // Density per unit mu0 and azimuth: 1 / (2 pi)
      double f = sigma_el * image_density_[k] / (2.0 * PI);
      out.C[0] += f * psi_s_[k];
      out.C[1] += f * psi_f_[k];
    }
  } else if (fission_evaluator && Ein.density > 0.0) {
    // Fission: OpenMC's tabulated nu-fission per nuclide, shared among
    // partial fission reactions by their cross sections (as sample_fission)
    probe_.total(d.material, Ein.value, d.sqrtkT, d.density_mult);
    const Particle& p = probe_.particle();
    double emission = 0.0;
    for (int i = 0; i < static_cast<int>(mat.nuclide_.size()); ++i) {
      int i_nuc = mat.nuclide_[i];
      const Nuclide& nuc = *data::nuclides[i_nuc];
      if (!nuc.fissionable_)
        continue;
      const auto& micro = p.neutron_xs(i_nuc);
      if (!(micro.fission > 0.0))
        continue;
      double nu_sigma_f =
        mat.atom_density(i, d.density_mult) * micro.nu_fission;
      if (nuc.has_partial_fission_) {
        for (const Reaction* rx : nuc.fission_rx_) {
          double xs = rx->xs(micro);
          if (xs > 0.0)
            emission += nu_sigma_f * xs / micro.fission *
                        fission_energy_density(i_nuc, *rx, Ein.value, s.E);
        }
      } else {
        emission += nu_sigma_f * fission_energy_density(
                                   i_nuc, *nuc.fission_rx_[0], Ein.value, s.E);
      }
    }
    double f = emission / (4.0 * PI * Ein.density);
    out.C[0] = f * psi_s_[0];
    out.C[1] = f * psi_f_[0];
  }

  // Sampler route: K incident energies on this path, each with one outgoing
  // state from OpenMC's samplers. With common random numbers the K draws use
  // the same stream, so their outgoing states move together with E'.
  if (K > 0) {
    const bool region = opt_.sampler.control == SamplerControl::region;
    uint64_t crn = *seed;
    candidates_.clear();
    for (int k = 0; k < K; ++k) {
      const auto& e = sampler_E_[k];
      if (!(e.density > 0.0))
        continue;
      size_t idx = first_sampled + k;
      probe_.total(d.material, e.value, d.sqrtkT, d.density_mult);
      uint64_t sk = crn;
      uint64_t* sp = opt_.sampler.common_numbers ? &sk : seed;
      auto smp = sample_sampler_route(registry_, t, omega_in, probe_, sp);
      if (!(smp.production > 0.0))
        continue;
      double f = smp.production / (K * e.density);
      SampledParticle q;
      q.state = {smp.E, smp.u};
      q.C[0] = f * psi_s_[idx];
      q.C[1] = f * psi_f_[idx];
      // Flux-difference control: subtract the region's lifted MG flux,
      // which does not depend on the amplitudes
      if (region)
        q.Z = -f * region_flux(d.region, e.value) / (4.0 * PI);
      q.channel = static_cast<int>(smp.kind);
      candidates_.push_back(q);
    }
    if (opt_.sampler.common_numbers)
      prn(seed); // move the main stream on
    if (opt_.sampler.net && candidates_.size() > 1) {
      // Biased (study only): one outgoing state, chosen by |weight| at
      // a = (1, 1), carries the net of the K
      double total = 0.0;
      SampledParticle net;
      for (const auto& q : candidates_) {
        total += std::abs(q.C[0] + q.C[1] + q.Z);
        net.C[0] += q.C[0];
        net.C[1] += q.C[1];
        net.Z += q.Z;
      }
      double x = prn(seed) * total, c = 0.0;
      size_t pick = candidates_.size() - 1;
      for (size_t i = 0; i < candidates_.size(); ++i) {
        c += std::abs(
          candidates_[i].C[0] + candidates_[i].C[1] + candidates_[i].Z);
        if (c > x) {
          pick = i;
          break;
        }
      }
      net.state = candidates_[pick].state;
      net.channel = candidates_[pick].channel;
      out.sampled.push_back(net);
    } else {
      for (const auto& q : candidates_)
        out.sampled.push_back(q);
    }
  }

  // MG control on this direction
  if (opt_.control)
    mg_terms(t, d, s, mg_s_, mg_f_, Ein.value, Ein.density, out.M);
}

double ResidualSampler::region_flux(int64_t region, double E) const
{
  int g = field_.group(E);
  if (g < 0 || region < 0)
    return 0.0;
  return field_.flux(region, g) *
         field_.shape(FrozenField::Shape::phi, region, E);
}

void ResidualSampler::region_term(
  Tag t, const Destination& d, uint64_t* seed, vector<SampledParticle>& out)
{
  // The region term of the flux-difference control,
  //   int int Sigma_P^samp(E' -> x | Omega') phi_{R,g} h_phi(E') / 4 pi,
  // flat in the region and isotropic in Omega': E' from phi_{R,g} h_phi,
  // Omega' uniform, the outgoing state from OpenMC's samplers. No backtrace.
  const int G = field_.n_groups();
  double sum = 0.0;
  for (int g = 0; g < G; ++g)
    sum += field_.flux(d.region, g);
  if (!(sum > 0.0))
    return;
  const int M = opt_.sampler.region_samples;
  for (int i = 0; i < M; ++i) {
    double x = prn(seed) * sum, c = 0.0;
    int g = 0;
    for (; g < G - 1; ++g) {
      c += field_.flux(d.region, g);
      if (c > x)
        break;
    }
    double dens;
    double E =
      field_.sample_shape(FrozenField::Shape::phi, d.region, g, seed, dens);
    double q = field_.flux(d.region, g) / sum * dens;
    if (!(q > 0.0))
      continue;
    double mu = 2.0 * prn(seed) - 1.0;
    double ph = 2.0 * PI * prn(seed);
    double st = std::sqrt(std::max(0.0, 1.0 - mu * mu));
    Direction u {st * std::cos(ph), st * std::sin(ph), mu};
    probe_.total(d.material, E, d.sqrtkT, d.density_mult);
    auto smp = sample_sampler_route(registry_, t, u, probe_, seed);
    if (!(smp.production > 0.0))
      continue;
    SampledParticle p;
    p.state = {smp.E, smp.u};
    // Omega' density 1 / 4 pi cancels the 1 / 4 pi of the lifted flux
    p.Z = smp.production * region_flux(d.region, E) / q / M;
    p.channel = static_cast<int>(smp.kind);
    out.push_back(p);
  }
}

void ResidualSampler::control_mean(Tag t, const Destination& d,
  const OutgoingState& s, uint64_t* seed, double R[2])
{
  R[0] = R[1] = 0.0;
  const int L = opt_.n_control_directions;
  vector<Direction> dirs;
  stratified_directions(L, seed, dirs);
  const double omega = 4.0 * PI / L;
  vector<double> none;
  for (const auto& omega_in : dirs) {
    backward_.evaluate_all(d.r, -omega_in, none, tables_, opt_.tail, seed,
      psi_s_, psi_f_, mg_s_, mg_f_);
    // The same MG estimator as on the J directions, so the same mean
    Draw<double> Ein {0.0, 0.0};
    if (t == Tag::f && opt_.control_energy == ControlEnergy::same_energy)
      Ein = incident(t, d, mg_s_, mg_f_, seed);
    double M[2];
    mg_terms(t, d, s, mg_s_, mg_f_, Ein.value, Ein.density, M);
    R[0] += omega * M[0];
    R[1] += omega * M[1];
  }
}

void ResidualSampler::push_sampled(ResidualRecords& out, Position r,
  const SampledParticle& q, double omega, double density, Tag t, int kind)
{
  out.r.insert(out.r.end(), {r.x, r.y, r.z});
  out.u.insert(out.u.end(), {q.state.u.x, q.state.u.y, q.state.u.z});
  out.E.push_back(q.state.E);
  out.density.push_back(density); // the destination's: (E, Omega) are sampled
  out.tag.push_back(static_cast<int>(t));
  out.sampler.push_back(1);
  out.kind.push_back(kind);
  out.channel.push_back(q.channel);
  out.C_s.push_back(omega * q.C[0]);
  out.C_f.push_back(omega * q.C[1]);
  out.Z.push_back(omega * q.Z);
  for (auto* v : {&out.M_s, &out.M_f, &out.R_s, &out.R_f, &out.U})
    v->push_back(0.0);
}

void ResidualSampler::sample(
  Tag t, int64_t n, uint64_t seed, ResidualRecords& out)
{
  const int G = field_.n_groups();
  vector<Direction> dirs;
  vector<double> weights;
  vector<double> Psi_s, Psi_f;
  if (keep_) {
    kept_ce_.clear();
    kept_mg_.clear();
  }

  for (int64_t i = 0; i < n; ++i) {
    // Each sample has its own stream, so results do not depend on order
    uint64_t s = future_seed(static_cast<uint64_t>(i) * 1000003ULL, seed);

    // Destination and its state
    int64_t region;
    auto rd = voxel_.sample(&s, region);
    Destination d;
    d.r = rd.value;
    if (!locate(field_, d.r, d.region, d.material, d.sqrtkT, d.density_mult))
      fatal_error("RMC residual: destination outside the geometry.");

    // Outgoing state, and the field at it
    auto draw = outgoing_.sample(t, d.region, &s);
    const OutgoingState& st = draw.value;
    double U = field_.evaluate(d.region, t, st.E);

    double C[2] = {0.0, 0.0}, M[2] = {0.0, 0.0}, R[2] = {0.0, 0.0};
    bool active = d.material != MATERIAL_VOID;
    if (t == Tag::f && active)
      active = model::materials[d.material]->fissionable();

    if (active) {
      // Directions and weights
      const double* seg_s = nullptr;
      const double* seg_f = nullptr;
      if (opt_.directions == DirectionMode::isotropic) {
        stratified_directions(opt_.n_directions, &s, dirs);
        weights.assign(dirs.size(), 4.0 * PI / opt_.n_directions);
        picks_.clear();
      } else {
        // Sweep segments in the region; omega_j = 4 pi l_j / (rho V_R)
        int64_t lo = (d.region >= 0) ? sweep_->offset[d.region] : 0;
        int64_t cnt = (d.region >= 0) ? sweep_->count(d.region) : 0;
        double scale = 4.0 * PI / (sweep_->rho * field_.volume(d.region));
        picks_.clear();
        weights.clear();
        dirs.clear();
        double total_w = 0.0;
        for (int64_t j = lo; j < lo + cnt; ++j)
          total_w += scale * sweep_->length[j];
        if (opt_.n_directions <= 0 || cnt <= opt_.n_directions) {
          for (int64_t j = lo; j < lo + cnt; ++j) {
            picks_.push_back(j);
            weights.push_back(scale * sweep_->length[j]);
          }
        } else {
          // Subsample with probability proportional to omega_j: each pick
          // estimates the region's sum with weight sum(omega) / J
          for (int k = 0; k < opt_.n_directions; ++k) {
            double x = prn(&s) * total_w, c = 0.0;
            int64_t j = lo;
            for (; j < lo + cnt - 1; ++j) {
              c += scale * sweep_->length[j];
              if (c > x)
                break;
            }
            picks_.push_back(j);
            weights.push_back(total_w / opt_.n_directions);
          }
        }
        for (int64_t j : picks_)
          dirs.push_back(sweep_->u[j]);

        // The sweep's region emission Q-hat (restored mean, and the swept
        // field candidate): sum_j omega_j over every segment, which is linear
        // in psi, so one MG production of the omega-weighted flux
        {
          Psi_s.assign(G, 0.0);
          Psi_f.assign(G, 0.0);
          for (int64_t j = lo; j < lo + cnt; ++j) {
            double w = scale * sweep_->length[j];
            for (int g = 0; g < G; ++g) {
              Psi_s[g] += w * sweep_->psi_s[j * G + g];
              Psi_f[g] += w * sweep_->psi_f[j * G + g];
            }
          }
          MGControl ctl(field_, *tables_);
          R[0] = d.density_mult *
                 ctl.production(t, d.region, d.material, d.sqrtkT, st.E, Psi_s);
          R[1] = d.density_mult *
                 ctl.production(t, d.region, d.material, d.sqrtkT, st.E, Psi_f);
        }
      }

      for (size_t j = 0; j < dirs.size(); ++j) {
        DirectionTerms dt;
        const double* ps = nullptr;
        const double* pf = nullptr;
        if (opt_.directions == DirectionMode::sweep) {
          ps = &sweep_->psi_s[picks_[j] * G];
          pf = &sweep_->psi_f[picks_[j] * G];
        }
        direction(t, d, dirs[j], st, ps, pf, &s, dt);
        C[0] += weights[j] * dt.C[0];
        C[1] += weights[j] * dt.C[1];
        M[0] += weights[j] * dt.M[0];
        M[1] += weights[j] * dt.M[1];
        if (keep_) {
          kept_ce_.push_back(dt.C[0] + dt.C[1]);
          kept_mg_.push_back(dt.M[0] + dt.M[1]);
        }
        for (const auto& q : dt.sampled)
          push_sampled(out, d.r, q, weights[j], rd.density, t, 1);
      }
      if (opt_.control && opt_.directions == DirectionMode::isotropic)
        control_mean(t, d, st, &s, R);
      if (opt_.sampler.control == SamplerControl::region &&
          opt_.sampler_route && (t == Tag::s || registry_.all_sampler())) {
        region_particles_.clear();
        region_term(t, d, &s, region_particles_);
        for (const auto& q : region_particles_)
          push_sampled(out, d.r, q, 1.0, rd.density, t, 2);
      }
    }

    // The density-branch record
    out.r.insert(out.r.end(), {d.r.x, d.r.y, d.r.z});
    out.u.insert(out.u.end(), {st.u.x, st.u.y, st.u.z});
    out.E.push_back(st.E);
    out.density.push_back(draw.density * rd.density);
    out.tag.push_back(static_cast<int>(t));
    out.sampler.push_back(0);
    out.kind.push_back(0);
    out.channel.push_back(-1);
    out.Z.push_back(0.0);
    out.C_s.push_back(C[0]);
    out.C_f.push_back(C[1]);
    out.M_s.push_back(M[0]);
    out.M_f.push_back(M[1]);
    out.R_s.push_back(R[0]);
    out.R_f.push_back(R[1]);
    out.U.push_back(U);
  }
}

} // namespace rmc
} // namespace openmc
