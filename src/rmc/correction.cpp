#include "openmc/rmc/correction.h"

#include <algorithm>
#include <cmath>
#include <unordered_map>

#include "openmc/cell.h"
#include "openmc/constants.h"
#include "openmc/error.h"
#include "openmc/geometry.h"
#include "openmc/material.h"
#include "openmc/nuclide.h"
#include "openmc/random_lcg.h"
#include "openmc/reaction.h"
#include "openmc/rmc/production.h"
#include "openmc/rmc/residual.h"

namespace openmc {
namespace rmc {

namespace {

bool locate_point(const FrozenField& field, Position r, int64_t& region,
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

} // namespace

void field_fission_production(const FrozenField& field, const MGTables& mg,
  int64_t n_points, int n_E, const TailOptions& tail, uint64_t seed,
  double X[2], double var[2])
{
  // X_y = int dr dE' dOmega nu Sigma_f(r, E') psi_y(r, E', Omega), sampled
  // with the density of the field's own lifted fission rate on the lifts'
  // fine grid: region r and fine bin i with probability
  //   V_r phi_{r,g(i)} h_phi(r, E_i) dE_i nuSigma_f(m_r, E_i),
  // the energy uniform in the bin, and the direction isotropic. The score
  // is then the total of those weights times two ratios near one: the CE
  // nu-fission cross section at E' to its value at the bin's midpoint, and
  // the path's CE angular flux to the region's lifted scalar flux per unit
  // solid angle. (mg is not used: the proposal needs no MG cross sections.)
  (void)mg;
  const int64_t R = field.n_regions();
  const auto& fe = field.fine_edges();
  const int F = static_cast<int>(fe.size()) - 1;
  const double sqrtkT0 = std::sqrt(K_BOLTZMANN * 293.6);
  XsProbe probe;
  // nu Sigma_f at the fine-bin midpoints, per fissionable material
  std::unordered_map<int32_t, vector<double>> nsf;
  vector<int64_t> regions; //!< fissionable regions
  vector<double> wr(R, 0.0);
  vector<vector<double>> cdf(R);
  for (int64_t r = 0; r < R; ++r) {
    int32_t m = field.region_material(r);
    if (m == MATERIAL_VOID || !model::materials[m]->fissionable())
      continue;
    auto it = nsf.find(m);
    if (it == nsf.end()) {
      vector<double> v(F);
      for (int i = 0; i < F; ++i) {
        probe.total(m, std::sqrt(fe[i] * fe[i + 1]), sqrtkT0, 1.0);
        v[i] = probe.particle().macro_xs().nu_fission;
      }
      it = nsf.emplace(m, std::move(v)).first;
    }
    cdf[r].resize(F);
    double c = 0.0;
    for (int i = 0; i < F; ++i) {
      double Em = std::sqrt(fe[i] * fe[i + 1]);
      int g = field.group(Em);
      double phi = (g >= 0) ? field.flux(r, g) *
                                field.shape(FrozenField::Shape::phi, r, Em)
                            : 0.0;
      c += field.volume(r) * phi * (fe[i + 1] - fe[i]) * it->second[i];
      cdf[r][i] = c;
    }
    wr[r] = c;
  }
  VoxelOptions vo;
  vo.mode = VoxelMode::tabulated;
  vo.weights = wr;
  VoxelProposal voxel(field, vo);
  BackwardFlux backward(field);
  vector<double> E(n_E), q(n_E), psi_s, psi_f, mg_s, mg_f;
  double sum[2] = {0.0, 0.0}, sq[2] = {0.0, 0.0};
  for (int64_t i = 0; i < n_points; ++i) {
    uint64_t s = future_seed(static_cast<uint64_t>(i) * 1000003ULL + 11, seed);
    int64_t region;
    auto rd = voxel.sample(&s, region);
    int32_t mat;
    double kT, dm;
    if (!locate_point(field, rd.value, region, mat, kT, dm) || region < 0)
      fatal_error("RMC field production: point outside the field.");
    const auto& c = cdf[region];
    for (int k = 0; k < n_E; ++k) {
      double x = prn(&s) * wr[region];
      int b = static_cast<int>(std::upper_bound(c.begin(), c.end(), x) - c.begin());
      b = std::min(b, F - 1);
      double mass = c[b] - (b > 0 ? c[b - 1] : 0.0);
      E[k] = fe[b] + (fe[b + 1] - fe[b]) * prn(&s);
      q[k] = mass / wr[region] / (fe[b + 1] - fe[b]);
    }
    double mu = 2.0 * prn(&s) - 1.0, ph = 2.0 * PI * prn(&s);
    double st = std::sqrt(std::max(0.0, 1.0 - mu * mu));
    Direction u {st * std::cos(ph), st * std::sin(ph), mu};
    backward.evaluate_all(
      rd.value, u, E, nullptr, tail, &s, psi_s, psi_f, mg_s, mg_f);
    double sc[2] = {0.0, 0.0};
    for (int k = 0; k < n_E; ++k) {
      if (!(q[k] > 0.0))
        continue;
      probe.total(mat, E[k], kT, dm);
      double f = probe.particle().macro_xs().nu_fission * 4.0 * PI /
                 (rd.density * q[k] * n_E);
      sc[0] += f * psi_s[k];
      sc[1] += f * psi_f[k];
    }
    for (int y = 0; y < 2; ++y) {
      sum[y] += sc[y];
      sq[y] += sc[y] * sc[y];
    }
  }
  for (int y = 0; y < 2; ++y) {
    X[y] = sum[y] / n_points;
    var[y] = (n_points > 1)
               ? (sq[y] / n_points - X[y] * X[y]) / (n_points - 1)
               : 0.0;
  }
}

void field_reaction_tally(const FrozenField& field, int64_t n_points,
  int n_E, const TailOptions& tail, uint64_t seed, FieldTallyResult& out,
  const MGTables* mg)
{
  const int64_t R = field.n_regions();
  const int G = field.n_groups();
  out.fission.assign(2 * R, 0.0);
  out.nu_fission.assign(2 * R, 0.0);
  out.fission_sq.assign(2 * R, 0.0);
  out.nu_fission_mg.assign(2 * R, 0.0);
  out.diff_sq.assign(2 * R, 0.0);
  BackwardFlux backward(field);
  XsProbe probe;
  Position lo = field.box_lower_left(), hi = field.box_upper_right();
  Position w = hi - lo;
  const double V_box = w.x * w.y * w.z;
  const double E_lo = 1e-5, E_hi = 2e7;
  vector<double> E(n_E), q(n_E), psi_s, psi_f, mg_s, mg_f;

  for (int64_t i = 0; i < n_points; ++i) {
    uint64_t s = future_seed(static_cast<uint64_t>(i) * 1000003ULL + 7, seed);
    Position r {lo.x + w.x * prn(&s), lo.y + w.y * prn(&s), lo.z + w.z * prn(&s)};
    int64_t region;
    int32_t mat;
    double kT, dm;
    if (!locate_point(field, r, region, mat, kT, dm))
      fatal_error("RMC field tally: point outside the geometry.");
    if (region < 0 || mat == MATERIAL_VOID ||
        !model::materials[mat]->fissionable())
      continue;
    // Incident energies from the region's lifted MG flux
    double sum = 0.0;
    for (int g = 0; g < G; ++g)
      sum += field.flux(region, g);
    for (int k = 0; k < n_E; ++k) {
      if (sum > 0.0) {
        double x = prn(&s) * sum, c = 0.0;
        int g = 0;
        for (; g < G - 1; ++g) {
          c += field.flux(region, g);
          if (c > x)
            break;
        }
        double dens;
        E[k] = field.sample_shape(FrozenField::Shape::phi, region, g, &s, dens);
        q[k] = field.flux(region, g) / sum * dens;
      } else {
        E[k] = E_lo * std::exp(std::log(E_hi / E_lo) * prn(&s));
        q[k] = 1.0 / (E[k] * std::log(E_hi / E_lo));
      }
    }
    double mu = 2.0 * prn(&s) - 1.0, ph = 2.0 * PI * prn(&s);
    double st = std::sqrt(std::max(0.0, 1.0 - mu * mu));
    Direction u {st * std::cos(ph), st * std::sin(ph), mu};
    backward.evaluate_all(r, u, E, mg, tail, &s, psi_s, psi_f, mg_s, mg_f);
    // MG nu-fission rate of the MG flux on the same path (group-integrated
    // angular fluxes per unit solid angle)
    double sc_mg[2] = {0.0, 0.0};
    if (mg) {
      for (int g = 0; g < G; ++g) {
        double nsf = dm * mg->nu_sigma_f(mat, kT, g);
        sc_mg[0] += V_box * 4.0 * PI * nsf * mg_s[g];
        sc_mg[1] += V_box * 4.0 * PI * nsf * mg_f[g];
      }
    }
    double sc_f[2] = {0.0, 0.0}, sc_nf[2] = {0.0, 0.0};
    for (int k = 0; k < n_E; ++k) {
      if (!(q[k] > 0.0))
        continue;
      probe.total(mat, E[k], kT, dm);
      const auto& xs = probe.particle().macro_xs();
      // 4 pi for the isotropic direction, V_box for the point
      double f = V_box * 4.0 * PI / (n_E * q[k]);
      sc_f[0] += f * xs.fission * psi_s[k];
      sc_f[1] += f * xs.fission * psi_f[k];
      sc_nf[0] += f * xs.nu_fission * psi_s[k];
      sc_nf[1] += f * xs.nu_fission * psi_f[k];
    }
    for (int y = 0; y < 2; ++y) {
      out.fission[2 * region + y] += sc_f[y] / n_points;
      out.nu_fission[2 * region + y] += sc_nf[y] / n_points;
      out.fission_sq[2 * region + y] += sc_f[y] * sc_f[y];
      out.nu_fission_mg[2 * region + y] += sc_mg[y] / n_points;
      double dlt = sc_nf[y] - sc_mg[y];
      out.diff_sq[2 * region + y] += dlt * dlt;
    }
  }
}

double fission_chi(int i_nuclide, const XsProbe& probe, double E_out)
{
  const Nuclide& nuc = *data::nuclides[i_nuclide];
  if (!nuc.fissionable_)
    return 0.0;
  const double E_in = probe.E();
  if (!nuc.has_partial_fission_)
    return fission_energy_density(i_nuclide, *nuc.fission_rx_[0], E_in, E_out);
  const auto& micro = probe.particle().neutron_xs(i_nuclide);
  if (!(micro.fission > 0.0))
    return 0.0;
  double f = 0.0;
  for (const Reaction* rx : nuc.fission_rx_) {
    double xs = rx->xs(micro);
    if (xs > 0.0)
      f += xs / micro.fission *
           fission_energy_density(i_nuclide, *rx, E_in, E_out);
  }
  return f;
}

double near_kernel(Position x, Position xp, double E, double d_max,
  Position lo, Position hi, KernelScratch& tracer, XsProbe& probe,
  SegmentList& seg)
{
  // Images of xp under reflection in the box faces: per axis, a + 2 n L
  // and (2 lo - a) + 2 n L, with L the width
  const double L[3] = {hi.x - lo.x, hi.y - lo.y, hi.z - lo.z};
  const double a[3] = {xp.x, xp.y, xp.z};
  const double c[3] = {x.x, x.y, x.z};
  const double l[3] = {lo.x, lo.y, lo.z};
  vector<double> img[3];
  for (int ax = 0; ax < 3; ++ax) {
    int n_max = static_cast<int>(std::ceil(d_max / (2.0 * L[ax]))) + 1;
    for (int n = -n_max; n <= n_max; ++n) {
      for (double base : {a[ax], 2.0 * l[ax] - a[ax]}) {
        double v = base + 2.0 * n * L[ax];
        if (std::abs(v - c[ax]) < d_max)
          img[ax].push_back(v);
      }
    }
  }
  double sum = 0.0;
  for (double ix : img[0])
    for (double iy : img[1])
      for (double iz : img[2]) {
        Position d_vec {ix - x.x, iy - x.y, iz - x.z};
        double d = d_vec.norm();
        if (!(d < d_max) || d <= 0.0)
          continue;
        Direction u = d_vec / d;
        // Optical depth along the reflected path of length d, with OpenMC's
        // geometry (surface crossings and reflections; no mesh splitting)
        Particle& g = tracer.scratch();
        g.r() = x;
        g.u() = u;
        g.r_last() = x;
        g.u_last() = u;
        g.wgt() = 1.0;
        g.surface() = SURFACE_NONE;
        g.n_coord() = 1;
        if (!exhaustive_find_cell(g))
          continue;
        double tau = 0.0, done = 0.0;
        for (int crossings = 0; crossings < 100000 && done < d; ++crossings) {
          const Cell& cell = *model::cells[g.lowest_coord().cell()];
          int32_t inst = g.cell_instance();
          int32_t mat = cell.material(inst);
          g.boundary() = distance_to_boundary(g);
          double len = std::min(g.boundary().distance(), d - done);
          if (mat != MATERIAL_VOID)
            tau += probe.total(mat, E, cell.sqrtkT(inst), cell.density_mult(inst)) * len;
          done += len;
          if (done >= d)
            break;
          for (int lev = 0; lev < g.n_coord(); ++lev)
            g.coord(lev).r() += len * g.coord(lev).u();
          g.event_cross_surface();
          if (!g.alive())
            break;
        }
        sum += std::exp(-tau) / (4.0 * PI * d * d);
      }
  return sum;
}

} // namespace rmc
} // namespace openmc
