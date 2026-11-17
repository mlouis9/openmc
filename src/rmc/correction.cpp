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
#include "openmc/rmc/channels.h"
#include "openmc/rmc/mg_control.h"
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
  double X[2], double var[2], int score, double du, double sigma_e,
  int classes)
{
  // score 0: nu Sigma_f. score 1: the net loss cross section, Sigma_t minus
  // the production of every non-fission channel (absorption less the gain
  // of multiplying reactions), in every material. The text below is for
  // score 0; score 1 replaces nu Sigma_f throughout.
  //
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
  ChannelRegistry registry;
  auto score_xs = [&](int32_t m, double dm) {
    if (score == 0)
      return probe.particle().macro_xs().nu_fission;
    double loss = probe.particle().macro_xs().total;
    for (const auto& c : registry.channels(m))
      if (c.kind != ChannelKind::fission)
        loss -= ChannelRegistry::production(c, dm, probe);
    return loss;
  };
  // The proposal's tables are constants of the field: built once per
  // (field, score, hyperfine settings) and kept between calls
  struct Hyper {
    double lo, hi;      //!< energy range of the group
    int n;              //!< hyperfine bins
    vector<int> order;  //!< bins sorted by class of Sigma_t
    vector<int> start;  //!< [classes + 1] first position of each class
    vector<double> cum; //!< cumulative weights in sorted order
    vector<double> Wc;  //!< weight of each class
    vector<double> Sc;  //!< weight-averaged Sigma_t of each class
  };
  struct Tables {
    const FrozenField* field {nullptr};
    int score {-1};
    double du {0.0};
    int classes {0};
    size_t F {0};
    vector<double> wr;
    vector<vector<double>> cdf;
    vector<int> bin_group;                //!< group of each fine bin
    vector<vector<double>> gmass;         //!< [region][group] proposal mass
    std::unordered_map<int64_t, Hyper> hyper; //!< key material * 1024 + group
  };
  static Tables tab;
  classes = std::min(classes, 64);
  if (tab.field != &field || tab.score != score || tab.du != du ||
      tab.classes != classes || tab.F != static_cast<size_t>(F)) {
    tab = Tables();
    tab.field = &field;
    tab.score = score;
    tab.du = du;
    tab.classes = classes;
    tab.F = F;
    tab.wr.assign(R, 0.0);
    tab.cdf.resize(R);
    tab.gmass.resize(R);
    tab.bin_group.resize(F);
    const int G = field.n_groups();
    for (int i = 0; i < F; ++i)
      tab.bin_group[i] = field.group(std::sqrt(fe[i] * fe[i + 1]));
    // The score's cross section at the fine-bin midpoints, per material
    std::unordered_map<int32_t, vector<double>> nsf;
    for (int64_t r = 0; r < R; ++r) {
      int32_t m = field.region_material(r);
      if (m == MATERIAL_VOID ||
          (score == 0 && !model::materials[m]->fissionable()))
        continue;
      auto it = nsf.find(m);
      if (it == nsf.end()) {
        vector<double> v(F);
        for (int i = 0; i < F; ++i) {
          probe.total(m, std::sqrt(fe[i] * fe[i + 1]), sqrtkT0, 1.0);
          v[i] = std::max(0.0, score_xs(m, 1.0));
        }
        it = nsf.emplace(m, std::move(v)).first;
        // Hyperfine tables of the groups inside the resonance range: bins
        // of lethargy width du with the weight of the narrow-resonance flux,
        // du Sigma(E) / (Sigma_t(E) + sigma_e)
        if (du > 0.0) {
          for (int g = 0; g < G; ++g) {
            double lo = 0.0, hi = 0.0;
            for (int i = 0; i < F; ++i)
              if (tab.bin_group[i] == g) {
                if (lo == 0.0)
                  lo = fe[i];
                hi = fe[i + 1];
              }
            if (!(lo >= 0.5 && hi <= 2.0e5))
              continue;
            int n = std::max(1, static_cast<int>(std::ceil(std::log(hi / lo) / du)));
            Hyper h;
            h.lo = lo;
            h.hi = hi;
            h.n = n;
            vector<double> v(n), st(n);
            double s_lo = INFTY, s_hi = 0.0;
            for (int j = 0; j < n; ++j) {
              double Em = lo * std::exp(std::log(hi / lo) * (j + 0.5) / n);
              probe.total(m, Em, sqrtkT0, 1.0);
              st[j] = probe.particle().macro_xs().total;
              v[j] = std::max(0.0, score_xs(m, 1.0)) + 1e-12;
              s_lo = std::min(s_lo, st[j]);
              s_hi = std::max(s_hi, st[j]);
            }
            // Classes of Sigma_t, equal in its logarithm
            const int C = std::max(1, classes);
            vector<int> cls(n);
            h.start.assign(C + 1, 0);
            h.Wc.assign(C, 0.0);
            h.Sc.assign(C, 0.0);
            double span = std::log(s_hi / s_lo);
            for (int j = 0; j < n; ++j) {
              int c = (span > 0.0)
                        ? std::min(C - 1, static_cast<int>(C * std::log(st[j] / s_lo) / span))
                        : 0;
              cls[j] = c;
              ++h.start[c + 1];
              h.Wc[c] += v[j];
              h.Sc[c] += v[j] * st[j];
            }
            for (int c = 0; c < C; ++c) {
              if (h.Wc[c] > 0.0)
                h.Sc[c] /= h.Wc[c];
              h.start[c + 1] += h.start[c];
            }
            h.order.resize(n);
            h.cum.resize(n);
            vector<int> fill(h.start.begin(), h.start.end() - 1);
            for (int j = 0; j < n; ++j)
              h.order[fill[cls[j]]++] = j;
            for (int c = 0; c < C; ++c) {
              double acc = 0.0;
              for (int q = h.start[c]; q < h.start[c + 1]; ++q) {
                acc += v[h.order[q]];
                h.cum[q] = acc;
              }
            }
            tab.hyper.emplace(static_cast<int64_t>(m) * 1024 + g, std::move(h));
          }
        }
      }
      tab.cdf[r].resize(F);
      tab.gmass[r].assign(G, 0.0);
      double c = 0.0;
      for (int i = 0; i < F; ++i) {
        double Em = std::sqrt(fe[i] * fe[i + 1]);
        int g = tab.bin_group[i];
        double phi = (g >= 0) ? field.flux(r, g) *
                                  field.shape(FrozenField::Shape::phi, r, Em)
                              : 0.0;
        double mass = field.volume(r) * phi * (fe[i + 1] - fe[i]) * it->second[i];
        c += mass;
        if (g >= 0)
          tab.gmass[r][g] += mass;
        tab.cdf[r][i] = c;
      }
      tab.wr[r] = c;
    }
  }
  const auto& wr = tab.wr;
  const auto& cdf = tab.cdf;
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
    double mu = 2.0 * prn(&s) - 1.0, ph = 2.0 * PI * prn(&s);
    double st = std::sqrt(std::max(0.0, 1.0 - mu * mu));
    Direction u {st * std::cos(ph), st * std::sin(ph), mu};
    // Length of this cell's material behind the point, along the path
    double d_first = 0.0;
    if (du > 0.0) {
      GeometryState gs;
      gs.init_from_r_u(rd.value, u);
      if (exhaustive_find_cell(gs))
        d_first = distance_to_boundary(gs).distance();
    }
    for (int k = 0; k < n_E; ++k) {
      double x = prn(&s) * wr[region];
      int b = static_cast<int>(std::upper_bound(c.begin(), c.end(), x) - c.begin());
      b = std::min(b, F - 1);
      double mass = c[b] - (b > 0 ? c[b - 1] : 0.0);
      int g = tab.bin_group[b];
      auto hy = (du > 0.0 && g >= 0)
                  ? tab.hyper.find(static_cast<int64_t>(mat) * 1024 + g)
                  : tab.hyper.end();
      if (hy != tab.hyper.end()) {
        // The group by its lifted mass; the class of Sigma_t by the flux a
        // path with d_first cm of this material behind it would see,
        //   W_c [(1 - exp(-S_c d)) / S_c + exp(-S_c d) / sigma_e];
        // the energy within the class by the score's cross section
        const Hyper& h = hy->second;
        const int C = static_cast<int>(h.Wc.size());
        double pc[64], tot = 0.0;
        for (int c = 0; c < C; ++c) {
          double S = h.Sc[c], t = std::exp(-S * d_first);
          pc[c] = (h.Wc[c] > 0.0) ? h.Wc[c] * ((1.0 - t) / S + t / sigma_e) : 0.0;
          tot += pc[c];
        }
        double y = prn(&s) * tot;
        int c = 0;
        for (; c < C - 1; ++c) {
          if (y < pc[c])
            break;
          y -= pc[c];
        }
        while (c > 0 && !(pc[c] > 0.0))
          --c;
        double z = prn(&s) * h.Wc[c];
        auto first = h.cum.begin() + h.start[c], last = h.cum.begin() + h.start[c + 1];
        int pos = static_cast<int>(std::upper_bound(first, last, z) - h.cum.begin());
        pos = std::min(pos, h.start[c + 1] - 1);
        int j = h.order[pos];
        double wj = h.cum[pos] - (pos > h.start[c] ? h.cum[pos - 1] : 0.0);
        double lr = std::log(h.hi / h.lo);
        double e0 = h.lo * std::exp(lr * j / h.n), e1 = h.lo * std::exp(lr * (j + 1) / h.n);
        E[k] = e0 + (e1 - e0) * prn(&s);
        q[k] = tab.gmass[region][g] / wr[region] * (pc[c] / tot) * wj / h.Wc[c] / (e1 - e0);
      } else {
        E[k] = fe[b] + (fe[b + 1] - fe[b]) * prn(&s);
        q[k] = mass / wr[region] / (fe[b + 1] - fe[b]);
      }
    }
    backward.evaluate_all(
      rd.value, u, E, nullptr, tail, &s, psi_s, psi_f, mg_s, mg_f);
    double sc[2] = {0.0, 0.0};
    for (int k = 0; k < n_E; ++k) {
      if (!(q[k] > 0.0))
        continue;
      probe.total(mat, E[k], kT, dm);
      double f = score_xs(mat, dm) * 4.0 * PI / (rd.density * q[k] * n_E);
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

void field_first_flight_rates(const FrozenField& field, int64_t n_rays,
  int n_E, const TailOptions& tail, uint64_t seed, double du, double sigma_e,
  double rate[4], double var[4], const MGTables* mg, double mg_rate[4],
  double diff_var[4])
{
  // With mg, the same rays are also scored with the multigroup cross
  // sections of the energy's group (mg_rate), and diff_var is the variance
  // of the mean of the CE score minus the multigroup score. The multigroup
  // rates of the field's own emission are known exactly (the field is their
  // fixed point), so rate - mg_rate + exact is a correlated estimate.
  // rate[2 * tag + x] = int dr dE dOmega Q_tag(r, E) / (4 pi) A_x(r, Omega, E),
  // x = 0 nu-fission, 1 net loss, with A_x the expected first-collision
  // score of a neutron born at r towards Omega at E:
  //   A_x = sum over the ray's segments of (Sigma_x / Sigma_t)
  //         (exp(-tau_in) - exp(-tau_out)).
  // The ray is integrated analytically, so the position of the collision
  // carries no noise, and it needs only the material of each segment. The
  // source point is drawn from the field's own emission times a rational
  // approximation of the loss probability,
  //   p(E) = Sigma_loss,F / (Sigma_t,F + sigma_e)
  //          + sigma_e / (Sigma_t,F + sigma_e) (Sigma_loss / Sigma_t)_other,
  // F the fissionable material, tabulated on sub-bins of lethargy width du
  // of the lifts' fine bins. n_E energies share a ray.
  const int64_t R = field.n_regions();
  const auto& fe = field.fine_edges();
  const int F = static_cast<int>(fe.size()) - 1;
  const double sqrtkT0 = std::sqrt(K_BOLTZMANN * 293.6);
  XsProbe probe;
  ChannelRegistry registry;
  auto loss_xs = [&](int32_t m, double dm) {
    // Below 1 MeV no reaction multiplies neutrons: the loss is absorption
    if (probe.E() < 1.0e6)
      return probe.particle().macro_xs().absorption;
    double loss = probe.particle().macro_xs().total;
    for (const auto& c : registry.channels(m))
      if (c.kind != ChannelKind::fission)
        loss -= ChannelRegistry::production(c, dm, probe);
    return loss;
  };
  struct Tables {
    const FrozenField* field {nullptr};
    double du {0.0}, sigma_e {0.0};
    size_t F {0};
    vector<int> start;        //!< [F + 1] first sub-bin of each fine bin
    vector<double> e_lo, e_hi; //!< sub-bin edges
    vector<double> p;         //!< proposal factor of each sub-bin
    vector<double> cum;       //!< cumulative p dE within each fine bin
    vector<int> bin_group;
    vector<double> wr;
    vector<double> wr_opt;      //!< region weights after the pilot
    vector<vector<double>> cdf; //!< [region][fine bin]
  };
  static Tables tab;
  if (tab.field != &field || tab.du != du || tab.sigma_e != sigma_e ||
      tab.F != static_cast<size_t>(F)) {
    tab = Tables();
    tab.field = &field;
    tab.du = du;
    tab.sigma_e = sigma_e;
    tab.F = F;
    vector<int32_t> fis, other;
    for (int64_t r = 0; r < R; ++r) {
      int32_t m = field.region_material(r);
      if (m == MATERIAL_VOID)
        continue;
      auto& v = model::materials[m]->fissionable() ? fis : other;
      if (std::find(v.begin(), v.end(), m) == v.end())
        v.push_back(m);
    }
    tab.start.assign(F + 1, 0);
    tab.bin_group.resize(F);
    vector<double> S(F, 0.0);
    for (int i = 0; i < F; ++i) {
      tab.bin_group[i] = field.group(std::sqrt(fe[i] * fe[i + 1]));
      double lr = std::log(fe[i + 1] / std::max(fe[i], 1e-11));
      int n = (fe[i] > 0.0 && du > 0.0)
                ? std::max(1, static_cast<int>(std::ceil(lr / du)))
                : 1;
      n = std::min(n, 4096);
      double c = 0.0;
      for (int j = 0; j < n; ++j) {
        double e0 = (fe[i] > 0.0) ? fe[i] * std::exp(lr * j / n)
                                  : fe[i] + (fe[i + 1] - fe[i]) * j / n;
        double e1 = (fe[i] > 0.0) ? fe[i] * std::exp(lr * (j + 1) / n)
                                  : fe[i] + (fe[i + 1] - fe[i]) * (j + 1) / n;
        double Em = 0.5 * (e0 + e1);
        double pf = 0.0, stf = 0.0;
        for (int32_t m : fis) {
          probe.total(m, Em, sqrtkT0, 1.0);
          double st = probe.particle().macro_xs().total;
          pf += std::max(0.0, loss_xs(m, 1.0)) / (st + sigma_e);
          stf += st;
        }
        if (!fis.empty()) {
          pf /= fis.size();
          stf /= fis.size();
        }
        double po = 0.0;
        for (int32_t m : other) {
          probe.total(m, Em, sqrtkT0, 1.0);
          double st = probe.particle().macro_xs().total;
          po += (st > 0.0) ? std::max(0.0, loss_xs(m, 1.0)) / st : 0.0;
        }
        if (!other.empty())
          po /= other.size();
        double pj = pf + (fis.empty() ? 1.0 : sigma_e / (stf + sigma_e)) * po + 1e-9;
        c += pj * (e1 - e0);
        tab.e_lo.push_back(e0);
        tab.e_hi.push_back(e1);
        tab.p.push_back(pj);
        tab.cum.push_back(c);
      }
      S[i] = c;
      tab.start[i + 1] = tab.start[i] + n;
    }
    tab.wr.assign(R, 0.0);
    tab.cdf.resize(R);
    for (int64_t r = 0; r < R; ++r) {
      if (field.region_material(r) == MATERIAL_VOID)
        continue;
      tab.cdf[r].resize(F);
      double c = 0.0;
      for (int i = 0; i < F; ++i) {
        int g = tab.bin_group[i];
        double Em = std::sqrt(std::max(fe[i], 1e-11) * fe[i + 1]);
        double q = (g >= 0) ? field.evaluate(r, Tag::s, Em) +
                                field.evaluate(r, Tag::f, Em)
                            : 0.0;
        c += std::max(0.0, q) * S[i];
        tab.cdf[r][i] = c;
      }
      tab.wr[r] = c * field.volume(r);
    }
  }
  KernelScratch scratch;
  const double survive = 0.25;
  struct Xs {
    int32_t mat {-2};
    double total, ratio[2];
    double mg_total, mg_ratio[2];
  };
  const int G = field.n_groups();
  vector<double> E(n_E), wE(n_E), T(n_E), acc(2 * n_E), Tm(n_E), accm(2 * n_E);
  vector<int> live(n_E);
  vector<Xs> xs(4 * n_E);
  // One ray: its region and its four scores
  auto ray = [&](const VoxelProposal& voxel, uint64_t s, int64_t& region,
               double sc[4], double scm[4]) {
    auto rd = voxel.sample(&s, region);
    int32_t mat0;
    double kT0, dm0;
    if (!locate_point(field, rd.value, region, mat0, kT0, dm0) || region < 0)
      fatal_error("RMC first-flight rates: point outside the field.");
    const auto& c = tab.cdf[region];
    const double tot = c[F - 1];
    double share[2][64];
    for (int k = 0; k < n_E; ++k) {
      double x = prn(&s) * tot;
      int b = static_cast<int>(std::upper_bound(c.begin(), c.end(), x) - c.begin());
      b = std::min(b, F - 1);
      int j0 = tab.start[b], j1 = tab.start[b + 1];
      double top = tab.cum[j1 - 1];
      double y = prn(&s) * top;
      int j = static_cast<int>(
        std::upper_bound(tab.cum.begin() + j0, tab.cum.begin() + j1, y) -
        tab.cum.begin());
      j = std::min(j, j1 - 1);
      E[k] = tab.e_lo[j] + (tab.e_hi[j] - tab.e_lo[j]) * prn(&s);
      double qs = std::max(0.0, field.evaluate(region, Tag::s, E[k]));
      double qf = std::max(0.0, field.evaluate(region, Tag::f, E[k]));
      // Proposal density per unit energy given the region, from the bin's
      // tabulated emission (at its midpoint) and the sub-bin's factor
      double Em = std::sqrt(std::max(fe[b], 1e-11) * fe[b + 1]);
      double qb = std::max(0.0, field.evaluate(region, Tag::s, Em) +
                                  field.evaluate(region, Tag::f, Em));
      double dens = qb * tab.p[j] / tot;
      // 4 pi Q_t(r, E) / (spatial density * energy density * n_E)
      double w = (dens > 0.0) ? 4.0 * PI / (rd.density * dens * n_E) : 0.0;
      share[0][k] = w * qs;
      share[1][k] = w * qf;
      wE[k] = w;
      T[k] = 1.0;
      Tm[k] = 1.0;
      live[k] = 1;
      acc[2 * k] = acc[2 * k + 1] = 0.0;
      accm[2 * k] = accm[2 * k + 1] = 0.0;
      for (int a = 0; a < 4; ++a)
        xs[4 * k + a].mat = -2;
    }
    double mu = 2.0 * prn(&s) - 1.0, ph = 2.0 * PI * prn(&s);
    double st = std::sqrt(std::max(0.0, 1.0 - mu * mu));
    Direction u {st * std::cos(ph), st * std::sin(ph), mu};
    Particle& g = scratch.scratch();
    g.r() = rd.value;
    g.u() = u;
    g.r_last() = rd.value;
    g.u_last() = u;
    g.wgt() = 1.0;
    g.surface() = SURFACE_NONE;
    g.n_coord() = 1;
    if (!exhaustive_find_cell(g))
      fatal_error("RMC first-flight rates: source point not in a cell.");
    int n_live = n_E;
    for (int64_t crossings = 0; crossings < tail.max_segments && n_live > 0;
         ++crossings) {
      const Cell& cell = *model::cells[g.lowest_coord().cell()];
      int32_t inst = g.cell_instance();
      int32_t m = cell.material(inst);
      g.boundary() = distance_to_boundary(g);
      double len = g.boundary().distance();
      if (m != MATERIAL_VOID) {
        for (int k = 0; k < n_E; ++k) {
          if (!live[k])
            continue;
          Xs* e = nullptr;
          for (int a = 0; a < 4; ++a) {
            Xs& cand = xs[4 * k + a];
            if (cand.mat == m) {
              e = &cand;
              break;
            }
            if (cand.mat == -2 || a == 3) {
              cand.mat = m;
              probe.total(m, E[k], cell.sqrtkT(inst), cell.density_mult(inst));
              const auto& mx = probe.particle().macro_xs();
              cand.total = mx.total;
              cand.ratio[0] = (mx.total > 0.0) ? mx.nu_fission / mx.total : 0.0;
              cand.ratio[1] = (mx.total > 0.0)
                                ? loss_xs(m, cell.density_mult(inst)) / mx.total
                                : 0.0;
              if (mg) {
                int gi = field.group(E[k]);
                double st = mg->sigma_t(m, cell.sqrtkT(inst), gi);
                double out = 0.0;
                for (int go = 0; go < G; ++go)
                  out += mg->nu_sigma_s(m, cell.sqrtkT(inst), gi, go);
                cand.mg_total = st;
                cand.mg_ratio[0] =
                  (st > 0.0) ? mg->nu_sigma_f(m, cell.sqrtkT(inst), gi) / st : 0.0;
                cand.mg_ratio[1] = (st > 0.0) ? (st - out) / st : 0.0;
              }
              e = &cand;
              break;
            }
          }
          double t_out = T[k] * std::exp(-e->total * len);
          double d = T[k] - t_out;
          acc[2 * k] += e->ratio[0] * d;
          acc[2 * k + 1] += e->ratio[1] * d;
          T[k] = t_out;
          if (mg) {
            double tm = Tm[k] * std::exp(-e->mg_total * len);
            accm[2 * k] += e->mg_ratio[0] * (Tm[k] - tm);
            accm[2 * k + 1] += e->mg_ratio[1] * (Tm[k] - tm);
            Tm[k] = tm;
          }
          // The roulette acts on both transmissions with one random number
          if (std::abs(T[k]) < tail.cutoff &&
              (!mg || std::abs(Tm[k]) < tail.cutoff)) {
            if (prn(&s) < survive) {
              T[k] /= survive;
              Tm[k] /= survive;
            } else {
              live[k] = 0;
              --n_live;
            }
          }
        }
      }
      if (n_live == 0)
        break;
      for (int lev = 0; lev < g.n_coord(); ++lev)
        g.coord(lev).r() += len * g.coord(lev).u();
      g.event_cross_surface();
      if (!g.alive())
        break;
    }
    for (int a = 0; a < 4; ++a)
      sc[a] = scm[a] = 0.0;
    for (int k = 0; k < n_E; ++k)
      for (int t = 0; t < 2; ++t)
        for (int x = 0; x < 2; ++x) {
          sc[2 * t + x] += share[t][k] * acc[2 * k + x];
          scm[2 * t + x] += share[t][k] * accm[2 * k + x];
        }
  };
  // Pilot, once per field: the root mean square of the total loss score by
  // region. Regions are then drawn in proportion to their emission-based
  // weight times that, which is the best allocation for the loss
  if (tab.wr_opt.empty()) {
    VoxelOptions vo;
    vo.mode = VoxelMode::tabulated;
    vo.weights = tab.wr;
    VoxelProposal voxel(field, vo);
    vector<double> s2(R, 0.0);
    vector<int64_t> cnt(R, 0);
    const int64_t n_pilot = 20 * R + 20000;
    for (int64_t i = 0; i < n_pilot; ++i) {
      int64_t region;
      double sc[4], scm[4];
      ray(voxel, future_seed(static_cast<uint64_t>(i) * 1000003ULL + 17, 987654321ULL),
        region, sc, scm);
      double v = sc[1] + sc[3];
      s2[region] += v * v;
      ++cnt[region];
    }
    double mean_rms = 0.0;
    int64_t n_r = 0;
    for (int64_t r = 0; r < R; ++r)
      if (cnt[r] > 0) {
        mean_rms += std::sqrt(s2[r] / cnt[r]);
        ++n_r;
      }
    mean_rms = (n_r > 0) ? mean_rms / n_r : 1.0;
    tab.wr_opt = tab.wr;
    for (int64_t r = 0; r < R; ++r) {
      // Regions with few pilot rays keep the average factor; a floor keeps
      // every region's density positive
      double f = (cnt[r] >= 20) ? std::sqrt(s2[r] / cnt[r]) : mean_rms;
      tab.wr_opt[r] *= std::max(f, 0.05 * mean_rms);
    }
  }
  VoxelOptions vo;
  vo.mode = VoxelMode::tabulated;
  vo.weights = tab.wr_opt;
  VoxelProposal voxel(field, vo);
  double sum[4] = {0.0, 0.0, 0.0, 0.0}, sq[4] = {0.0, 0.0, 0.0, 0.0};
  double summ[4] = {0.0, 0.0, 0.0, 0.0}, sqd[4] = {0.0, 0.0, 0.0, 0.0};
  for (int64_t i = 0; i < n_rays; ++i) {
    int64_t region;
    double sc[4], scm[4];
    ray(voxel, future_seed(static_cast<uint64_t>(i) * 1000003ULL + 13, seed), region, sc,
      scm);
    for (int a = 0; a < 4; ++a) {
      sum[a] += sc[a];
      sq[a] += sc[a] * sc[a];
      summ[a] += scm[a];
      sqd[a] += (sc[a] - scm[a]) * (sc[a] - scm[a]);
    }
  }
  if (mg)
    for (int a = 0; a < 4; ++a) {
      mg_rate[a] = summ[a] / n_rays;
      double dm = (sum[a] - summ[a]) / n_rays;
      diff_var[a] = (n_rays > 1) ? (sqd[a] / n_rays - dm * dm) / (n_rays - 1) : 0.0;
    }
  for (int a = 0; a < 4; ++a) {
    rate[a] = sum[a] / n_rays;
    var[a] = (n_rays > 1)
               ? (sq[a] / n_rays - rate[a] * rate[a]) / (n_rays - 1)
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

void connection(const FrozenField& field, const EmissionRecord& rec, Position x,
  int32_t mat, double sqrtkT, double dm, int l, double E_f, double d_max,
  double k, KernelScratch& tracer, XsProbe& probe, double& q, double& J)
{
  q = J = 0.0;
  Position lo = field.box_lower_left(), hi = field.box_upper_right();
  const double L[3] = {hi.x - lo.x, hi.y - lo.y, hi.z - lo.z};
  const double a[3] = {rec.r.x, rec.r.y, rec.r.z};
  const double c[3] = {x.x, x.y, x.z};
  const double lw[3] = {lo.x, lo.y, lo.z};
  // Images of the record's site under reflection in the box faces, with the
  // parity of each (an odd image flips that component of the departure
  // direction)
  vector<double> img[3];
  vector<int> par[3];
  for (int ax = 0; ax < 3; ++ax) {
    int n_max = static_cast<int>(std::ceil(d_max / (2.0 * L[ax]))) + 1;
    for (int n = -n_max; n <= n_max; ++n) {
      for (int odd = 0; odd < 2; ++odd) {
        double v = (odd ? 2.0 * lw[ax] - a[ax] : a[ax]) + 2.0 * n * L[ax];
        if (std::abs(v - c[ax]) < d_max) {
          img[ax].push_back(v);
          par[ax].push_back(odd);
        }
      }
    }
  }
  const double awr = (rec.kind == 1) ? data::nuclides[rec.nuclide]->awr_ : 0.0;
  const double kT0 = K_BOLTZMANN * 293.6;
  for (size_t ix = 0; ix < img[0].size(); ++ix)
    for (size_t iy = 0; iy < img[1].size(); ++iy)
      for (size_t iz = 0; iz < img[2].size(); ++iz) {
        Position d_vec {img[0][ix] - x.x, img[1][iy] - x.y, img[2][iz] - x.z};
        double d = d_vec.norm();
        if (!(d < d_max) || d <= 0.0)
          continue;
        Direction u = d_vec / d; // from the candidate back to the image
        // Departure direction at the real site: travel is along -u in the
        // unfolded picture; odd images flip the component
        Direction dep {par[0][ix] ? u.x : -u.x, par[1][iy] ? u.y : -u.y,
          par[2][iz] ? u.z : -u.z};
        double e, ang;
        if (rec.kind == 0) {
          e = rec.E;
          ang = 1.0 / (4.0 * PI);
        } else {
          double mu = std::max(-1.0, std::min(1.0, rec.u.dot(dep)));
          double root = awr * awr - 1.0 + mu * mu;
          if (root < 0.0)
            continue;
          double gam = (mu + std::sqrt(root)) / (awr + 1.0);
          gam *= gam;
          e = rec.E * gam;
          // p_lab(mu | E_in) from the inverse evaluator at the outgoing
          // energy: its density is p_lab / gamma
          auto im = elastic_stationary(rec.nuclide, kT0, e, mu);
          ang = im.density * gam / (2.0 * PI);
          if (!(ang > 0.0))
            continue;
        }
        // Optical depth along the reflected path of length d at energy e
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
        int32_t c_mat[2] = {-2, -2};
        double c_sig[2] = {0.0, 0.0};
        for (int crossings = 0; crossings < 100000 && done < d; ++crossings) {
          const Cell& cell = *model::cells[g.lowest_coord().cell()];
          int32_t inst = g.cell_instance();
          int32_t m = cell.material(inst);
          g.boundary() = distance_to_boundary(g);
          double len = std::min(g.boundary().distance(), d - done);
          if (m != MATERIAL_VOID) {
            int slot = (c_mat[0] == m) ? 0 : (c_mat[1] == m) ? 1 : -1;
            if (slot < 0) {
              slot = (c_mat[0] == -2) ? 0 : 1;
              c_mat[slot] = m;
              c_sig[slot] =
                probe.total(m, e, cell.sqrtkT(inst), cell.density_mult(inst));
            }
            tau += c_sig[slot] * len;
          }
          done += len;
          if (done >= d)
            break;
          for (int lev = 0; lev < g.n_coord(); ++lev)
            g.coord(lev).r() += len * g.coord(lev).u();
          g.event_cross_surface();
          if (!g.alive())
            break;
        }
        // Fission of nuclide l at the candidate, at the arriving energy
        probe.total(mat, e, sqrtkT, dm);
        const auto& micro = probe.particle().neutron_xs(l);
        if (!(micro.fission > 0.0))
          continue;
        double chi = fission_chi(l, probe, E_f);
        double base = ang * std::exp(-tau) / (d * d) * micro.fission * chi;
        q += base;
        J += rec.w * base * micro.nu_fission / micro.fission / k;
      }
}

bool grouped_fission_weight(const FrozenField& field,
  const vector<EmissionRecord>& group, Position x, int l, double E_f,
  double d_max, double k, KernelScratch& tracer, XsProbe& probe, double& W)
{
  int64_t region;
  int32_t mat;
  double kT, dm;
  if (!locate_point(field, x, region, mat, kT, dm) || mat == MATERIAL_VOID)
    return false;
  double num = 0.0, den = 0.0;
  for (const auto& rec : group) {
    double q, J;
    connection(field, rec, x, mat, kT, dm, l, E_f, d_max, k, tracer, probe, q, J);
    num += J;
    den += rec.m * q;
  }
  if (!(den > 0.0))
    return false;
  W = num / den;
  return true;
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
