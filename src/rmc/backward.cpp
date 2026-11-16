#include "openmc/rmc/backward.h"

#include <cmath>

#include "openmc/constants.h"
#include "openmc/random_lcg.h"

namespace openmc {
namespace rmc {

double attenuation_integral(double sigma, double length)
{
  double x = sigma * length;
  if (x < 1e-6) {
    // Series of (1 - e^-x) / x, accurate to round-off for x < 1e-6
    return length * (1.0 - 0.5 * x + x * x / 6.0);
  }
  return -std::expm1(-x) / sigma;
}

void BackwardFlux::evaluate(Position r, Direction u_back,
  const vector<double>& E, double a_s, double a_f, const TailOptions& opt,
  uint64_t* seed, vector<double>& psi, PathStats* stats)
{
  evaluate_tags(r, u_back, E, opt, seed, psi_s_, psi_f_, stats);
  psi.resize(E.size());
  for (size_t e = 0; e < E.size(); ++e)
    psi[e] = a_s * psi_s_[e] + a_f * psi_f_[e];
}

void BackwardFlux::evaluate_tags(Position r, Direction u_back,
  const vector<double>& E, const TailOptions& opt, uint64_t* seed,
  vector<double>& psi_s, vector<double>& psi_f, PathStats* stats)
{
  evaluate_all(
    r, u_back, E, nullptr, opt, seed, psi_s, psi_f, mg_s_, mg_f_, stats);
}

void BackwardFlux::evaluate_all(Position r, Direction u_back,
  const vector<double>& E, const MGTables* mg, const TailOptions& opt,
  uint64_t* seed, vector<double>& psi_s, vector<double>& psi_f,
  vector<double>& mg_s, vector<double>& mg_f, PathStats* stats)
{
  // Channels 0..n_ce-1 are CE energies, then the MG groups (ascending)
  const size_t n_ce = E.size();
  const int G = mg ? field_.n_groups() : 0;
  const size_t n = n_ce + G;
  psi_s.assign(n_ce, 0.0);
  psi_f.assign(n_ce, 0.0);
  mg_s.assign(G, 0.0);
  mg_f.assign(G, 0.0);
  T_.assign(n, 1.0);
  alive_.assign(n, 1);
  size_t n_alive = n;
  double travelled = 0.0;
  int64_t n_seg = 0, n_lookup = 0;
  bool hit_limit = false;
  if (n == 0)
    return;
  sig_.assign(n_ce * XS_CACHE, -1.0);
  states_.clear();
  live_.resize(n_ce);
  for (size_t e = 0; e < n_ce; ++e)
    live_[e] = e;
  int sid = -1;
  g_bin_.resize(n_ce);
  i_bin_.resize(n_ce);
  for (size_t e = 0; e < n_ce; ++e)
    field_.energy_bins(E[e], g_bin_[e], i_bin_[e]);

  tracer_.begin(r, u_back);
  while (n_alive > 0) {
    tracer_.extend(opt.chunk, chunk_);
    for (int k = 0; k < chunk_.size() && n_alive > 0; ++k) {
      double l = chunk_.length[k];
      // The fixed-length tail stops at exactly opt.length
      bool last = false;
      if (opt.mode == Tail::fixed_length && travelled + l >= opt.length) {
        l = opt.length - travelled;
        last = true;
      }
      travelled += l;
      ++n_seg;
      int64_t region = chunk_.region[k];
      int32_t mat = chunk_.material[k];

      // Material state of the segment: an index into the few states this
      // path has met, so that Sigma_t of each energy is looked up once per
      // state
      const double kT = chunk_.sqrtkT[k], dm = chunk_.density_mult[k];
      if (n_ce > 0 && (sid < 0 || states_[sid].mat != mat ||
                        states_[sid].sqrtkT != kT || states_[sid].dm != dm)) {
        sid = -1;
        for (int q = 0; q < static_cast<int>(states_.size()); ++q)
          if (states_[q].mat == mat && states_[q].sqrtkT == kT &&
              states_[q].dm == dm) {
            sid = q;
            break;
          }
        if (sid < 0) {
          if (static_cast<int>(states_.size()) == XS_CACHE) {
            states_.clear();
            std::fill(sig_.begin(), sig_.end(), -1.0);
          }
          sid = static_cast<int>(states_.size());
          states_.push_back({mat, kT, dm, 0.0});
        }
      }

      // CE energies still alive
      for (size_t a = 0; a < live_.size();) {
        const size_t e = live_[a];
        double& sg = sig_[e * XS_CACHE + sid];
        if (sg < 0.0) {
          sg = probe_.total(mat, E[e], kT, dm);
          ++n_lookup;
        }
        const double sigma = sg;
        const double x = sigma * l;
        double integral, ex;
        if (x < 1e-4) {
          // Series of (1 - e^-x) / x and of e^-x, to round-off for x < 1e-4
          integral = l * (1.0 - 0.5 * x + x * x / 6.0);
          ex = 1.0 - x + 0.5 * x * x - x * x * x / 6.0;
        } else {
          ex = std::exp(-x);
          integral = (1.0 - ex) / sigma;
        }
        if (region >= 0) {
          const double w = T_[e] * integral;
          psi_s[e] += w * field_.evaluate_bins(region, Tag::s, g_bin_[e], i_bin_[e]);
          psi_f[e] += w * field_.evaluate_bins(region, Tag::f, g_bin_[e], i_bin_[e]);
        }
        T_[e] *= ex;
        // Unbiased roulette: survive with probability T / cutoff and carry
        // on with T = cutoff, so the expected remaining tail is unchanged
        if (opt.mode == Tail::roulette && T_[e] < opt.cutoff) {
          if (prn(seed) < T_[e] / opt.cutoff) {
            T_[e] = opt.cutoff;
          } else {
            alive_[e] = 0;
            --n_alive;
            live_[a] = live_.back();
            live_.pop_back();
            continue;
          }
        }
        ++a;
      }

      // MG groups
      for (size_t e = n_ce; e < n; ++e) {
        if (!alive_[e])
          continue;
        // MG tables are for unit density; the cell instance scales them
        double sigma =
          mg->sigma_t(mat, chunk_.sqrtkT[k], static_cast<int>(e - n_ce)) *
          chunk_.density_mult[k];
        double x = sigma * l;
        double em1 = (x < 1e-6) ? -x * (1.0 - 0.5 * x + x * x / 6.0) : std::expm1(-x);
        double integral = (x < 1e-6) ? l * (1.0 - 0.5 * x + x * x / 6.0) : -em1 / sigma;
        double w = T_[e] * integral;
        if (region >= 0) {
          // Group-integrated, isotropic: Q_g / (4 pi)
          int g = static_cast<int>(e - n_ce);
          mg_s[g] += w * field_.emission(region, Tag::s, g) / (4.0 * PI);
          mg_f[g] += w * field_.emission(region, Tag::f, g) / (4.0 * PI);
        }
        T_[e] *= 1.0 + em1;
        if (opt.mode == Tail::roulette && T_[e] < opt.cutoff) {
          if (prn(seed) < T_[e] / opt.cutoff) {
            T_[e] = opt.cutoff;
          } else {
            alive_[e] = 0;
            --n_alive;
          }
        }
      }
      if (last)
        n_alive = 0;
    }
    if (chunk_.ends_in_vacuum)
      break;
    if (n_seg >= opt.max_segments) {
      hit_limit = n_alive > 0;
      break;
    }
  }
  if (stats) {
    stats->segments = n_seg;
    stats->lookups = n_lookup;
    stats->hit_limit = hit_limit;
  }
}

} // namespace rmc
} // namespace openmc
