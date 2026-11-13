#include "openmc/rmc/backward.h"

#include <cmath>

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
  const size_t n = E.size();
  psi_s.assign(n, 0.0);
  psi_f.assign(n, 0.0);
  T_.assign(n, 1.0);
  alive_.assign(n, 1);
  size_t n_alive = n;
  double travelled = 0.0;
  int64_t n_seg = 0, n_lookup = 0;
  bool hit_limit = false;

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

      for (size_t e = 0; e < n; ++e) {
        if (!alive_[e])
          continue;
        double sigma = probe_.total(
          chunk_.material[k], E[e], chunk_.sqrtkT[k], chunk_.density_mult[k]);
        ++n_lookup;
        double w = T_[e] * attenuation_integral(sigma, l);
        if (region >= 0) {
          psi_s[e] += w * field_.evaluate(region, Tag::s, E[e]);
          psi_f[e] += w * field_.evaluate(region, Tag::f, E[e]);
        }
        T_[e] *= std::exp(-sigma * l);

        // Unbiased roulette: survive with probability T / cutoff and carry
        // on with T = cutoff, so the expected remaining tail is unchanged
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
