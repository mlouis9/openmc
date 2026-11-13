#include "openmc/distribution_energy.h"

#include <algorithm> // for max, min, copy, move, upper_bound
#include <cmath>
#include <cstddef>   // for size_t
#include <iterator>  // for back_inserter

#include "openmc/tensor.h"

#include "openmc/constants.h"
#include "openmc/endf.h"
#include "openmc/error.h"
#include "openmc/hdf5_interface.h"
#include "openmc/math_functions.h"
#include "openmc/random_dist.h"
#include "openmc/random_lcg.h"

namespace openmc {

//==============================================================================
// DiscretePhoton implementation
//==============================================================================

DiscretePhoton::DiscretePhoton(hid_t group)
{
  read_attribute(group, "primary_flag", primary_flag_);
  read_attribute(group, "energy", energy_);
  read_attribute(group, "atomic_weight_ratio", A_);
}

double DiscretePhoton::sample(double E, uint64_t* seed) const
{
  if (primary_flag_ == 2) {
    return energy_ + A_ / (A_ + 1) * E;
  } else {
    return energy_;
  }
}

//==============================================================================
// LevelInelastic implementation
//==============================================================================

LevelInelastic::LevelInelastic(hid_t group)
{
  read_attribute(group, "threshold", threshold_);
  read_attribute(group, "mass_ratio", mass_ratio_);
}

double LevelInelastic::sample(double E, uint64_t* seed) const
{
  return mass_ratio_ * (E - threshold_);
}

//==============================================================================
// ContinuousTabular implementation
//==============================================================================

ContinuousTabular::ContinuousTabular(hid_t group)
{
  // Open incoming energy dataset
  hid_t dset = open_dataset(group, "energy");

  // Get interpolation parameters
  tensor::Tensor<int> temp;
  read_attribute(dset, "interpolation", temp);

  tensor::View<int> temp_b = temp.slice(0); // breakpoints
  tensor::View<int> temp_i = temp.slice(1); // interpolation parameters

  std::copy(temp_b.begin(), temp_b.end(), std::back_inserter(breakpoints_));
  for (const auto i : temp_i)
    interpolation_.push_back(int2interp(i));
  n_region_ = breakpoints_.size();

  // Get incoming energies
  read_dataset(dset, energy_);
  std::size_t n_energy = energy_.size();
  close_dataset(dset);

  // Get outgoing energy distribution data
  dset = open_dataset(group, "distribution");
  vector<int> offsets;
  vector<int> interp;
  vector<int> n_discrete;
  read_attribute(dset, "offsets", offsets);
  read_attribute(dset, "interpolation", interp);
  read_attribute(dset, "n_discrete_lines", n_discrete);

  tensor::Tensor<double> eout;
  read_dataset(dset, eout);
  close_dataset(dset);

  for (int i = 0; i < n_energy; ++i) {
    // Determine number of outgoing energies
    int j = offsets[i];
    int n;
    if (i < n_energy - 1) {
      n = offsets[i + 1] - j;
    } else {
      n = eout.shape(1) - j;
    }

    // Assign interpolation scheme and number of discrete lines
    CTTable d;
    d.interpolation = int2interp(interp[i]);
    d.n_discrete = n_discrete[i];

    // Copy data
    d.e_out = eout.slice(0, tensor::range(j, j + n));
    d.p = eout.slice(1, tensor::range(j, j + n));

    // To get answers that match ACE data, for now we still use the tabulated
    // CDF values that were passed through to the HDF5 library. At a later
    // time, we can remove the CDF values from the HDF5 library and
    // reconstruct them using the PDF
    if (true) {
      d.c = eout.slice(2, tensor::range(j, j + n));
    } else {
      // Calculate cumulative distribution function -- discrete portion
      for (int k = 0; k < d.n_discrete; ++k) {
        if (k == 0) {
          d.c[k] = d.p[k];
        } else {
          d.c[k] = d.c[k - 1] + d.p[k];
        }
      }

      // Continuous portion
      for (int k = d.n_discrete; k < n; ++k) {
        if (k == d.n_discrete) {
          d.c[k] = d.c[k - 1] + d.p[k];
        } else {
          if (d.interpolation == Interpolation::histogram) {
            d.c[k] = d.c[k - 1] + d.p[k - 1] * (d.e_out[k] - d.e_out[k - 1]);
          } else if (d.interpolation == Interpolation::lin_lin) {
            d.c[k] = d.c[k - 1] + 0.5 * (d.p[k - 1] + d.p[k]) *
                                    (d.e_out[k] - d.e_out[k - 1]);
          }
        }
      }

      // Normalize density and distribution functions
      d.p /= d.c[n - 1];
      d.c /= d.c[n - 1];
    }

    distribution_.push_back(std::move(d));
  } // incoming energies
}

double ContinuousTabular::sample(double E, uint64_t* seed) const
{
  // Read number of interpolation regions and incoming energies
  bool histogram_interp;
  if (n_region_ == 1) {
    histogram_interp = (interpolation_[0] == Interpolation::histogram);
  } else {
    histogram_interp = false;
  }

  // Find energy bin and calculate interpolation factor -- if the energy is
  // outside the range of the tabulated energies, choose the first or last bins
  int i;
  double r;
  get_energy_index(energy_, E, i, r);

  // Sample between the ith and [i+1]th bin
  int l;
  if (histogram_interp) {
    l = i;
  } else {
    l = r > prn(seed) ? i + 1 : i;
  }

  // Determine outgoing energy bin
  int n_energy_out = distribution_[l].e_out.size();
  int n_discrete = distribution_[l].n_discrete;
  double r1 = prn(seed);
  const auto& cdf = distribution_[l].c;
  double c_k = cdf[0];
  int k = 0;
  int end = n_energy_out - 2;

  // Discrete portion
  if (n_discrete > 0) {
    // Select the first discrete line whose CDF value exceeds r1
    k =
      std::upper_bound(cdf.begin(), cdf.begin() + n_discrete, r1) - cdf.begin();
    if (k < n_discrete) {
      end = k;
    } else {
      k = n_discrete - 1;
    }
    c_k = cdf[k];
  }

  // Continuous portion
  if (n_discrete < end) {
    // Select the bin whose upper CDF value is the first to exceed r1
    k = std::upper_bound(
          cdf.begin() + n_discrete + 1, cdf.begin() + end + 1, r1) -
        cdf.begin() - 1;
    c_k = cdf[k];
  }

  double E_l_k = distribution_[l].e_out[k];

  if (k < n_discrete) {
    // Discrete case
    return E_l_k;
  } else {
    // Continuous case
    double p_l_k = distribution_[l].p[k];
    double E_out;
    if (distribution_[l].interpolation == Interpolation::histogram) {
      // Histogram interpolation
      if (p_l_k > 0.0) {
        E_out = E_l_k + (r1 - c_k) / p_l_k;
      } else {
        E_out = E_l_k;
      }
    } else if (distribution_[l].interpolation == Interpolation::lin_lin) {
      // Linear-linear interpolation
      double E_l_k1 = distribution_[l].e_out[k + 1];
      double p_l_k1 = distribution_[l].p[k + 1];

      if (E_l_k != E_l_k1) {
        double frac = (p_l_k1 - p_l_k) / (E_l_k1 - E_l_k);
        if (frac == 0.0) {
          E_out = E_l_k + (r1 - c_k) / p_l_k;
        } else {
          E_out =
            E_l_k +
            (std::sqrt(std::max(0.0, p_l_k * p_l_k + 2.0 * frac * (r1 - c_k))) -
              p_l_k) /
              frac;
        }
      } else {
        E_out = E_l_k;
      }
    } else {
      throw std::runtime_error {
        "Unexpected interpolation for continuous energy "
        "distribution."};
    }

    // Now interpolate between incident energy bins i and i + 1
    if (!histogram_interp && n_energy_out > 1) {
      // Interpolation for energy E1 and EK
      n_energy_out = distribution_[i].e_out.size();
      n_discrete = distribution_[i].n_discrete;
      const double E_i_1 = distribution_[i].e_out[n_discrete];
      const double E_i_K = distribution_[i].e_out[n_energy_out - 1];

      n_energy_out = distribution_[i + 1].e_out.size();
      n_discrete = distribution_[i + 1].n_discrete;
      const double E_i1_1 = distribution_[i + 1].e_out[n_discrete];
      const double E_i1_K = distribution_[i + 1].e_out[n_energy_out - 1];

      const double E_1 = E_i_1 + r * (E_i1_1 - E_i_1);
      const double E_K = E_i_K + r * (E_i1_K - E_i_K);

      if (l == i) {
        return E_1 + (E_out - E_i_1) * (E_K - E_1) / (E_i_K - E_i_1);
      } else {
        return E_1 + (E_out - E_i1_1) * (E_K - E_1) / (E_i1_K - E_i1_1);
      }
    } else {
      return E_out;
    }
  }
}

ContinuousTabular::Mix ContinuousTabular::mix(double E) const
{
  // The same table choice as sample()
  bool histogram_interp =
    (n_region_ == 1) && (interpolation_[0] == Interpolation::histogram);
  int i;
  double r;
  get_energy_index(energy_, E, i, r);

  Mix m;
  if (histogram_interp || energy_.size() < 2) {
    m.n = 1;
    m.l[0] = i;
    m.w[0] = 1.0;
    m.a[0] = 0.0;
    m.b[0] = 1.0;
    return m;
  }

  // sample() picks table i + 1 with probability r, then maps the sampled
  // energy linearly from that table's continuous range [E_l_1, E_l_K] onto
  // the interpolated range [E_1, E_K]. A table with a single outgoing energy
  // is not rescaled.
  m.n = 2;
  m.l[0] = i;
  m.l[1] = i + 1;
  m.w[0] = 1.0 - r;
  m.w[1] = r;
  const auto& di = distribution_[i];
  const auto& dj = distribution_[i + 1];
  const double E_i_1 = di.e_out[di.n_discrete];
  const double E_i_K = di.e_out[di.e_out.size() - 1];
  const double E_j_1 = dj.e_out[dj.n_discrete];
  const double E_j_K = dj.e_out[dj.e_out.size() - 1];
  const double E_1 = E_i_1 + r * (E_j_1 - E_i_1);
  const double E_K = E_i_K + r * (E_j_K - E_i_K);
  const double ends[2][2] = {{E_i_1, E_i_K}, {E_j_1, E_j_K}};
  for (int t = 0; t < 2; ++t) {
    if (distribution_[m.l[t]].e_out.size() > 1) {
      // Inverse of E_out = E_1 + (E_tab - E_t1) (E_K - E_1) / (E_tK - E_t1)
      m.b[t] = (ends[t][1] - ends[t][0]) / (E_K - E_1);
      m.a[t] = ends[t][0] - m.b[t] * E_1;
    } else {
      m.a[t] = 0.0;
      m.b[t] = 1.0;
    }
  }
  return m;
}

double ContinuousTabular::table_pdf(int l, double x) const
{
  const auto& d = distribution_[l];
  if (d.n_discrete > 0) {
    fatal_error("ContinuousTabular::evaluate does not support discrete lines.");
  }
  const int n = d.e_out.size();
  if (n < 2 || x < d.e_out[0] || x >= d.e_out[n - 1])
    return 0.0;
  // Bin k with e_out[k] <= x < e_out[k + 1]. sample() inverts the tabulated
  // CDF within the bin, so the density is p_k (histogram) or the linear
  // interpolation of p (lin-lin). The tabulated CDF agrees with the integral
  // of p to round-off in OpenMC's data, so the bins are taken as tabulated.
  int k =
    std::upper_bound(d.e_out.begin(), d.e_out.end(), x) - d.e_out.begin() - 1;
  double p_k = d.p[k];
  if (d.interpolation == Interpolation::histogram)
    return p_k;
  double E_k = d.e_out[k];
  double E_k1 = d.e_out[k + 1];
  return p_k + (d.p[k + 1] - p_k) * (x - E_k) / (E_k1 - E_k);
}

double ContinuousTabular::table_cdf(int l, double x) const
{
  const auto& d = distribution_[l];
  if (d.n_discrete > 0) {
    fatal_error("ContinuousTabular::cdf does not support discrete lines.");
  }
  const int n = d.e_out.size();
  if (n < 2 || x <= d.e_out[0])
    return 0.0;
  if (x >= d.e_out[n - 1])
    return d.c[n - 1];
  int k =
    std::upper_bound(d.e_out.begin(), d.e_out.end(), x) - d.e_out.begin() - 1;
  double dx = x - d.e_out[k];
  double p_k = d.p[k];
  if (d.interpolation == Interpolation::histogram)
    return d.c[k] + p_k * dx;
  double frac = (d.p[k + 1] - p_k) / (d.e_out[k + 1] - d.e_out[k]);
  return d.c[k] + p_k * dx + 0.5 * frac * dx * dx;
}

double ContinuousTabular::evaluate(double E, double E_out) const
{
  Mix m = mix(E);
  double f = 0.0;
  for (int t = 0; t < m.n; ++t) {
    if (m.w[t] > 0.0)
      f += m.w[t] * table_pdf(m.l[t], m.a[t] + m.b[t] * E_out) * m.b[t];
  }
  return f;
}

double ContinuousTabular::cdf(double E, double E_out) const
{
  Mix m = mix(E);
  double F = 0.0;
  for (int t = 0; t < m.n; ++t) {
    if (m.w[t] > 0.0)
      F += m.w[t] * table_cdf(m.l[t], m.a[t] + m.b[t] * E_out);
  }
  return F;
}

double EnergyDistribution::evaluate(double E, double E_out) const
{
  fatal_error("evaluate() is not implemented for this energy distribution.");
}

double EnergyDistribution::cdf(double E, double E_out) const
{
  fatal_error("cdf() is not implemented for this energy distribution.");
}

//==============================================================================
// MaxwellEnergy implementation
//==============================================================================

MaxwellEnergy::MaxwellEnergy(hid_t group)
{
  read_attribute(group, "u", u_);
  hid_t dset = open_dataset(group, "theta");
  theta_ = Tabulated1D {dset};
  close_dataset(dset);
}

double MaxwellEnergy::sample(double E, uint64_t* seed) const
{
  // Get temperature corresponding to incoming energy
  double theta = theta_(E);

  while (true) {
    // Sample maxwell fission spectrum
    double E_out = maxwell_spectrum(theta, seed);

    // Accept energy based on restriction energy
    if (E_out <= E - u_)
      return E_out;
  }
}

//==============================================================================
// Evaporation implementation
//==============================================================================

Evaporation::Evaporation(hid_t group)
{
  read_attribute(group, "u", u_);
  hid_t dset = open_dataset(group, "theta");
  theta_ = Tabulated1D {dset};
  close_dataset(dset);
}

double Evaporation::sample(double E, uint64_t* seed) const
{
  // Get temperature corresponding to incoming energy
  double theta = theta_(E);

  double y = (E - u_) / theta;
  double v = 1.0 - std::exp(-y);

  // Sample outgoing energy based on evaporation spectrum probability
  // density function
  double x;
  while (true) {
    x = -std::log((1.0 - v * prn(seed)) * (1.0 - v * prn(seed)));
    if (x <= y)
      break;
  }

  return x * theta;
}

// Lower incomplete gamma functions gamma(3/2, y) and gamma(2, y), the
// normalizations of the Maxwell and evaporation spectra on [0, y theta]
namespace {
double lower_gamma_3_2(double y)
{
  double s = std::sqrt(y);
  return 0.5 * SQRT_PI * std::erf(s) - s * std::exp(-y);
}
double lower_gamma_2(double y)
{
  // 1 - e^-y (1 + y), with a series for small y to avoid cancellation
  if (y < 1e-3)
    return y * y * (0.5 - y / 3.0 + y * y / 8.0);
  return -std::expm1(-y) - y * std::exp(-y);
}
} // namespace

double MaxwellEnergy::evaluate(double E, double E_out) const
{
  double theta = theta_(E);
  double y = (E - u_) / theta;
  if (E_out <= 0.0 || E_out > E - u_ || y <= 0.0)
    return 0.0;
  double x = E_out / theta;
  return std::sqrt(x) * std::exp(-x) / (theta * lower_gamma_3_2(y));
}

double MaxwellEnergy::cdf(double E, double E_out) const
{
  double theta = theta_(E);
  double y = (E - u_) / theta;
  if (E_out <= 0.0 || y <= 0.0)
    return 0.0;
  if (E_out >= E - u_)
    return 1.0;
  return lower_gamma_3_2(E_out / theta) / lower_gamma_3_2(y);
}

double Evaporation::evaluate(double E, double E_out) const
{
  double theta = theta_(E);
  double y = (E - u_) / theta;
  if (E_out <= 0.0 || E_out > E - u_ || y <= 0.0)
    return 0.0;
  double x = E_out / theta;
  return x * std::exp(-x) / (theta * lower_gamma_2(y));
}

double Evaporation::cdf(double E, double E_out) const
{
  double theta = theta_(E);
  double y = (E - u_) / theta;
  if (E_out <= 0.0 || y <= 0.0)
    return 0.0;
  if (E_out >= E - u_)
    return 1.0;
  return lower_gamma_2(E_out / theta) / lower_gamma_2(y);
}

//==============================================================================
// WattEnergy implementation
//==============================================================================

WattEnergy::WattEnergy(hid_t group)
{
  // Read restriction energy
  read_attribute(group, "u", u_);

  // Read tabulated functions
  hid_t dset = open_dataset(group, "a");
  a_ = Tabulated1D {dset};
  close_dataset(dset);
  dset = open_dataset(group, "b");
  b_ = Tabulated1D {dset};
  close_dataset(dset);
}

double WattEnergy::sample(double E, uint64_t* seed) const
{
  // Determine Watt parameters at incident energy
  double a = a_(E);
  double b = b_(E);

  while (true) {
    // Sample energy-dependent Watt fission spectrum
    double E_out = watt_spectrum(a, b, seed);

    // Accept energy based on restriction energy
    if (E_out <= E - u_)
      return E_out;
  }
}

} // namespace openmc
