#include "openmc/rmc/production.h"

#include <cmath>

#include <fmt/format.h>

#include "openmc/angle_energy.h"
#include "openmc/distribution_energy.h"
#include "openmc/error.h"
#include "openmc/nuclide.h"
#include "openmc/particle_data.h"
#include "openmc/reaction.h"
#include "openmc/reaction_product.h"
#include "openmc/rmc/kinematics.h"
#include "openmc/secondary_uncorrelated.h"
#include "openmc/settings.h"

namespace openmc {
namespace rmc {

namespace {

//! Energy distribution of an uncorrelated angle-energy secondary
const EnergyDistribution& energy_law(const AngleEnergy& d)
{
  auto* u = dynamic_cast<const UncorrelatedAngleEnergy*>(&d);
  if (!u || !u->energy()) {
    fatal_error("RMC production: only uncorrelated angle-energy distributions "
                "with an energy law are supported for fission neutrons.");
  }
  return *u->energy();
}

//! Density (or CDF) of one product, as ReactionProduct::sample_dist() picks
//! among its distributions: distribution i with probability applicability_i
//! for all but the last, which takes the rest.
template<typename F>
double product_mix(const ReactionProduct& prod, double E_in, F&& law)
{
  const int n = prod.applicability_.size();
  if (n <= 1)
    return law(energy_law(*prod.distribution_.back()));
  double sum = 0.0, rest = 1.0;
  for (int i = 0; i < n - 1; ++i) {
    double w = prod.applicability_[i](E_in);
    sum += w * law(energy_law(*prod.distribution_[i]));
    rest -= w;
  }
  return sum + rest * law(energy_law(*prod.distribution_.back()));
}

//! Probability of each fission neutron product (index into rx.products_), as
//! sample_fission_neutron() chooses: prompt with 1 - nu_d / nu_t, delayed
//! group g (1..n_precursor) with yield_g / nu_t, the last group taking any
//! remainder of the delayed probability.
template<typename F>
double over_products(const Nuclide& nuc, const Reaction& rx, double E_in, F&& f)
{
  double nu_t = nuc.nu(E_in, Nuclide::EmissionMode::total);
  double nu_d = nuc.nu(E_in, Nuclide::EmissionMode::delayed);
  double beta = nu_d / nu_t;
  double result = (1.0 - beta) * f(rx.products_[0]);
  if (beta > 0.0) {
    double used = 0.0;
    for (int g = 1; g < nuc.n_precursor_; ++g) {
      double p = (*rx.products_[g].yield_)(E_in) / nu_d;
      result += beta * p * f(rx.products_[g]);
      used += p;
    }
    result += beta * (1.0 - used) * f(rx.products_[nuc.n_precursor_]);
  }
  return result;
}

} // namespace

double fission_energy_cdf(
  int i_nuclide, const Reaction& rx, double E_in, double E_out)
{
  const Nuclide& nuc = *data::nuclides[i_nuclide];
  const double E_max =
    data::energy_max[ParticleType::neutron().transport_index()];
  // Each product's sample is repeated until it falls below E_max, so its law
  // is conditioned on E_out < E_max
  auto conditioned = [&](const ReactionProduct& prod) {
    auto cdf_at = [&](double x) {
      return product_mix(prod, E_in,
        [&](const EnergyDistribution& d) { return d.cdf(E_in, x); });
    };
    double Z = cdf_at(E_max);
    return cdf_at(std::min(E_out, E_max)) / Z;
  };
  return over_products(nuc, rx, E_in, conditioned);
}

double fission_energy_density(
  int i_nuclide, const Reaction& rx, double E_in, double E_out)
{
  const Nuclide& nuc = *data::nuclides[i_nuclide];
  const double E_max =
    data::energy_max[ParticleType::neutron().transport_index()];
  if (E_out >= E_max)
    return 0.0;
  auto conditioned = [&](const ReactionProduct& prod) {
    double f = product_mix(prod, E_in,
      [&](const EnergyDistribution& d) { return d.evaluate(E_in, E_out); });
    double Z = product_mix(prod, E_in,
      [&](const EnergyDistribution& d) { return d.cdf(E_in, E_max); });
    return f / Z;
  };
  return over_products(nuc, rx, E_in, conditioned);
}

ElasticImage elastic_stationary(int i_nuclide, double kT, double E, double mu0)
{
  const Nuclide& nuc = *data::nuclides[i_nuclide];
  const double A = nuc.awr_;
  ElasticImage out {0.0, 0.0};
  if (A <= 1.0 || nuc.resonant_)
    return out;

  double gamma = gamma_A(A, mu0);
  out.E_in = E / gamma;
  // Where OpenMC samples a moving target, or has no data, the stationary
  // kernel does not apply
  const auto& grid = nuc.grid_[0].energy;
  if (out.E_in < settings::free_gas_threshold * kT || out.E_in < grid.front() ||
      out.E_in > grid.back())
    return out;

  // CM angular density at the CM cosine that maps to mu0, isotropic if the
  // data have no angular distribution (as elastic_scatter() does)
  const auto& d = nuc.reactions_[0]->products_[0].distribution_[0];
  auto* ua = dynamic_cast<const UncorrelatedAngleEnergy*>(d.get());
  if (!ua)
    fatal_error("RMC production: elastic scattering without an uncorrelated "
                "angle distribution.");
  double mu_cm = mu_cm_from_lab(A, mu0);
  double p_cm =
    ua->angle().empty() ? 0.5 : ua->angle().evaluate(out.E_in, mu_cm);
  out.density = p_cm * dmu_cm_dmu_lab(A, mu0) / gamma;
  return out;
}

} // namespace rmc
} // namespace openmc
