//! A compiled source that samples the frozen field of multilevel RMC,
//! a_s Q_s + a_f Q_f, with single-sample weights (docs/source/methods/rmc.rst,
//! methods_rmc_field_source). Used as the fixed source of reference runs.
//!
//! Parameters, comma-separated key=value: field, lifts, volumes (file paths);
//! a_s, a_f (amplitudes); lo0..lo2, hi0..hi2 (sampling box [cm]);
//! E_lo, E_hi (log-uniform energy range [eV]).

#include <cmath>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>

#include "openmc/constants.h"
#include "openmc/geometry.h"
#include "openmc/random_lcg.h"
#include "openmc/rmc/field.h"
#include "openmc/source.h"

using namespace openmc;

class FieldSampler : public Source {
public:
  explicit FieldSampler(std::unordered_map<std::string, std::string> p)
    : params_(std::move(p))
  {
    a_s_ = std::stod(params_.at("a_s"));
    a_f_ = std::stod(params_.at("a_f"));
    for (int i = 0; i < 3; ++i) {
      lo_[i] = std::stod(params_.at("lo" + std::to_string(i)));
      hi_[i] = std::stod(params_.at("hi" + std::to_string(i)));
    }
    E_lo_ = std::stod(params_.at("E_lo"));
    E_hi_ = std::stod(params_.at("E_hi"));
  }

  SourceSite sample(uint64_t* seed) const override;

private:
  //! The field needs the geometry, which does not exist yet when OpenMC
  //! constructs sources, so it is loaded on first use. call_once makes that
  //! safe when several threads sample at once.
  const rmc::FrozenField& field() const
  {
    std::call_once(once_, [this] {
      field_ = std::make_unique<rmc::FrozenField>(
        params_.at("field"), params_.at("lifts"), params_.at("volumes"));
    });
    return *field_;
  }

  std::unordered_map<std::string, std::string> params_;
  double a_s_, a_f_, lo_[3], hi_[3], E_lo_, E_hi_;
  mutable std::once_flag once_;
  mutable std::unique_ptr<rmc::FrozenField> field_;
};

SourceSite FieldSampler::sample(uint64_t* seed) const
{
  const auto& f = field();

  SourceSite site;
  site.particle = ParticleType::neutron();

  // Position: uniform in the box, density 1/V
  double V = 1.0;
  for (int i = 0; i < 3; ++i) {
    site.r[i] = lo_[i] + (hi_[i] - lo_[i]) * prn(seed);
    V *= hi_[i] - lo_[i];
  }

  // Direction: isotropic, density 1/(4 pi)
  double mu = 2.0 * prn(seed) - 1.0;
  double phi = 2.0 * PI * prn(seed);
  double st = std::sqrt(1.0 - mu * mu);
  site.u = {st * std::cos(phi), st * std::sin(phi), mu};

  // Energy: log-uniform on [E_lo, E_hi], density 1/(E ln(E_hi/E_lo))
  double L = std::log(E_hi_ / E_lo_);
  site.E = E_lo_ * std::exp(L * prn(seed));
  double p_E = 1.0 / (site.E * L);

  // Single-sample weight: field value over the joint sampling density
  GeometryState g;
  g.init_from_r_u(site.r, site.u);
  int64_t R = exhaustive_find_cell(g) ? f.locate(g) : -1;
  double Q = a_s_ * f.evaluate(R, rmc::Tag::s, site.E) +
             a_f_ * f.evaluate(R, rmc::Tag::f, site.E);
  site.wgt = 4.0 * PI * V * Q / p_E;
  return site;
}

extern "C" std::unique_ptr<FieldSampler> openmc_create_source(
  std::string parameters)
{
  std::unordered_map<std::string, std::string> p;
  std::stringstream ss(parameters);
  std::string item;
  while (std::getline(ss, item, ',')) {
    item.erase(0, item.find_first_not_of(' '));
    auto eq = item.find('=');
    p[item.substr(0, eq)] = item.substr(eq + 1);
  }
  return std::make_unique<FieldSampler>(std::move(p));
}
