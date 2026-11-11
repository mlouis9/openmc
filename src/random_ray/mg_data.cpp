#include "openmc/random_ray/mg_data.h"

#include "openmc/mgxs_interface.h"
#include "openmc/search.h"

#include <algorithm>
#include <cmath>

namespace openmc {

namespace data {
RandomRayMGData random_ray_mg;
} // namespace data

void RandomRayMGData::clear()
{
  *this = RandomRayMGData {};
}

void RandomRayMGData::fill_from_mgxs()
{
  clear();

  // Temperature and angle indices, if using multiple temperature data sets
  // and/or anisotropic data sets.
  // TODO: Currently assumes we are only using single angle data.
  const int a = 0;

  n_groups = data::mg.num_energy_groups_;
  rev_energy_bins = data::mg.rev_energy_bins_;
  energy_bin_avg = data::mg.energy_bin_avg_;

  n_materials = data::mg.macro_xs_.size();
  n_temperatures = 1;
  for (int i = 0; i < n_materials; i++) {
    n_temperatures =
      std::max(n_temperatures, data::mg.macro_xs_[i].n_temperature_points());
  }

  for (int i = 0; i < n_materials; i++) {
    auto& m = data::mg.macro_xs_[i];
    exists_in_model.push_back(m.exists_in_model);
    is_isotropic.push_back(m.is_isotropic);
    kTs.emplace_back();
    for (int t = 0; t < m.n_temperature_points(); t++) {
      kTs.back().push_back(m.kT(t));
    }

    for (int t = 0; t < n_temperatures; t++) {
      for (int g_out = 0; g_out < n_groups; g_out++) {
        if (m.exists_in_model && t < m.n_temperature_points()) {
          sigma_t.push_back(
            m.get_xs(MgxsType::TOTAL, g_out, NULL, NULL, NULL, t, a));
          nu_sigma_f.push_back(
            m.get_xs(MgxsType::NU_FISSION, g_out, NULL, NULL, NULL, t, a));
          sigma_f.push_back(
            m.get_xs(MgxsType::FISSION, g_out, NULL, NULL, NULL, t, a));

          double c =
            m.get_xs(MgxsType::CHI_PROMPT, g_out, &g_out, NULL, NULL, t, a);
          if (!std::isfinite(c)) {
            // MGXS interface may return NaN in some cases, such as when
            // material is fissionable but has very small sigma_f.
            c = 0.0;
          }
          chi.push_back(c);

          kappa_fission.push_back(
            m.get_xs(MgxsType::KAPPA_FISSION, g_out, NULL, NULL, NULL, t, a));

          for (int g_in = 0; g_in < n_groups; g_in++) {
            sigma_s.push_back(
              m.get_xs(MgxsType::NU_SCATTER, g_in, &g_out, NULL, NULL, t, a));
          }
        } else {
          sigma_t.push_back(0);
          nu_sigma_f.push_back(0);
          sigma_f.push_back(0);
          chi.push_back(0);
          kappa_fission.push_back(0);
          for (int g_in = 0; g_in < n_groups; g_in++) {
            sigma_s.push_back(0);
          }
        }
      }
    }
  }
}

int RandomRayMGData::temperature_index(int m, double sqrtkT) const
{
  // Nearest temperature point; the first one on a tie, as in
  // Mgxs::get_temperature_index()
  const auto& kt = kTs[m];
  double kT = sqrtkT * sqrtkT;
  int best = 0;
  for (int t = 1; t < static_cast<int>(kt.size()); t++) {
    if (std::abs(kt[t] - kT) < std::abs(kt[best] - kT)) {
      best = t;
    }
  }
  return best;
}

int RandomRayMGData::group_index(double E) const
{
  int g = lower_bound_index(rev_energy_bins.begin(), rev_energy_bins.end(), E);
  return n_groups - g - 1;
}

} // namespace openmc
