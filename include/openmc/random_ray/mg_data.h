#ifndef OPENMC_RANDOM_RAY_MG_DATA_H
#define OPENMC_RANDOM_RAY_MG_DATA_H

#include "openmc/vector.h"

namespace openmc {

//==============================================================================
//! The multigroup data the random ray solver reads.
//!
//! Random ray needs multigroup cross sections only as a few flat tables. This
//! class holds them, so that the solver does not read data::mg directly and
//! can run in a process that loaded continuous-energy data instead (as
//! multilevel RMC does). It is filled either from the MGXS library through
//! fill_from_mgxs(), which is what an ordinary random ray run does, or by
//! other code that writes the members directly (see rmc::load_mg_tables()).
//!
//! Groups follow the MGXS library convention: group 0 has the highest
//! energy. Tables are indexed by material (the index into model::materials),
//! then temperature, then group:
//!
//!   sigma_t[(m * n_temperatures + t) * n_groups + g]
//!   sigma_s[((m * n_temperatures + t) * n_groups + g_out) * n_groups + g_in]
//!
//! where sigma_s is the nu-scattering cross section from g_in to g_out.
//==============================================================================

class RandomRayMGData {
public:
  //! Fill every member from data::mg (the MGXS library of a multigroup run).
  void fill_from_mgxs();

  //! Empty every table and reset the counts.
  void clear();

  //! Index of the temperature point of material m nearest to sqrtkT (the
  //! square root of kT, in sqrt(eV)), as Mgxs::get_temperature_index().
  int temperature_index(int m, double sqrtkT) const;

  //! Group containing energy E [eV], as MgxsInterface::get_group_index().
  int group_index(double E) const;

  //! True once filled
  bool filled() const { return n_groups > 0; }

  int n_groups {0};
  int n_materials {0};
  int n_temperatures {0};

  //! Group edges [eV] in ascending energy (n_groups + 1 values)
  vector<double> rev_energy_bins;
  //! Midpoint energy [eV] of each group, by group index
  vector<double> energy_bin_avg;

  //! Per material: present in the model, isotropic data, and the kT values
  //! [eV] of its temperature points
  vector<char> exists_in_model;
  vector<char> is_isotropic;
  vector<vector<double>> kTs;

  //! Flat tables, layout as in the class description
  vector<double> sigma_t;
  vector<double> nu_sigma_f;
  vector<double> sigma_f;
  vector<double> chi;
  vector<double> kappa_fission;
  vector<double> sigma_s;
};

namespace data {
//! The multigroup data used by the random ray solver in this process
extern RandomRayMGData random_ray_mg;
} // namespace data

} // namespace openmc

#endif // OPENMC_RANDOM_RAY_MG_DATA_H
