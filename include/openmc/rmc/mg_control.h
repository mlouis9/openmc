//! \file mg_control.h
//! \brief Multigroup tables in the CE process, and the multigroup production
//! at a common output that controls the CE production (Section 6.5 of the
//! implementation guide).
//!
//! Theory: docs/source/methods/rmc.rst, methods_rmc_mg_control.

#ifndef OPENMC_RMC_MG_CONTROL_H
#define OPENMC_RMC_MG_CONTROL_H

#include <cstdint>
#include <string>

#include "openmc/memory.h"
#include "openmc/random_ray/mg_data.h"
#include "openmc/rmc/field.h"
#include "openmc/vector.h"

namespace openmc {
namespace rmc {

//==============================================================================
//! The multigroup cross sections of the field file, with groups in the
//! field's ascending order (random ray's tables are descending). Materials
//! are indices into model::materials; MATERIAL_VOID has zero cross sections.
//==============================================================================

class MGTables {
public:
  explicit MGTables(const std::string& field_file);

  int n_groups() const { return d_.n_groups; }
  double sigma_t(int32_t m, double sqrtkT, int g) const;
  //! nu-scattering from g_in to g_out
  double nu_sigma_s(int32_t m, double sqrtkT, int g_in, int g_out) const;
  double nu_sigma_f(int32_t m, double sqrtkT, int g) const;
  double chi(int32_t m, double sqrtkT, int g) const;

private:
  int64_t base(int32_t m, double sqrtkT) const;
  int desc(int g) const { return d_.n_groups - 1 - g; }
  RandomRayMGData d_;
};

//==============================================================================
//! Multigroup production at a common output (E, Omega) at a destination in
//! region R with material m, from group-integrated angular fluxes psi^MG_g'
//! on one incident direction:
//!
//!   q^MG_t(E) = h_{t,R}(E) / (4 pi) * sum_g' K^t_m(g' -> g(E)) psi^MG_g',
//!
//! K^s = nu Sigma_s and K^f = chi nu Sigma_f (1/k removed). With the
//! same-energy form for fission, the sum over g' is replaced by its estimate
//! at one incident energy E' drawn with density p(E'):
//!
//!   K^f(g(E') -> g(E)) psi^MG_g(E') h_phi(E') / p(E'),
//!
//! which shares E' with the CE fission estimate on the same direction.
//==============================================================================

class MGControl {
public:
  MGControl(const FrozenField& field, const MGTables& tables)
    : field_(field), tables_(tables)
  {}

  //! Exact sum over incident groups
  double production(Tag t, int64_t region, int32_t m, double sqrtkT, double E,
    const vector<double>& psi_mg) const;

  //! Fission, estimated at the sampled incident energy E_in (density p_in)
  double fission_same_energy(int64_t region, int32_t m, double sqrtkT,
    double E, double E_in, double p_in, const vector<double>& psi_mg) const;

  const MGTables& tables() const { return tables_; }

private:
  const FrozenField& field_;
  const MGTables& tables_;
};

//! The tables of the field loaded through the C API, read on first use and
//! released with it (src/rmc/capi.cpp)
extern unique_ptr<MGTables> loaded_mg;
const MGTables& loaded_mg_tables();

} // namespace rmc
} // namespace openmc

#endif // OPENMC_RMC_MG_CONTROL_H
