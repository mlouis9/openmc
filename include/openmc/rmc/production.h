//! \file production.h
//! \brief Production densities at a given outgoing state, written as the
//! densities of what OpenMC's collision samplers do.
//!
//! Theory: docs/source/methods/rmc.rst, methods_rmc_production.

#ifndef OPENMC_RMC_PRODUCTION_H
#define OPENMC_RMC_PRODUCTION_H

namespace openmc {

class Reaction;

namespace rmc {

//==============================================================================
// Fission
//==============================================================================

//! Outgoing-energy density of one fission neutron of reaction rx of nuclide
//! i_nuclide at incident energy E_in, as sample_fission_neutron() samples it:
//! the prompt spectrum with probability 1 - nu_d / nu_t, delayed group g with
//! probability yield_g / nu_t, each product's distributions mixed by their
//! applicability, and each conditioned on E_out < data::energy_max.
//! \return Density in [1/eV]
double fission_energy_density(
  int i_nuclide, const Reaction& rx, double E_in, double E_out);

//! Cumulative distribution consistent with fission_energy_density().
double fission_energy_cdf(
  int i_nuclide, const Reaction& rx, double E_in, double E_out);

//==============================================================================
// Elastic scattering off a stationary target
//==============================================================================

//! The incident state that elastic scattering off a stationary target maps
//! to an outgoing energy E at laboratory cosine mu0, and the density there.
struct ElasticImage {
  double E_in;    //!< incident energy [eV], E / gamma_A(mu0)
  double density; //!< p_lab(mu0 | E_in) / gamma_A(mu0) [1/eV per unit mu0]
};

//! For an elastic collision rate s(E') per unit incident energy, the joint
//! density of the outgoing (E, mu0) is s(E_in) * density with
//! (E_in, density) = elastic_stationary(...) (eq. rmc-elastic-image). The
//! density is 0 where OpenMC would not use a stationary target: E_in below
//! free_gas_threshold * kT, a nuclide with A <= 1 or resonance scattering, or
//! E_in outside the nuclide's data.
ElasticImage elastic_stationary(int i_nuclide, double kT, double E, double mu0);

} // namespace rmc
} // namespace openmc

#endif // OPENMC_RMC_PRODUCTION_H
