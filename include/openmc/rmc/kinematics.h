//! \file kinematics.h
//! \brief Elastic kinematics for a stationary target with A > 1.
//!
//! Theory: docs/source/methods/rmc.rst, section "Elastic Kinematics,
//! Stationary Target" (label methods_rmc_kinematics). Equation labels are
//! given per function.

#ifndef OPENMC_RMC_KINEMATICS_H
#define OPENMC_RMC_KINEMATICS_H

namespace openmc {
namespace rmc {

//! Ratio of outgoing to incident energy, E/E', eq. rmc-gamma.
//! \param A    Target-to-neutron mass ratio, A > 1
//! \param mu0  Laboratory scattering cosine in [-1, 1]
//! \return     E/E' in [((A-1)/(A+1))^2, 1]
double gamma_A(double A, double mu0);

//! Center-of-mass cosine for laboratory cosine mu0, eq. rmc-mu-cm.
//! \param A    Target-to-neutron mass ratio, A > 1
//! \param mu0  Laboratory scattering cosine in [-1, 1]
//! \return     Center-of-mass cosine in [-1, 1]
double mu_cm_from_lab(double A, double mu0);

//! Jacobian d(mu_cm)/d(mu0), eq. rmc-dmu. Positive for A > 1.
//! \param A    Target-to-neutron mass ratio, A > 1
//! \param mu0  Laboratory scattering cosine in [-1, 1]
//! \return     d(mu_cm)/d(mu0)
double dmu_cm_dmu_lab(double A, double mu0);

} // namespace rmc
} // namespace openmc

#endif // OPENMC_RMC_KINEMATICS_H
