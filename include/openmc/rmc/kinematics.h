//! \file kinematics.h
//! Two-body elastic kinematics for a stationary target (A > 1).
//! Used by the common-output elastic evaluator (method doc §10).

#ifndef OPENMC_RMC_KINEMATICS_H
#define OPENMC_RMC_KINEMATICS_H

namespace openmc {
namespace rmc {

//! Ratio E_out / E_in for elastic scattering at laboratory cosine mu0.
//! \param A    target-to-neutron mass ratio, A > 1
//! \param mu0  laboratory scattering cosine in [-1, 1]
double gamma_A(double A, double mu0);

//! Centre-of-mass cosine corresponding to laboratory cosine mu0.
double mu_cm_from_lab(double A, double mu0);

//! Derivative d(mu_cm)/d(mu0). Positive for A > 1.
double dmu_cm_dmu_lab(double A, double mu0);

} // namespace rmc
} // namespace openmc

#endif // OPENMC_RMC_KINEMATICS_H