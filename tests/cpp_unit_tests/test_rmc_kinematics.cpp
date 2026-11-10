#include <cmath>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "openmc/rmc/kinematics.h"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using namespace openmc::rmc;

namespace {
const double MASSES[] = {1.0001, 2.0, 11.9, 15.86, 236.0};

// Forward relation: lab cosine from centre-of-mass cosine.
double mu_lab_from_cm(double A, double mc)
{
  return (1.0 + A * mc) / std::sqrt(A * A + 2.0 * A * mc + 1.0);
}
} // namespace

TEST_CASE("mu_cm_from_lab inverts the forward relation")
{
  for (double A : MASSES) {
    for (int i = 0; i <= 200; ++i) {
      double mc = -1.0 + 2.0 * i / 200.0;
      double mu0 = mu_lab_from_cm(A, mc);
      REQUIRE_THAT(mu_cm_from_lab(A, mu0), WithinAbs(mc, 1e-12));
    }
  }
}

TEST_CASE("Jacobian is the reciprocal of the forward derivative")
{
  for (double A : MASSES) {
    for (int i = 1; i < 200; ++i) {
      double mc = -1.0 + 2.0 * i / 200.0;
      double mu0 = mu_lab_from_cm(A, mc);
      double dmu0_dmc =
        A * A * (A + mc) / std::pow(A * A + 2.0 * A * mc + 1.0, 1.5);
      REQUIRE_THAT(dmu_cm_dmu_lab(A, mu0), WithinRel(1.0 / dmu0_dmc, 1e-10));
    }
  }
}

TEST_CASE("gamma_A endpoints")
{
  for (double A : MASSES) {
    double alpha = std::pow((A - 1.0) / (A + 1.0), 2);
    REQUIRE_THAT(gamma_A(A, 1.0), WithinAbs(1.0, 1e-14));
    REQUIRE_THAT(gamma_A(A, -1.0), WithinAbs(alpha, 1e-14));
  }
}

TEST_CASE("Isotropic CM scattering gives a flat lab energy distribution")
{
  // p_cm = 1/2 on [-1,1]. Push it through mu0 -> E/E' = gamma_A(mu0) with the
  // Jacobian, on a fine mu0 grid, and histogram E/E' on [alpha, 1].
  for (double A : {2.0, 11.9, 236.0}) {
    const double alpha = std::pow((A - 1.0) / (A + 1.0), 2);
    const int n_bins = 20, n_mu = 2000000;
    double hist[20] = {};
    double dmu = 2.0 / n_mu;
    for (int i = 0; i < n_mu; ++i) {
      double mu0 = -1.0 + (i + 0.5) * dmu;
      // Lab cosines below the minimum reachable value carry zero density;
      // the Jacobian handles this smoothly for A > 1.
      double p_lab = 0.5 * dmu_cm_dmu_lab(A, mu0);
      double x = (gamma_A(A, mu0) - alpha) / (1.0 - alpha);
      int b = std::min(n_bins - 1, std::max(0, static_cast<int>(x * n_bins)));
      hist[b] += p_lab * dmu;
    }
    for (double h : hist)
      REQUIRE_THAT(h, WithinAbs(1.0 / n_bins, 1e-4));
  }
}
