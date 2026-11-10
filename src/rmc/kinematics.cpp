#include "openmc/rmc/kinematics.h"

#include <cmath>

namespace openmc {
namespace rmc {

namespace {
// Helpers in an unnamed namespace are visible only in this file,
// like a private fn in a Rust module.
double s_of(double A, double mu0)
{
  return std::sqrt(A * A - 1.0 + mu0 * mu0);
}
} // namespace

double gamma_A(double A, double mu0)
{
  // (mu0 + s)/(A + 1) = (A - 1)/(s - mu0): no cancellation as mu0 -> -1
  double r = (A - 1.0) / (s_of(A, mu0) - mu0);
  return r * r;
}

double mu_cm_from_lab(double A, double mu0)
{
  return (mu0 * s_of(A, mu0) - (1.0 - mu0 * mu0)) / A;
}

double dmu_cm_dmu_lab(double A, double mu0)
{
  // (2 mu0 + s + mu0^2/s)/A = (s + mu0)^2/(A s) = (A^2 - 1)^2 / (A s (s -
  // mu0)^2)
  double s = s_of(A, mu0);
  double d = s - mu0;
  return (A * A - 1.0) * (A * A - 1.0) / (A * s * d * d);
}

} // namespace rmc
} // namespace openmc