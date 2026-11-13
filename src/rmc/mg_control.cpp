#include "openmc/rmc/mg_control.h"

#include "openmc/constants.h"
#include "openmc/rmc/mg_tables.h"

namespace openmc {
namespace rmc {

MGTables::MGTables(const std::string& field_file)
{
  load_mg_tables(field_file, d_);
}

int64_t MGTables::base(int32_t m, double sqrtkT) const
{
  int t = d_.temperature_index(m, sqrtkT);
  return (static_cast<int64_t>(m) * d_.n_temperatures + t) * d_.n_groups;
}

double MGTables::sigma_t(int32_t m, double sqrtkT, int g) const
{
  if (m == MATERIAL_VOID)
    return 0.0;
  return d_.sigma_t[base(m, sqrtkT) + desc(g)];
}

double MGTables::nu_sigma_s(int32_t m, double sqrtkT, int g_in, int g_out) const
{
  if (m == MATERIAL_VOID)
    return 0.0;
  const int G = d_.n_groups;
  return d_.sigma_s[(base(m, sqrtkT) + desc(g_out)) * G + desc(g_in)];
}

double MGTables::nu_sigma_f(int32_t m, double sqrtkT, int g) const
{
  if (m == MATERIAL_VOID)
    return 0.0;
  return d_.nu_sigma_f[base(m, sqrtkT) + desc(g)];
}

double MGTables::chi(int32_t m, double sqrtkT, int g) const
{
  if (m == MATERIAL_VOID)
    return 0.0;
  return d_.chi[base(m, sqrtkT) + desc(g)];
}

double MGControl::production(Tag t, int64_t region, int32_t m, double sqrtkT,
  double E, const vector<double>& psi_mg) const
{
  int g = field_.group(E);
  if (g < 0 || m == MATERIAL_VOID)
    return 0.0;
  const int G = tables_.n_groups();
  double sum = 0.0;
  if (t == Tag::s) {
    for (int gp = 0; gp < G; ++gp)
      sum += tables_.nu_sigma_s(m, sqrtkT, gp, g) * psi_mg[gp];
    return field_.shape(FrozenField::Shape::s, region, E) / (4.0 * PI) * sum;
  }
  for (int gp = 0; gp < G; ++gp)
    sum += tables_.nu_sigma_f(m, sqrtkT, gp) * psi_mg[gp];
  return field_.shape(FrozenField::Shape::f, region, E) / (4.0 * PI) *
         tables_.chi(m, sqrtkT, g) * sum;
}

double MGControl::fission_same_energy(int64_t region, int32_t m,
  double sqrtkT, double E, double E_in, double p_in,
  const vector<double>& psi_mg) const
{
  int g = field_.group(E);
  int gp = field_.group(E_in);
  if (g < 0 || gp < 0 || m == MATERIAL_VOID || !(p_in > 0.0))
    return 0.0;
  double h_phi = field_.shape(FrozenField::Shape::phi, region, E_in);
  return field_.shape(FrozenField::Shape::f, region, E) / (4.0 * PI) *
         tables_.chi(m, sqrtkT, g) * tables_.nu_sigma_f(m, sqrtkT, gp) *
         psi_mg[gp] * h_phi / p_in;
}

} // namespace rmc
} // namespace openmc
