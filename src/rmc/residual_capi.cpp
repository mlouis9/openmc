//! \file residual_capi.cpp
//! \brief C API for residual samples (Stage 4).

#include <algorithm>

#include "openmc/capi.h"
#include "openmc/error.h"
#include "openmc/material.h"
#include "openmc/rmc/capi.h"
#include "openmc/rmc/residual.h"
#include "openmc/rmc/xs_probe.h"

using namespace openmc;

extern "C" int openmc_rmc_xs(int32_t material_id, int64_t n, const double* E,
  double sqrtkT, double density_mult, double* total, double* elastic,
  double* nu_fission)
{
  auto it = model::material_map.find(material_id);
  if (it == model::material_map.end()) {
    set_errmsg("No material with that ID.");
    return OPENMC_E_INVALID_ID;
  }
  // Uses the same grid set-up as openmc_rmc_xs_total
  double dummy;
  int err = openmc_rmc_xs_total(material_id, 1, E, sqrtkT, density_mult, &dummy);
  if (err)
    return err;
  const Material& mat = *model::materials[it->second];
  rmc::XsProbe probe;
  for (int64_t i = 0; i < n; ++i) {
    total[i] = probe.total(it->second, E[i], sqrtkT, density_mult);
    double el = 0.0;
    for (int j = 0; j < static_cast<int>(mat.nuclide_.size()); ++j)
      el += mat.atom_density(j, density_mult) *
            probe.micro_elastic(mat.nuclide_[j]);
    elastic[i] = el;
    nu_fission[i] = probe.particle().macro_xs().nu_fission;
  }
  return 0;
}

extern "C" int openmc_rmc_residual_sample(int tag, int64_t n, uint64_t seed,
  int n_directions, int tail, double cutoff, double length, double E_min,
  double E_max, double Ein_min, double Ein_max, double* r, double* u,
  double* E, double* density, double* C_s, double* C_f, double* U)
{
  if (!rmc::loaded_field) {
    set_errmsg("No RMC field is loaded; call openmc_rmc_field_load first.");
    return OPENMC_E_ALLOCATE;
  }
  if (tag < 0 || tag > 1 || tail < 0 || tail > 2 || n_directions < 1) {
    set_errmsg("Invalid tag, tail, or number of directions.");
    return OPENMC_E_INVALID_ARGUMENT;
  }
  // Cross-section lookups need the energy grid (see backward_capi.cpp)
  double dummy;
  int32_t any_id = model::materials[0]->id();
  if (int err = openmc_rmc_xs_total(any_id, 1, &E_min, 0.16, 1.0, &dummy))
    return err;

  rmc::ResidualOptions opt;
  opt.n_directions = n_directions;
  opt.tail.mode = static_cast<rmc::Tail>(tail);
  opt.tail.cutoff = cutoff;
  opt.tail.length = length;
  opt.E_min = E_min;
  opt.E_max = E_max;
  opt.Ein_min = Ein_min;
  opt.Ein_max = Ein_max;
  rmc::ResidualSampler sampler(*rmc::loaded_field, opt);
  rmc::ResidualRecords rec;
  sampler.sample(static_cast<rmc::Tag>(tag), n, seed, rec);
  std::copy(rec.r.begin(), rec.r.end(), r);
  std::copy(rec.u.begin(), rec.u.end(), u);
  std::copy(rec.E.begin(), rec.E.end(), E);
  std::copy(rec.density.begin(), rec.density.end(), density);
  std::copy(rec.C_s.begin(), rec.C_s.end(), C_s);
  std::copy(rec.C_f.begin(), rec.C_f.end(), C_f);
  std::copy(rec.U.begin(), rec.U.end(), U);
  return 0;
}

extern "C" int openmc_rmc_residual_weights(int64_t n, const int* tag,
  const double* C_s, const double* C_f, const double* U, const double* density,
  int64_t N, double a_s, double a_f, double b_s, double b_f, double k,
  double* w)
{
  rmc::ResidualRecords rec;
  rec.tag.assign(tag, tag + n);
  rec.C_s.assign(C_s, C_s + n);
  rec.C_f.assign(C_f, C_f + n);
  rec.U.assign(U, U + n);
  rec.density.assign(density, density + n);
  rec.E.assign(n, 0.0);
  vector<double> out;
  rmc::weight_records(rec, N, a_s, a_f, b_s, b_f, k, out);
  std::copy(out.begin(), out.end(), w);
  return 0;
}
