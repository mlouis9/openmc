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

namespace {

//! The records of the last openmc_rmc_residual_sample call, and the
//! per-direction diagnostics if requested
rmc::ResidualRecords last_records;
vector<double> last_ce, last_mg;

} // namespace

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
  int err =
    openmc_rmc_xs_total(material_id, 1, E, sqrtkT, density_mult, &dummy);
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

extern "C" int openmc_rmc_residual_sample(const openmc_rmc_residual_config* cfg,
  int tag, int64_t n, uint64_t seed, int keep_directions, int64_t* n_records)
{
  if (!rmc::loaded_field) {
    set_errmsg("No RMC field is loaded; call openmc_rmc_field_load first.");
    return OPENMC_E_ALLOCATE;
  }
  if (tag < 0 || tag > 1 || cfg->tail < 0 || cfg->tail > 2) {
    set_errmsg("Invalid tag or tail.");
    return OPENMC_E_INVALID_ARGUMENT;
  }
  auto dir_mode = static_cast<rmc::DirectionMode>(cfg->directions);
  if (dir_mode == rmc::DirectionMode::sweep && !rmc::loaded_sweep) {
    set_errmsg("Sweep directions need a field sweep (openmc_rmc_sweep_field).");
    return OPENMC_E_ALLOCATE;
  }
  // Cross-section lookups need the energy grid (see backward_capi.cpp)
  double dummy, E1 = 1.0;
  int32_t any_id = model::materials[0]->id();
  if (int err = openmc_rmc_xs_total(any_id, 1, &E1, 0.16, 1.0, &dummy))
    return err;

  rmc::ResidualOptions opt;
  opt.directions = dir_mode;
  opt.n_directions = cfg->n_directions;
  opt.control = cfg->control != 0;
  opt.n_control_directions = cfg->n_control_directions;
  opt.control_energy = static_cast<rmc::ControlEnergy>(cfg->control_energy);
  opt.sampler_route = cfg->sampler_route != 0;
  opt.tail.mode = static_cast<rmc::Tail>(cfg->tail);
  opt.tail.cutoff = cfg->cutoff;
  opt.tail.length = cfg->length;
  opt.tail.max_segments = cfg->max_segments;
  opt.voxel.mode = static_cast<rmc::VoxelMode>(cfg->voxel_mode);
  if (cfg->n_voxel_weights > 0)
    opt.voxel.weights.assign(
      cfg->voxel_weights, cfg->voxel_weights + cfg->n_voxel_weights);
  opt.voxel.weights_per_volume = cfg->weights_per_volume != 0;
  opt.outgoing.beta = cfg->beta;
  opt.outgoing.E_min = cfg->E_min;
  opt.outgoing.E_max = cfg->E_max;
  opt.incident.mode = static_cast<rmc::IncidentMode>(cfg->incident_mode);
  opt.incident.beta = cfg->beta_in;
  opt.incident.E_min = cfg->Ein_min;
  opt.incident.E_max = cfg->Ein_max;
  opt.sampler.all_sampler = cfg->all_sampler != 0;
  opt.sampler.energies = cfg->n_energies;
  opt.sampler.common_numbers = cfg->common_numbers != 0;
  opt.sampler.control = static_cast<rmc::SamplerControl>(cfg->sampler_control);
  opt.sampler.region_samples = cfg->region_samples;
  opt.sampler.net = cfg->net != 0;

  bool need_tables = opt.control ||
                     opt.incident.mode == rmc::IncidentMode::flux ||
                     dir_mode == rmc::DirectionMode::sweep;
  const rmc::MGTables* tables =
    need_tables ? &rmc::loaded_mg_tables() : nullptr;
  rmc::ResidualSampler sampler(
    *rmc::loaded_field, opt, tables, rmc::loaded_sweep.get());
  sampler.keep_directions(keep_directions != 0);
  last_records.clear();
  sampler.sample(static_cast<rmc::Tag>(tag), n, seed, last_records);
  last_ce = sampler.kept_ce();
  last_mg = sampler.kept_mg();
  *n_records = last_records.size();
  return 0;
}

extern "C" int openmc_rmc_residual_records(double* r, double* u, double* E,
  double* density, int* tag, int* sampler, int* kind, int* channel,
  double* terms)
{
  const auto& rec = last_records;
  const int64_t n = rec.size();
  std::copy(rec.r.begin(), rec.r.end(), r);
  std::copy(rec.u.begin(), rec.u.end(), u);
  std::copy(rec.E.begin(), rec.E.end(), E);
  std::copy(rec.density.begin(), rec.density.end(), density);
  std::copy(rec.tag.begin(), rec.tag.end(), tag);
  std::copy(rec.sampler.begin(), rec.sampler.end(), sampler);
  std::copy(rec.kind.begin(), rec.kind.end(), kind);
  std::copy(rec.channel.begin(), rec.channel.end(), channel);
  // terms [8][n]: C_s, C_f, M_s, M_f, R_s, R_f, U, Z
  const vector<double>* cols[8] = {
    &rec.C_s, &rec.C_f, &rec.M_s, &rec.M_f, &rec.R_s, &rec.R_f, &rec.U, &rec.Z};
  for (int c = 0; c < 8; ++c)
    std::copy(cols[c]->begin(), cols[c]->end(), terms + c * n);
  return 0;
}

extern "C" int openmc_rmc_residual_directions(
  int64_t* n, double* ce, double* mg)
{
  *n = static_cast<int64_t>(last_ce.size());
  if (ce)
    std::copy(last_ce.begin(), last_ce.end(), ce);
  if (mg)
    std::copy(last_mg.begin(), last_mg.end(), mg);
  return 0;
}

extern "C" int openmc_rmc_residual_weights(int64_t n, const int* tag,
  const int* sampler, const double* terms, const double* density, int64_t N,
  const double* scalars, double* w)
{
  rmc::ResidualRecords rec;
  rec.tag.assign(tag, tag + n);
  rec.sampler.assign(sampler, sampler + n);
  vector<double>* cols[8] = {
    &rec.C_s, &rec.C_f, &rec.M_s, &rec.M_f, &rec.R_s, &rec.R_f, &rec.U, &rec.Z};
  for (int c = 0; c < 8; ++c)
    cols[c]->assign(terms + c * n, terms + (c + 1) * n);
  rec.density.assign(density, density + n);
  rec.E.assign(n, 0.0);
  // scalars: a_s, a_f, b_s, b_f, k, c, lambda
  rmc::WeightScalars x;
  x.a[0] = scalars[0];
  x.a[1] = scalars[1];
  x.b[0] = scalars[2];
  x.b[1] = scalars[3];
  x.k = scalars[4];
  x.c = scalars[5];
  x.lambda = scalars[6];
  vector<double> out;
  rmc::weight_records(rec, N, x, out);
  std::copy(out.begin(), out.end(), w);
  return 0;
}

extern "C" int openmc_rmc_residual_records_aux(double* E_in, int* nuclide)
{
  std::copy(last_records.E_in.begin(), last_records.E_in.end(), E_in);
  std::copy(last_records.nuclide.begin(), last_records.nuclide.end(), nuclide);
  return 0;
}
