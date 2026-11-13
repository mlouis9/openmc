//! \file random_ray_capi.cpp
//! \brief C API that runs OpenMC's random ray solver inside a
//! continuous-energy RMC process, either as a whole solve or one batch at a
//! time.

#include <algorithm>

#include <fmt/format.h>

#include "openmc/capi.h"
#include "openmc/error.h"
#include "openmc/memory.h"
#include "openmc/random_ray/flat_source_domain.h"
#include "openmc/random_ray/random_ray.h"
#include "openmc/random_ray/random_ray_simulation.h"
#include "openmc/rmc/capi.h"
#include "openmc/rmc/field.h"
#include "openmc/rmc/residual.h"
#include "openmc/settings.h"
#include "openmc/simulation.h"

using namespace openmc;

namespace {

//! Checks shared by every way of running random ray in a CE process
int check_ce_random_ray(const char* fn)
{
  if (!settings::run_CE) {
    set_errmsg(fmt::format("{} requires a continuous-energy model; use "
                           "openmc_run for a multigroup random ray run.",
      fn));
    return OPENMC_E_INVALID_ARGUMENT;
  }
  if (!settings::rmc_enabled || settings::rmc_field_file.empty()) {
    set_errmsg(fmt::format("{} requires settings.rmc['field_file'].", fn));
    return OPENMC_E_INVALID_ARGUMENT;
  }
  if (!RandomRay::ray_source_) {
    set_errmsg(
      fmt::format("{} requires a <random_ray> element in the settings.", fn));
    return OPENMC_E_INVALID_ARGUMENT;
  }
  if (simulation::initialized) {
    set_errmsg(fmt::format("{} cannot start while a simulation is "
                           "initialized; call openmc_simulation_finalize "
                           "first.",
      fn));
    return OPENMC_E_ALLOCATE;
  }
  return 0;
}

//! A random ray solve driven one batch at a time, and the settings it
//! changes, which openmc_rmc_sweep_finalize() restores
struct SweepState {
  unique_ptr<RandomRaySimulation> sim;
  SolverType solver;
  bool source_write;
};
unique_ptr<SweepState> sweep_state;

int no_sweep()
{
  set_errmsg("No random ray sweep is in progress; call "
             "openmc_rmc_sweep_init first.");
  return OPENMC_E_ALLOCATE;
}

} // namespace

extern "C" int openmc_rmc_run_random_ray()
{
  if (int err = check_ce_random_ray("openmc_rmc_run_random_ray"))
    return err;
  if (sweep_state) {
    set_errmsg("A random ray sweep is in progress.");
    return OPENMC_E_ALLOCATE;
  }

  // For the duration of the solve, random ray is the solver. OpenMC's batch,
  // eigenvalue, and tally code branches on the solver type: random ray has no
  // source or fission bank, its k needs no normalization by the number of
  // particles, and its tallies are not normalized per source particle.
  // Random ray also turns off source writing, which a later Monte Carlo run
  // in this process may want, so both settings are restored afterwards.
  SolverType solver = settings::solver_type;
  bool source_write = settings::source_write;
  settings::solver_type = SolverType::RANDOM_RAY;
  openmc_run_random_ray();
  settings::solver_type = solver;
  settings::source_write = source_write;
  return 0;
}

//==============================================================================
// Random ray one batch at a time
//==============================================================================

extern "C" int openmc_rmc_sweep_init()
{
  if (int err = check_ce_random_ray("openmc_rmc_sweep_init"))
    return err;
  if (sweep_state) {
    set_errmsg("A random ray sweep is already in progress.");
    return OPENMC_E_ALLOCATE;
  }
  if (FlatSourceDomain::adjoint_requested_) {
    set_errmsg("openmc_rmc_sweep_init does not support adjoint solves.");
    return OPENMC_E_INVALID_ARGUMENT;
  }

  // As in openmc_rmc_run_random_ray(): random ray is the solver until
  // openmc_rmc_sweep_finalize()
  sweep_state = make_unique<SweepState>();
  sweep_state->solver = settings::solver_type;
  sweep_state->source_write = settings::source_write;
  settings::solver_type = SolverType::RANDOM_RAY;

  // The same initialization as the forward eigenvalue solve of
  // openmc_run_random_ray()
  resolve_random_ray_volume_estimator();
  FlatSourceDomain::solve_ = RandomRaySolve::FORWARD;
  initialize_random_ray_solve();
  sweep_state->sim = make_unique<RandomRaySimulation>();
  sweep_state->sim->apply_fixed_sources_and_mesh_domains();
  return 0;
}

extern "C" int openmc_rmc_sweep_iterate(int n)
{
  if (!sweep_state)
    return no_sweep();
  for (int i = 0; i < n; ++i) {
    if (simulation::current_batch >= settings::n_batches) {
      set_errmsg(fmt::format("All {} batches have run; increase the number "
                             "of batches in the settings.",
        settings::n_batches));
      return OPENMC_E_OUT_OF_BOUNDS;
    }
    sweep_state->sim->power_iteration();
  }
  return 0;
}

extern "C" int openmc_rmc_sweep_set_recording(int64_t n, const int64_t* regions)
{
  if (!sweep_state)
    return no_sweep();
  vector<int64_t> r;
  if (n > 0)
    r.assign(regions, regions + n);
  sweep_state->sim->domain()->set_segment_recording(r, n < 0);
  return 0;
}

extern "C" int openmc_rmc_sweep_n_records(int64_t* n)
{
  if (!sweep_state)
    return no_sweep();
  int64_t total = 0;
  for (const auto& rec : sweep_state->sim->domain()->segment_records_)
    total += rec.size();
  *n = total;
  return 0;
}

extern "C" int openmc_rmc_sweep_records(
  int64_t* region, double* u, double* length, double* psi_bar)
{
  if (!sweep_state)
    return no_sweep();
  SegmentRecords rec = sweep_state->sim->domain()->gather_segment_records();
  std::copy(rec.region.begin(), rec.region.end(), region);
  std::copy(rec.u.begin(), rec.u.end(), u);
  std::copy(rec.length.begin(), rec.length.end(), length);
  std::copy(rec.psi_bar.begin(), rec.psi_bar.end(), psi_bar);
  return 0;
}

extern "C" int openmc_rmc_sweep_n_regions(int64_t* n, int* n_groups)
{
  if (!sweep_state)
    return no_sweep();
  *n = sweep_state->sim->domain()->n_source_regions();
  *n_groups = sweep_state->sim->domain()->n_groups();
  return 0;
}

extern "C" int openmc_rmc_sweep_regions(int32_t* cell_id, int32_t* instance,
  int32_t* mesh_id, int64_t* mesh_bin, double* volume, double* phi, double* k)
{
  if (!sweep_state)
    return no_sweep();
  const FlatSourceDomain& d = *sweep_state->sim->domain();
  vector<int32_t> c, i, m;
  vector<int64_t> b;
  d.source_region_keys(c, i, m, b);
  std::copy(c.begin(), c.end(), cell_id);
  std::copy(i.begin(), i.end(), instance);
  std::copy(m.begin(), m.end(), mesh_id);
  std::copy(b.begin(), b.end(), mesh_bin);
  const int G = d.n_groups();
  for (int64_t sr = 0; sr < d.n_source_regions(); ++sr) {
    volume[sr] = d.source_regions_.volume(sr) * d.simulation_volume();
    // After the flux swap, the old flux holds the latest iterate
    for (int g = 0; g < G; ++g)
      phi[sr * G + g] = d.source_regions_.scalar_flux_old(sr, g);
  }
  *k = d.k_eff_;
  return 0;
}

extern "C" int openmc_rmc_sweep_finalize()
{
  if (!sweep_state)
    return no_sweep();
  sweep_state->sim.reset();
  openmc_simulation_finalize();
  settings::solver_type = sweep_state->solver;
  settings::source_write = sweep_state->source_write;
  sweep_state.reset();
  return 0;
}

//==============================================================================
// One transport sweep from the frozen field (the tandem and sweep-control
// residual estimators)
//==============================================================================

extern "C" int openmc_rmc_sweep_field(int64_t batch, int64_t* n_records)
{
  if (!sweep_state)
    return no_sweep();
  if (!rmc::loaded_field) {
    set_errmsg("No RMC field is loaded; call openmc_rmc_field_load first.");
    return OPENMC_E_ALLOCATE;
  }
  const auto& field = *rmc::loaded_field;
  if (!field.has_volumes()) {
    set_errmsg("A field sweep needs region volumes (load the field with a "
               "volume file).");
    return OPENMC_E_INVALID_ARGUMENT;
  }
  FlatSourceDomain& d = *sweep_state->sim->domain();
  const int G = d.n_groups();
  if (G != field.n_groups()) {
    set_errmsg("The sweep and the field have different group structures.");
    return OPENMC_E_INVALID_ARGUMENT;
  }

  // Field region of every source region of the solve
  vector<int32_t> cell_id, instance, mesh_id;
  vector<int64_t> mesh_bin;
  d.source_region_keys(cell_id, instance, mesh_id, mesh_bin);
  const int64_t n_sr = d.n_source_regions();
  vector<int64_t> to_field(n_sr);
  for (int64_t sr = 0; sr < n_sr; ++sr)
    to_field[sr] = field.region(cell_id[sr], instance[sr], mesh_bin[sr]);

  // Rays of the given batch, the same for both sweeps
  int saved_batch = simulation::current_batch;
  simulation::current_batch = static_cast<int>(batch);
  d.set_segment_recording({}, true);

  SegmentRecords rec[2];
  for (int y = 0; y < 2; ++y) {
    // Source of tag y alone: the reduced source Q / Sigma_t, as
    // update_single_neutron_source() writes it
    for (int64_t sr = 0; sr < n_sr; ++sr) {
      int m = d.source_regions_.material(sr);
      for (int g = 0; g < G; ++g) {
        double q = 0.0;
        int64_t r = to_field[sr];
        if (m != MATERIAL_VOID && r >= 0) {
          int t = d.source_regions_.temperature_idx(sr);
          double dm = d.source_regions_.density_mult(sr);
          double sigma_t =
            d.sigma_t_[(static_cast<int64_t>(m) * d.ntemperature_ + t) * G + g] *
            dm;
          if (sigma_t > 0.0)
            q = field.emission(r, static_cast<rmc::Tag>(y), G - 1 - g) / sigma_t;
        }
        d.source_regions_.source(sr, g) = static_cast<float>(q);
      }
    }
    sweep_state->sim->sweep();
    if (d.n_source_regions() != n_sr) {
      simulation::current_batch = saved_batch;
      set_errmsg("The field sweep discovered new source regions; run random "
                 "ray batches first (openmc_rmc_sweep_iterate).");
      return OPENMC_E_ALLOCATE;
    }
    rec[y] = d.gather_segment_records();
  }
  simulation::current_batch = saved_batch;
  if (rec[0].size() != rec[1].size() || rec[0].length != rec[1].length) {
    set_errmsg("The two field sweeps did not trace the same rays.");
    return OPENMC_E_UNASSIGNED;
  }

  // Index the segments by field region (CSR), groups ascending, psi per
  // unit solid angle (records carry 4 pi psi)
  auto out = make_unique<rmc::SweepRecords>();
  out->n_groups = G;
  out->rho = settings::n_particles * RandomRay::distance_active_ /
             d.simulation_volume();
  const int64_t R = field.n_regions();
  const int64_t n = rec[0].size();
  out->offset.assign(R + 1, 0);
  for (int64_t i = 0; i < n; ++i) {
    int64_t r = to_field[rec[0].region[i]];
    if (r >= 0)
      ++out->offset[r + 1];
  }
  for (int64_t r = 0; r < R; ++r)
    out->offset[r + 1] += out->offset[r];
  const int64_t total = out->offset[R];
  out->u.resize(total);
  out->length.resize(total);
  out->psi_s.resize(total * G);
  out->psi_f.resize(total * G);
  vector<int64_t> next(out->offset.begin(), out->offset.end() - 1);
  for (int64_t i = 0; i < n; ++i) {
    int64_t r = to_field[rec[0].region[i]];
    if (r < 0)
      continue;
    int64_t j = next[r]++;
    out->u[j] = {rec[0].u[3 * i], rec[0].u[3 * i + 1], rec[0].u[3 * i + 2]};
    out->length[j] = rec[0].length[i];
    for (int g = 0; g < G; ++g) {
      out->psi_s[j * G + (G - 1 - g)] = rec[0].psi_bar[i * G + g] / (4.0 * PI);
      out->psi_f[j * G + (G - 1 - g)] = rec[1].psi_bar[i * G + g] / (4.0 * PI);
    }
  }
  rmc::loaded_sweep = std::move(out);
  *n_records = total;
  return 0;
}

extern "C" int openmc_rmc_sweep_field_region(int64_t region, double* rho,
  int64_t* n, double* u, double* length, double* psi_s, double* psi_f)
{
  if (!rmc::loaded_sweep) {
    set_errmsg("No field sweep is loaded.");
    return OPENMC_E_ALLOCATE;
  }
  const auto& sw = *rmc::loaded_sweep;
  *rho = sw.rho;
  int64_t lo = sw.offset[region], cnt = sw.count(region);
  *n = cnt;
  if (!u)
    return 0;
  const int G = sw.n_groups;
  for (int64_t j = 0; j < cnt; ++j) {
    u[3 * j] = sw.u[lo + j].x;
    u[3 * j + 1] = sw.u[lo + j].y;
    u[3 * j + 2] = sw.u[lo + j].z;
    length[j] = sw.length[lo + j];
    for (int g = 0; g < G; ++g) {
      psi_s[j * G + g] = sw.psi_s[(lo + j) * G + g];
      psi_f[j * G + g] = sw.psi_f[(lo + j) * G + g];
    }
  }
  return 0;
}

