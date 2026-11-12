//! \file random_ray_capi.cpp
//! \brief C API that runs OpenMC's random ray solver inside a
//! continuous-energy RMC process.

#include "openmc/capi.h"
#include "openmc/error.h"
#include "openmc/random_ray/random_ray.h"
#include "openmc/rmc/capi.h"
#include "openmc/settings.h"
#include "openmc/simulation.h"

using namespace openmc;

extern "C" int openmc_rmc_run_random_ray()
{
  if (!settings::run_CE) {
    set_errmsg("openmc_rmc_run_random_ray requires a continuous-energy model; "
               "use openmc_run for a multigroup random ray run.");
    return OPENMC_E_INVALID_ARGUMENT;
  }
  if (!settings::rmc_enabled || settings::rmc_field_file.empty()) {
    set_errmsg(
      "openmc_rmc_run_random_ray requires settings.rmc['field_file'].");
    return OPENMC_E_INVALID_ARGUMENT;
  }
  if (!RandomRay::ray_source_) {
    set_errmsg("openmc_rmc_run_random_ray requires a <random_ray> element in "
               "the settings.");
    return OPENMC_E_INVALID_ARGUMENT;
  }
  if (simulation::initialized) {
    set_errmsg("openmc_rmc_run_random_ray cannot run while a simulation is "
               "initialized; call openmc_simulation_finalize first.");
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
