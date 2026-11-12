//! \file production_capi.cpp
//! \brief C API for the production evaluators, and for OpenMC's own collision
//! samplers called on a scratch particle (the references the evaluators are
//! tested against).

#include "openmc/capi.h"
#include "openmc/constants.h"
#include "openmc/distribution_energy.h"
#include "openmc/error.h"
#include "openmc/hdf5_interface.h"
#include "openmc/nuclide.h"
#include "openmc/particle.h"
#include "openmc/physics.h"
#include "openmc/rmc/capi.h"
#include "openmc/rmc/production.h"

using namespace openmc;

namespace {

int check_nuclide(int i_nuclide, bool fissionable)
{
  if (i_nuclide < 0 || i_nuclide >= static_cast<int>(data::nuclides.size())) {
    set_errmsg("Nuclide index out of range.");
    return OPENMC_E_OUT_OF_BOUNDS;
  }
  if (fissionable && !data::nuclides[i_nuclide]->fissionable_) {
    set_errmsg("Nuclide is not fissionable.");
    return OPENMC_E_INVALID_ARGUMENT;
  }
  return 0;
}

//! A scratch particle with its own random number stream
void scratch(Particle& p, uint64_t seed)
{
  for (int s = 0; s < N_STREAMS; ++s)
    p.seeds(s) = seed + s;
  p.stream() = STREAM_TRACKING;
}

} // namespace

extern "C" int openmc_rmc_fission_density(int i_nuclide, double E_in, int64_t n,
  const double* E_out, double* density, double* cdf)
{
  if (int err = check_nuclide(i_nuclide, true))
    return err;
  const Reaction& rx = *data::nuclides[i_nuclide]->fission_rx_[0];
  for (int64_t i = 0; i < n; ++i) {
    density[i] = rmc::fission_energy_density(i_nuclide, rx, E_in, E_out[i]);
    cdf[i] = rmc::fission_energy_cdf(i_nuclide, rx, E_in, E_out[i]);
  }
  return 0;
}

extern "C" int openmc_rmc_sample_fission_energy(
  int i_nuclide, double E_in, int64_t n, uint64_t seed, double* E_out)
{
  if (int err = check_nuclide(i_nuclide, true))
    return err;
  const Reaction& rx = *data::nuclides[i_nuclide]->fission_rx_[0];
  Particle p;
  scratch(p, seed);
  p.E() = E_in;
  SourceSite site;
  for (int64_t i = 0; i < n; ++i) {
    sample_fission_neutron(i_nuclide, rx, &site, p);
    E_out[i] = site.E;
  }
  return 0;
}

extern "C" int openmc_rmc_elastic_image(int i_nuclide, double kT, int64_t n,
  const double* E, const double* mu0, double* E_in, double* density)
{
  if (int err = check_nuclide(i_nuclide, false))
    return err;
  for (int64_t i = 0; i < n; ++i) {
    auto img = rmc::elastic_stationary(i_nuclide, kT, E[i], mu0[i]);
    E_in[i] = img.E_in;
    density[i] = img.density;
  }
  return 0;
}

extern "C" int openmc_rmc_sample_elastic(int i_nuclide, double kT, int64_t n,
  const double* E_in, uint64_t seed, double* E_out, double* mu0)
{
  if (int err = check_nuclide(i_nuclide, false))
    return err;
  const auto& nuc = *data::nuclides[i_nuclide];
  Particle p;
  scratch(p, seed);
  for (int64_t i = 0; i < n; ++i) {
    p.E() = E_in[i];
    p.u() = {0.0, 0.0, 1.0};
    p.neutron_xs(i_nuclide).use_ptable = false;
    elastic_scatter(i_nuclide, *nuc.reactions_[0], kT, p);
    E_out[i] = p.E();
    mu0[i] = p.mu();
  }
  return 0;
}

extern "C" int openmc_rmc_tabular_law(const char* path, double E_in, int64_t n,
  const double* E_out, double* density, double* cdf, int64_t n_samples,
  uint64_t seed, double* samples)
{
  // A ContinuousTabular law read from group "law" of an HDF5 file, so that
  // tests can build laws whose tables have different outgoing ranges
  hid_t file = file_open(path, 'r');
  hid_t group = open_group(file, "law");
  ContinuousTabular law(group);
  close_group(group);
  file_close(file);
  for (int64_t i = 0; i < n; ++i) {
    density[i] = law.evaluate(E_in, E_out[i]);
    cdf[i] = law.cdf(E_in, E_out[i]);
  }
  uint64_t s = seed;
  for (int64_t i = 0; i < n_samples; ++i)
    samples[i] = law.sample(E_in, &s);
  return 0;
}
