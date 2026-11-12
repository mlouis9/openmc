//! \file distribution_energy.h
//! Energy distributions that depend on incident particle energy

#ifndef OPENMC_DISTRIBUTION_ENERGY_H
#define OPENMC_DISTRIBUTION_ENERGY_H

#include "hdf5.h"
#include "openmc/tensor.h"

#include "openmc/constants.h"
#include "openmc/endf.h"
#include "openmc/vector.h"

namespace openmc {

//===============================================================================
//! Abstract class defining an energy distribution that is a function of the
//! incident energy of a projectile. Each derived type must implement a sample()
//! function that returns a sampled outgoing energy given an incoming energy
//===============================================================================

class EnergyDistribution {
public:
  virtual double sample(double E, uint64_t* seed) const = 0;

  //! Probability density of the outgoing energy that sample() produces
  //! \param[in] E Incident energy in [eV]
  //! \param[in] E_out Outgoing energy in [eV]
  //! \return Density in [1/eV]. The default stops with an error, so that a
  //!   law without an implementation is never silently treated as zero.
  virtual double evaluate(double E, double E_out) const;

  virtual ~EnergyDistribution() = default;
};

//===============================================================================
//! Discrete photon energy distribution
//===============================================================================

class DiscretePhoton : public EnergyDistribution {
public:
  explicit DiscretePhoton(hid_t group);

  //! Sample energy distribution
  //! \param[in] E Incident particle energy in [eV]
  //! \param[inout] seed Pseudorandom number seed pointer
  //! \return Sampled energy in [eV]
  double sample(double E, uint64_t* seed) const override;

private:
  int primary_flag_; //!< Indicator of whether the photon is a primary or
                     //!< non-primary photon.
  double energy_;    //!< Photon energy or binding energy
  double A_;         //!< Atomic weight ratio of the target nuclide
};

//===============================================================================
//! Level inelastic scattering distribution
//===============================================================================

class LevelInelastic : public EnergyDistribution {
public:
  explicit LevelInelastic(hid_t group);

  //! Sample energy distribution
  //! \param[in] E Incident particle energy in [eV]
  //! \param[inout] seed Pseudorandom number seed pointer
  //! \return Sampled energy in [eV]
  double sample(double E, uint64_t* seed) const override;

private:
  double threshold_;  //!< Energy threshold in lab, (A + 1)/A * |Q|
  double mass_ratio_; //!< (A/(A+1))^2
};

//===============================================================================
//! An energy distribution represented as a tabular distribution with histogram
//! or linear-linear interpolation. This corresponds to ACE law 4, which NJOY
//! produces for a number of ENDF energy distributions.
//===============================================================================

class ContinuousTabular : public EnergyDistribution {
public:
  explicit ContinuousTabular(hid_t group);

  //! Sample energy distribution
  //! \param[in] E Incident particle energy in [eV]
  //! \param[inout] seed Pseudorandom number seed pointer
  //! \return Sampled energy in [eV]
  double sample(double E, uint64_t* seed) const override;

  //! Density of the outgoing energy that sample() produces, including the
  //! stochastic choice between neighboring incident tables and the rescaling
  //! of the sampled energy to interpolated endpoints. Continuous parts only;
  //! tables with discrete lines are refused.
  double evaluate(double E, double E_out) const override;

  //! Cumulative distribution of the outgoing energy, consistent with
  //! evaluate(): the probability that sample() returns less than E_out.
  double cdf(double E, double E_out) const;

private:
  //! Interpolation weights of the incident tables sample() may choose, and
  //! the linear maps from the interpolated outgoing range onto each table.
  struct Mix {
    int n;       //!< number of tables (1 or 2)
    int l[2];    //!< table indices
    double w[2]; //!< probabilities of choosing them
    double a[2]; //!< E_table = a + b * E_out
    double b[2]; //!< dE_table / dE_out
  };
  Mix mix(double E) const;

  //! Density and CDF of a single table at its own outgoing energy
  double table_pdf(int l, double E_out) const;
  double table_cdf(int l, double E_out) const;

  //! Outgoing energy for a single incoming energy
  struct CTTable {
    Interpolation interpolation;  //!< Interpolation law
    int n_discrete;               //!< Number of of discrete energies
    tensor::Tensor<double> e_out; //!< Outgoing energies in [eV]
    tensor::Tensor<double> p;     //!< Probability density
    tensor::Tensor<double> c;     //!< Cumulative distribution
  };

  int n_region_;                        //!< Number of inteprolation regions
  vector<int> breakpoints_;             //!< Breakpoints between regions
  vector<Interpolation> interpolation_; //!< Interpolation laws
  vector<double> energy_;               //!< Incident energy in [eV]
  vector<CTTable> distribution_; //!< Distributions for each incident energy
};

//===============================================================================
//! Evaporation spectrum corresponding to ACE law 9 and ENDF File 5, LF=9.
//===============================================================================

class Evaporation : public EnergyDistribution {
public:
  explicit Evaporation(hid_t group);

  //! Sample energy distribution
  //! \param[in] E Incident particle energy in [eV]
  //! \param[inout] seed Pseudorandom number seed pointer
  //! \return Sampled energy in [eV]
  double sample(double E, uint64_t* seed) const override;

private:
  Tabulated1D theta_; //!< Incoming energy dependent parameter
  double u_;          //!< Restriction energy
};

//===============================================================================
//! Energy distribution of neutrons emitted from a Maxwell fission spectrum.
//! This corresponds to ACE law 7 and ENDF File 5, LF=7.
//===============================================================================

class MaxwellEnergy : public EnergyDistribution {
public:
  explicit MaxwellEnergy(hid_t group);

  //! Sample energy distribution
  //! \param[in] E Incident particle energy in [eV]
  //! \param[inout] seed Pseudorandom number seed pointer
  //! \return Sampled energy in [eV]
  double sample(double E, uint64_t* seed) const override;

private:
  Tabulated1D theta_; //!< Incoming energy dependent parameter
  double u_;          //!< Restriction energy
};

//===============================================================================
//! Energy distribution of neutrons emitted from a Watt fission spectrum. This
//! corresponds to ACE law 11 and ENDF File 5, LF=11.
//===============================================================================

class WattEnergy : public EnergyDistribution {
public:
  explicit WattEnergy(hid_t group);

  //! Sample energy distribution
  //! \param[in] E Incident particle energy in [eV]
  //! \param[inout] seed Pseudorandom number seed pointer
  //! \return Sampled energy in [eV]
  double sample(double E, uint64_t* seed) const override;

private:
  Tabulated1D a_; //!< Energy-dependent 'a' parameter
  Tabulated1D b_; //!< Energy-dependent 'b' parameter
  double u_;      //!< Restriction energy
};

} // namespace openmc

#endif // OPENMC_DISTRIBUTION_ENERGY_H
