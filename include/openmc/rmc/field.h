//! \file field.h
//! \brief The frozen multigroup field of multilevel RMC.
//!
//! Theory: docs/source/methods/rmc.rst, section "The Frozen Field" (label
//! methods_rmc_field).

#ifndef OPENMC_RMC_FIELD_H
#define OPENMC_RMC_FIELD_H

#include <cstdint>
#include <string>
#include <unordered_map>

#include "openmc/particle_data.h"
#include "openmc/vector.h"

namespace openmc {
namespace rmc {

//! Emission tag: scattering-born (s) or fission-born (f).
enum class Tag : int { s = 0, f = 1 };

//==============================================================================
//! A pointwise-evaluable emission source.
//!
//! Abstract so that a linear-source or swept field can replace the frozen
//! flat one without changing its users.
//==============================================================================

class FieldSource {
public:
  virtual ~FieldSource() = default;

  //! Region index at a located geometry state.
  //! \param p  Geometry state whose cell has been found
  //! \return   Region index, or -1 where the field is zero by definition
  virtual int64_t locate(const GeometryState& p) const = 0;

  //! Emission per unit volume, energy, and solid angle, eq. rmc-field.
  //! \param region  Index returned by locate(); -1 gives 0
  //! \param t       Tag
  //! \param E       Energy [eV]
  //! \return        Q_t [1/(cm^3 eV sr)] in the field's normalization
  virtual double evaluate(int64_t region, Tag t, double E) const = 0;

  //! Field mass <1, Q_t> under the frozen volumes, eq. rmc-field-norm.
  virtual double mass(Tag t) const = 0;
};

//==============================================================================
//! The frozen flat-source field read from the files written by random ray
//! (field), the lift tallies (lifts), and the region-volume calculation
//! (volumes).
//==============================================================================

class FrozenField final : public FieldSource {
public:
  //! \param field_file   Written by random ray (rmc_field_file setting)
  //! \param lift_file    Within-group shapes per material
  //! \param volume_file  Region volumes and normalization; an empty string
  //!                     loads the field unnormalized, with NaN masses
  FrozenField(const std::string& field_file, const std::string& lift_file,
    const std::string& volume_file);

  int64_t locate(const GeometryState& p) const override;
  double evaluate(int64_t region, Tag t, double E) const override;
  double mass(Tag t) const override { return mass_[static_cast<int>(t)]; }

  int64_t n_regions() const { return static_cast<int64_t>(lift_row_.size()); }
  int n_groups() const { return n_groups_; }
  //! Index into model::meshes of the source-region mesh, or -1.
  int32_t mesh_index() const { return mesh_index_; }

private:
  //----------------------------------------------------------------------------
  // Region keys

  //! Identifies a region the way random ray does: lowest-level cell ID, cell
  //! instance, and source-region mesh bin.
  struct Key {
    int32_t cell;
    int32_t instance;
    int64_t mesh_bin;
    bool operator==(const Key& o) const
    {
      return cell == o.cell && instance == o.instance && mesh_bin == o.mesh_bin;
    }
  };
  struct KeyHash {
    size_t operator()(const Key& k) const;
  };

  //----------------------------------------------------------------------------
  // Data. Flat arrays with documented index order.

  std::unordered_map<Key, int64_t, KeyHash> key_to_region_;
  int32_t mesh_index_ {-1}; //!< index into model::meshes, or -1

  int n_groups_ {0};
  vector<double> group_edges_; //!< [G+1] ascending energy [eV]
  vector<double>
    emission_; //!< [2][R][G]: emission_[(t*R + r)*G + g], normalized

  int n_fine_ {0};
  int n_lift_materials_ {0};
  vector<double> fine_edges_; //!< [F+1] ascending energy [eV]
  vector<double> lift_;       //!< [2][M][F]: lift_[(t*M + row)*F + i] [1/eV]
  vector<int32_t> lift_row_;  //!< [R] row of the region's material, or -1

  double mass_[2] {0.0, 0.0};
};

} // namespace rmc
} // namespace openmc

#endif // OPENMC_RMC_FIELD_H
