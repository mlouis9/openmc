//! \file field.h
//! \brief The frozen multigroup field of residual Monte Carlo (RMC).
//!
//! Theory: docs/source/methods/rmc.rst, section "The Frozen Field" (label
//! methods_rmc_field).

#ifndef OPENMC_RMC_FIELD_H
#define OPENMC_RMC_FIELD_H

#include <cmath>
#include <cstdint>
#include <string>
#include <unordered_map>

#include "openmc/memory.h"
#include "openmc/particle_data.h"
#include "openmc/position.h"
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

  //! Indices into model::meshes of every source-region mesh the field uses,
  //! in order of first appearance. Different cell instances may use
  //! different meshes, as in random ray; each instance uses at most one.
  const vector<int32_t>& mesh_indices() const { return mesh_indices_; }

  //! The mesh whose voxels contain region r: an index into model::meshes, or
  //! -1 for a region of a cell instance without a mesh, or outside its mesh
  //! (mesh bin -1). Such regions lie only in the ray source box.
  int32_t region_mesh(int64_t r) const { return region_mesh_[r]; }

  //! Mesh applied to a cell instance: an index into model::meshes, -1 for
  //! none, or -2 for an instance random ray never reached (no regions).
  int32_t mesh_of(int32_t cell_id, int32_t instance) const;

  //! Region with the given key, or -1 if the field has no such region.
  int64_t region(int32_t cell_id, int32_t instance, int64_t mesh_bin) const;

  //! Ray source box of the random ray solve, which contains every region
  Position box_lower_left() const { return box_lower_left_; }
  Position box_upper_right() const { return box_upper_right_; }

  //----------------------------------------------------------------------------
  // Group structure, emission, and shapes (Stage 4 proposals and controls)

  //! Path of the field file (its material tables feed the MG control)
  const std::string& file() const { return field_file_; }

  //! Group edges [eV], ascending; group g is [edges[g], edges[g+1])
  const vector<double>& group_edges() const { return group_edges_; }

  //! Ascending group containing E, or -1 outside the group structure
  int group(double E) const;

  //! Group-integrated emission Q_{t,R,g} per unit volume, in the field's
  //! normalization (ascending g). evaluate() is Q * h_t(E) / (4 pi).
  double emission(int64_t region, Tag t, int g) const
  {
    return emission_[(static_cast<int64_t>(t) * n_regions() + region) *
                       n_groups_ + g];
  }

  //! Within-group shapes [1/eV], each normalized to 1 over every group:
  //! the emission lifts h_s, h_f and the flux shape h_phi
  enum class Shape { s = 0, f = 1, phi = 2 };
  bool has_shape(Shape k) const
  {
    return k != Shape::phi || !lift_phi_.empty();
  }
  double shape(Shape k, int64_t region, double E) const;

  //! Draw E from shape k within group g of the region's material; the shape
  //! is piecewise constant on the fine grid, so E is uniform within the fine
  //! bin drawn. Where the shape vanishes over the whole group, E is drawn
  //! log-uniformly in the group instead. Returns E; density is set to the
  //! density of the draw [1/eV].
  double sample_shape(
    Shape k, int64_t region, int g, uint64_t* seed, double& density) const;
  //! Density of sample_shape() at E in group g
  double shape_draw_density(Shape k, int64_t region, int g, double E) const;

  //! Region volume [cm^3] from the volume file; NaN without one
  double volume(int64_t region) const
  {
    return volume_.empty() ? std::nan("") : volume_[region];
  }
  bool has_volumes() const { return !volume_.empty(); }

  //! Material of a region: index into model::materials, or MATERIAL_VOID
  int32_t region_material(int64_t region) const
  {
    return region_material_[region];
  }

  //! Key of a region: cell ID, cell instance, and source-region mesh bin
  void region_key(int64_t region, int32_t& cell_id, int32_t& instance,
    int64_t& mesh_bin) const
  {
    cell_id = region_cell_[region];
    instance = region_instance_[region];
    mesh_bin = region_bin_[region];
  }

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

  //! A base region: a cell instance, before any mesh subdivision.
  struct Base {
    int32_t cell;
    int32_t instance;
    bool operator==(const Base& o) const
    {
      return cell == o.cell && instance == o.instance;
    }
  };
  struct BaseHash {
    size_t operator()(const Base& b) const;
  };

  //----------------------------------------------------------------------------
  // Data. Flat arrays with documented index order.

  std::unordered_map<Key, int64_t, KeyHash> key_to_region_;
  //! Mesh of each base region in the field: index into model::meshes, or -1
  //! for none. Base regions random ray never reached have no entry.
  std::unordered_map<Base, int32_t, BaseHash> mesh_of_base_;
  vector<int32_t> mesh_indices_; //!< distinct entries of mesh_of_base_ >= 0
  vector<int32_t> region_mesh_;  //!< [R], see region_mesh()
  Position box_lower_left_;
  Position box_upper_right_;

  int n_groups_ {0};
  vector<double> group_edges_; //!< [G+1] ascending energy [eV]
  vector<double>
    emission_; //!< [2][R][G]: emission_[(t*R + r)*G + g], normalized

  int n_fine_ {0};
  int n_lift_materials_ {0};
  vector<double> fine_edges_; //!< [F+1] ascending energy [eV]
  vector<double> lift_;       //!< [2][M][F]: lift_[(t*M + row)*F + i] [1/eV]
  vector<int32_t> lift_row_;  //!< [R] row of the region's material, or -1
  vector<double> lift_phi_;  //!< [M][F] flux shape h_phi, if the lift file
                             //!< has one
  vector<int> group_fine_;   //!< [G+1] fine-bin index of each group edge

  std::string field_file_;
  vector<double> volume_;           //!< [R] from the volume file, or empty
  vector<int32_t> region_material_; //!< [R] index into model::materials
  vector<int32_t> region_cell_;     //!< [R] cell ID
  vector<int32_t> region_instance_; //!< [R]
  vector<int64_t> region_bin_;      //!< [R]

  double mass_[2] {0.0, 0.0};
};

//! The field loaded through the C API (openmc_rmc_field_load), defined in
//! src/rmc/capi.cpp
extern unique_ptr<FrozenField> loaded_field;

} // namespace rmc
} // namespace openmc

#endif // OPENMC_RMC_FIELD_H
