#include "openmc/rmc/tracer.h"

#include <cmath>

#include <fmt/format.h>

#include "openmc/cell.h"
#include "openmc/constants.h"
#include "openmc/error.h"
#include "openmc/geometry.h"
#include "openmc/mesh.h"

namespace openmc {
namespace rmc {

CharacteristicTracer::CharacteristicTracer(const FrozenField& field)
  : field_(field)
{}

void CharacteristicTracer::begin(Position r, Direction u)
{
  // A scratch particle: only its geometry state and boundary handling are
  // used. Its coordinate levels are reset by the cell search.
  p_.r() = r;
  p_.u() = u;
  p_.r_last() = r;
  p_.u_last() = u;
  p_.r_born() = r;
  p_.wgt() = 1.0;
  p_.surface() = SURFACE_NONE;
  p_.n_coord() = 1;
  if (!exhaustive_find_cell(p_)) {
    fatal_error(fmt::format("RMC tracer: start point ({}, {}, {}) is outside "
                            "the geometry.",
      r.x, r.y, r.z));
  }
  pending_.clear();
  next_ = 0;
  ended_ = false;
}

void CharacteristicTracer::step()
{
  pending_.clear();
  next_ = 0;

  // Distance to the next surface (or lattice) crossing
  p_.boundary() = distance_to_boundary(p_);
  double d = p_.boundary().distance();

  // What the path passes through until then
  const Cell& cell = *model::cells[p_.lowest_coord().cell()];
  int32_t inst = p_.cell_instance();
  int32_t material = cell.material(inst);
  double sqrtkT = cell.sqrtkT(inst);
  double dm = cell.density_mult(inst);
  int32_t mesh_idx = field_.mesh_of(cell.id_, inst);

  if (mesh_idx < 0) {
    // No mesh on this instance (or an instance the field never reached)
    int64_t region = (mesh_idx == -1) ? field_.region(cell.id_, inst, 0) : -1;
    pending_.push_back(d, material, sqrtkT, dm, region);
  } else {
    // Split at mesh bin crossings, as RandomRay::attenuate_flux() does
    Position start = p_.r();
    Position end = start + d * p_.u();
    bins_.clear();
    fractions_.clear();
    model::meshes[mesh_idx]->bins_crossed(
      start, end, p_.u(), bins_, fractions_);
    double covered = 0.0;
    for (size_t b = 0; b < bins_.size(); ++b) {
      pending_.push_back(d * fractions_[b], material, sqrtkT, dm,
        field_.region(cell.id_, inst, bins_[b]));
      covered += fractions_[b];
    }
    if (std::abs(covered - 1.0) > 1e-9) {
      fatal_error("RMC tracer: a source-region mesh does not cover the cell it "
                  "is applied to.");
    }
  }

  // Move to the surface and cross it: reflection changes the direction, a
  // vacuum boundary kills the particle
  for (int j = 0; j < p_.n_coord(); ++j)
    p_.coord(j).r() += d * p_.coord(j).u();
  p_.event_cross_surface();
  if (!p_.alive())
    ended_ = true;
}

void CharacteristicTracer::extend(int max_segments, SegmentList& out)
{
  out.clear();
  while (out.size() < max_segments) {
    if (next_ < pending_.size()) {
      int k = next_++;
      out.push_back(pending_.length[k], pending_.material[k],
        pending_.sqrtkT[k], pending_.density_mult[k], pending_.region[k]);
    } else if (ended_) {
      out.ends_in_vacuum = true;
      return;
    } else {
      step();
    }
  }
  // A full chunk that happens to hold the last segment
  if (done())
    out.ends_in_vacuum = true;
}

void CharacteristicTracer::trace(
  Position r, Direction u, int max_segments, SegmentList& out)
{
  begin(r, u);
  extend(max_segments, out);
}

} // namespace rmc
} // namespace openmc
