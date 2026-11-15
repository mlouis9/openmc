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

namespace {

//! Bins of a 3-D regular mesh crossed by the segment start -> end (inside the
//! mesh), with the fraction of the length in each: an incremental grid
//! traversal (Amanatides-Woo). Returns false if the mesh is not a 3-D
//! regular mesh, so the caller falls back to Mesh::bins_crossed.
bool regular_bins_crossed(const Mesh& mesh, Position start, Position end,
  Direction u, vector<int>& bins, vector<double>& fractions)
{
  auto* rm = dynamic_cast<const RegularMesh*>(&mesh);
  if (!rm || rm->n_dimension_ != 3)
    return false;
  const double L = (end - start).norm();
  if (!(L > 0.0))
    return true;
  double ll[3] = {rm->lower_left_[0], rm->lower_left_[1], rm->lower_left_[2]};
  double w[3] = {rm->width_[0], rm->width_[1], rm->width_[2]};
  int n[3] = {rm->shape_[0], rm->shape_[1], rm->shape_[2]};
  double r0[3] = {start.x, start.y, start.z};
  double uu[3] = {u.x, u.y, u.z};
  int ijk[3];
  double t_max[3], t_delta[3];
  for (int a = 0; a < 3; ++a) {
    // Index of the start, nudged along the direction off any plane
    double x = (r0[a] + 1e-12 * L * uu[a] - ll[a]) / w[a];
    int i = static_cast<int>(std::floor(x));
    ijk[a] = std::min(std::max(i, 0), n[a] - 1);
    if (uu[a] > 0.0) {
      double plane = ll[a] + (ijk[a] + 1) * w[a];
      t_max[a] = (plane - r0[a]) / uu[a];
      t_delta[a] = w[a] / uu[a];
    } else if (uu[a] < 0.0) {
      double plane = ll[a] + ijk[a] * w[a];
      t_max[a] = (plane - r0[a]) / uu[a];
      t_delta[a] = -w[a] / uu[a];
    } else {
      t_max[a] = INFTY;
      t_delta[a] = INFTY;
    }
  }
  double t = 0.0;
  while (true) {
    int a = (t_max[0] < t_max[1]) ? ((t_max[0] < t_max[2]) ? 0 : 2)
                                   : ((t_max[1] < t_max[2]) ? 1 : 2);
    double t_next = std::min(t_max[a], L);
    double seg = t_next - t;
    if (seg > 0.0) {
      bins.push_back(ijk[0] + n[0] * (ijk[1] + n[1] * ijk[2]));
      fractions.push_back(seg / L);
    }
    t = t_next;
    if (t >= L)
      break;
    ijk[a] += (uu[a] > 0.0) ? 1 : -1;
    if (ijk[a] < 0 || ijk[a] >= n[a]) {
      // Left the mesh before the end (round-off at the edge): the caller
      // gives the remainder to the midpoint's bin
      break;
    }
    t_max[a] += t_delta[a];
  }
  return true;
}

} // namespace

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
  // The cell instance's mesh and region table, cached across steps
  if (cell.id_ != last_cell_ || inst != last_inst_) {
    last_cell_ = cell.id_;
    last_inst_ = inst;
    last_mesh_ = field_.mesh_of(cell.id_, inst);
    last_dense_ = field_.dense_regions(cell.id_, inst);
  }
  int32_t mesh_idx = last_mesh_;
  auto region_of = [&](int64_t bin) -> int64_t {
    if (last_dense_ && bin >= 0 && bin < static_cast<int64_t>(last_dense_->size()))
      return (*last_dense_)[bin];
    return field_.region(cell.id_, inst, bin);
  };

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
    if (!regular_bins_crossed(
          *model::meshes[mesh_idx], start, end, p_.u(), bins_, fractions_))
      model::meshes[mesh_idx]->bins_crossed(
        start, end, p_.u(), bins_, fractions_);
    double covered = 0.0;
    for (size_t b = 0; b < bins_.size(); ++b) {
      pending_.push_back(d * fractions_[b], material, sqrtkT, dm,
        region_of(bins_[b]));
      covered += fractions_[b];
    }
    // Round-off in the mesh crossing (very short or grazing segments): give
    // the missing length to the bin of the segment's midpoint. A gap larger
    // than round-off means the mesh does not cover the cell.
    double gap = 1.0 - covered;
    if (gap > 1e-9) {
      if (std::abs(gap) * d > 1e-6 && std::abs(gap) > 1e-3) {
        fatal_error("RMC tracer: a source-region mesh does not cover the cell "
                    "it is applied to.");
      }
      int bin = model::meshes[mesh_idx]->get_bin(start + 0.5 * d * p_.u());
      pending_.push_back(d * gap, material, sqrtkT, dm,
        bin >= 0 ? region_of(bin) : -1);
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
