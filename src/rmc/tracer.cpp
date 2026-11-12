#include "openmc/rmc/tracer.h"

#include <cmath>

#include <fmt/format.h>

#include "openmc/cell.h"
#include "openmc/constants.h"
#include "openmc/error.h"
#include "openmc/geometry.h"
#include "openmc/mesh.h"
#include "openmc/particle.h"

namespace openmc {
namespace rmc {

void CharacteristicTracer::trace(
  Position r, Direction u, int max_segments, SegmentList& out)
{
  out.clear();

  // A scratch particle: only its geometry state and boundary handling are used
  Particle p;
  p.r() = r;
  p.u() = u;
  p.r_last() = r;
  p.u_last() = u;
  p.r_born() = r;
  if (!exhaustive_find_cell(p)) {
    fatal_error(fmt::format("RMC tracer: start point ({}, {}, {}) is outside "
                            "the geometry.",
      r.x, r.y, r.z));
  }

  while (out.size() < max_segments) {
    // Distance to the next surface (or lattice) crossing
    p.boundary() = distance_to_boundary(p);
    double d = p.boundary().distance();

    // What the path passes through until then
    const Cell& cell = *model::cells[p.lowest_coord().cell()];
    int32_t inst = p.cell_instance();
    int32_t material = cell.material(inst);
    double sqrtkT = cell.sqrtkT(inst);
    double dm = cell.density_mult(inst);
    int32_t mesh_idx = field_.mesh_of(cell.id_, inst);

    if (mesh_idx < 0) {
      // No mesh on this instance (or an instance the field never reached)
      int64_t region = (mesh_idx == -1) ? field_.region(cell.id_, inst, 0) : -1;
      out.push_back(d, material, sqrtkT, dm, region);
    } else {
      // Split at mesh bin crossings, as RandomRay::attenuate_flux() does
      Position start = p.r();
      Position end = start + d * p.u();
      bins_.clear();
      fractions_.clear();
      model::meshes[mesh_idx]->bins_crossed(
        start, end, p.u(), bins_, fractions_);
      double covered = 0.0;
      for (size_t b = 0; b < bins_.size(); ++b) {
        if (out.size() >= max_segments)
          return;
        out.push_back(d * fractions_[b], material, sqrtkT, dm,
          field_.region(cell.id_, inst, bins_[b]));
        covered += fractions_[b];
      }
      if (std::abs(covered - 1.0) > 1e-9) {
        fatal_error("RMC tracer: a source-region mesh does not cover the cell "
                    "it is applied to.");
      }
    }

    // Move to the surface and cross it: reflection changes the direction,
    // a vacuum boundary kills the particle
    for (int j = 0; j < p.n_coord(); ++j)
      p.coord(j).r() += d * p.coord(j).u();
    p.event_cross_surface();
    if (!p.alive()) {
      out.ends_in_vacuum = true;
      return;
    }
  }
}

} // namespace rmc
} // namespace openmc
