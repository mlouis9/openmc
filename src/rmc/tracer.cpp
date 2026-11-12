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

  // TODO(you): trace the path and fill `out`. Theory: docs/source/methods/
  // rmc.rst, methods_rmc_tracer. Model it on random ray: read
  // RandomRay::transport_history_based_single_ray(), event_advance_ray(), and
  // attenuate_flux() in src/random_ray/random_ray.cpp first.
  //
  //   1. Make a scratch Particle (particle.h) at r with direction u. Set r(),
  //      u(), and also r_last(), u_last(), r_born() so that nothing reads
  //      uninitialized state. Find its cell with exhaustive_find_cell()
  //      (geometry.h); if that fails, fatal_error(): the C API checks the
  //      start point before calling you.
  //   2. Loop while out.size() < max_segments:
  //      a. p.boundary() = distance_to_boundary(p); d = its distance().
  //      b. The cell instance the particle is in: the lowest-level cell
  //         model::cells[p.lowest_coord().cell()] and p.cell_instance(). From
  //         the Cell, take material(instance), sqrtkT(instance), and
  //         density_mult(instance), as random ray does when it discovers a
  //         region.
  //      c. field_.mesh_of(cell.id_, instance) says how to split [0, d]:
  //         - -1 (no mesh): one segment, region field_.region(id, inst, 0);
  //         - -2 (an instance the field never reached): one segment, region
  //           -1 (the field is zero there, but the path still attenuates);
  //         - a mesh index m: call model::meshes[m]->bins_crossed(start, end,
  //           u, bins_, fractions_) with start = p.r(), end = start + d * u,
  //           after clearing the two scratch vectors. Each bin is a segment of
  //           length d * fraction, region field_.region(id, inst, bin). Stop
  //           as soon as out reaches max_segments. If the fractions do not
  //           add up to 1 (to 1e-9), the mesh does not cover the cell:
  //           fatal_error().
  //         Use out.push_back(length, material, sqrtkT, density_mult, region).
  //      d. Move every coordinate level to the surface:
  //         p.coord(j).r() += d * p.coord(j).u() for j < p.n_coord().
  //      e. p.event_cross_surface() applies the boundary condition: a
  //         reflection changes the direction, a vacuum boundary kills the
  //         particle. If !p.alive(), set out.ends_in_vacuum = true and return.
  //
  // tests/rmc/test_tracer.py compares the result with closed-form paths in
  // a slab box to 1e-9 cm, and test_multiple_meshes.py runs it through
  // several meshes. The C API (openmc_rmc_trace) and the Python binding
  // (openmc.lib.rmc.trace) are already written.
}

} // namespace rmc
} // namespace openmc
