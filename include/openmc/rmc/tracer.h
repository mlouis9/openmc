//! \file tracer.h
//! \brief The geometry half of the backward continuous-energy characteristic.

#ifndef OPENMC_RMC_TRACER_H
#define OPENMC_RMC_TRACER_H

#include <cstdint>

#include "openmc/particle.h"
#include "openmc/position.h"
#include "openmc/rmc/field.h"
#include "openmc/vector.h"

namespace openmc {
namespace rmc {

//==============================================================================
//! The segments of one backward path, as structure-of-arrays.
//!
//! Each segment lies in one cell instance and, if a source-region mesh is
//! applied to that instance, in one bin of it. Segments are in order along
//! the path, starting at the destination. Reflections change the path's
//! direction but are not recorded: the field is isotropic, so only the
//! lengths and what the path passes through matter.
//==============================================================================

struct SegmentList {
  vector<double> length;       //!< [cm]
  vector<int32_t> material;    //!< index into model::materials, or
                               //!< MATERIAL_VOID
  vector<double> sqrtkT;       //!< sqrt(kT) [sqrt(eV)] of the cell instance
  vector<double> density_mult; //!< density multiplier of the cell instance
  vector<int64_t> region;      //!< field region, or -1 where the field is 0
  bool ends_in_vacuum {false}; //!< the path left through a vacuum boundary

  int size() const { return static_cast<int>(length.size()); }

  //! Empty the list, keeping capacity (no allocation in steady state)
  void clear()
  {
    length.clear();
    material.clear();
    sqrtkT.clear();
    density_mult.clear();
    region.clear();
    ends_in_vacuum = false;
  }

  void push_back(double l, int32_t m, double kT, double dm, int64_t r)
  {
    length.push_back(l);
    material.push_back(m);
    sqrtkT.push_back(kT);
    density_mult.push_back(dm);
    region.push_back(r);
  }
};

//==============================================================================
//! Traces straight paths through the model, applying boundary conditions,
//! and splits them into segments by cell instance and source-region mesh
//! bin, as random ray does (RandomRay::event_advance_ray() and
//! RandomRay::attenuate_flux()).
//!
//! Theory: docs/source/methods/rmc.rst, methods_rmc_tracer.
//==============================================================================

class CharacteristicTracer {
public:
  explicit CharacteristicTracer(const FrozenField& field);

  //! Trace from r in direction u (for the backward characteristic, u is
  //! minus the incident direction) until the path leaves through a vacuum
  //! boundary or max_segments segments have been recorded. The same as
  //! begin(r, u) followed by one extend(max_segments, out).
  //! \param r             Start point; must be inside the geometry
  //! \param u             Unit direction
  //! \param max_segments  Upper bound on out.size()
  //! \param out           Cleared, then filled
  void trace(Position r, Direction u, int max_segments, SegmentList& out);

  //! Start a path at r in direction u, to be read in chunks with extend().
  void begin(Position r, Direction u);

  //! Clear out and fill it with the next segments of the path, at most
  //! max_segments of them. out.ends_in_vacuum is set on the chunk that holds
  //! the path's last segment; after that, done() is true.
  void extend(int max_segments, SegmentList& out);

  //! True once every segment of a path that left through vacuum was returned
  bool done() const { return ended_ && next_ == pending_.size(); }

private:
  //! Trace one cell crossing: its segments (split by mesh bin) go into
  //! pending_, and the particle moves to the surface and crosses it.
  void step();

  const FrozenField& field_;
  Particle p_;               //!< scratch particle, reused between paths
  SegmentList pending_;      //!< segments of the current cell crossing
  int next_ {0};             //!< next segment of pending_ to return
  bool ended_ {true};        //!< the path left through a vacuum boundary
  vector<int> bins_;         //!< scratch for Mesh::bins_crossed
  vector<double> fractions_; //!< scratch for Mesh::bins_crossed
};

} // namespace rmc
} // namespace openmc

#endif // OPENMC_RMC_TRACER_H
