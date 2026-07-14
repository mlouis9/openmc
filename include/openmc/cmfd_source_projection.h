//! \file cmfd_source_projection.h
//! \brief External source projection onto CMFD mesh for subcritical
//! multiplication mode

#ifndef OPENMC_CMFD_SOURCE_PROJECTION_H
#define OPENMC_CMFD_SOURCE_PROJECTION_H

#include "xtensor/xtensor.hpp"
#include <array>
#include <vector>

namespace openmc {

// Forward declarations
class StructuredMesh;
struct SourceSite;

namespace cmfd {

/**
 * Projects external source distribution onto CMFD mesh for subcritical
 * multiplication mode.
 *
 * In subcritical multiplication mode (fixed-source with fission), the CMFD
 * solver iteratively solves the fixed-source linear system:
 *
 *     M*φ = F*φ + Q
 *
 * Where:
 *     M = total interaction matrix (diagonal: Σ_t)
 *     F = fission production matrix (ν*Σ_f)
 *     Q = external source term (to be computed here)
 *     φ = scalar flux (solution computed by solver)
 *
 * Note: This is NOT an eigenvalue problem. The system uses a fixed source Q.
 * After solving for φ, the multiplication factor k_eff is computed as:
 *
 *     k_eff = (total fission source from φ) / (total absorption of Q)
 *
 * This represents how many fissions are produced per neutron absorbed from
 * the external source.
 *
 * Samples N_sample particles from the external source distribution and bins
 * them onto the CMFD mesh (spatial × energy). Returns a normalized source
 * distribution that sums to 1.0 across all mesh cells and energy groups.
 *
 * @param mesh              Structured mesh for CMFD (must be non-null)
 * @param egrid             Energy grid (size ng+1) in ascending order [eV]
 * @param n_sample_particles Number of source particles to sample (higher =
 * better statistics)
 * @param mesh_dims         Array [nx, ny, nz, ng] of mesh dimensions
 * @return                  xt::xtensor<double, 4> of shape (nx, ny, nz, ng)
 *                          Returns normalized source Q (integral = 1.0)
 *
 * @throws fatal_error if mesh is null or egrid size doesn't match ng
 *
 * \note This function performs MPI reduction if compiled with OPENMC_MPI
 * \note The returned array is normalized such that sum(Q) = 1.0
 * \note Q is passed to CMFD solver as the RHS of: M*φ = F*φ + Q
 */
xt::xtensor<double, 4> project_external_source_to_mesh(StructuredMesh* mesh,
  const std::vector<double>& egrid, int64_t n_sample_particles,
  const std::array<int, 4>& mesh_dims);

/**
 * Bins a single source site into CMFD mesh.
 *
 * Determines the flat bin index (mesh_bin * ng + energy_bin) for a source site.
 * Returns -1 if the site is outside the mesh or energy bounds.
 *
 * @param site    SourceSite to bin (with r, E, wgt fields)
 * @param mesh    Structured mesh for spatial binning
 * @param egrid   Energy grid (size ng+1) in ascending order
 * @param ng      Number of energy groups
 * @return        Flat bin index [0, nx*ny*nz*ng), or -1 if outside mesh/bounds
 *
 * \note Does not perform bounds checking; caller should verify result >= 0
 */
int bin_source_site_to_cmfd(const SourceSite& site, StructuredMesh* mesh,
  const std::vector<double>& egrid, int ng);

/**
 * Get energy bin index for a given energy value.
 *
 * Returns the energy group index g such that egrid[g] <= E < egrid[g+1].
 * Handles out-of-bounds gracefully with warnings.
 *
 * @param E       Energy value in eV
 * @param egrid   Energy grid (size ng+1) in ascending order
 * @param ng      Number of energy groups
 * @return        Energy bin index [0, ng), clamped to valid range
 *
 * \note Logs warnings for out-of-bounds energies but doesn't fail
 */
int get_energy_bin_cmfd(double E, const std::vector<double>& egrid, int ng);

} // namespace cmfd
} // namespace openmc

#endif // OPENMC_CMFD_SOURCE_PROJECTION_H