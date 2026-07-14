//! \file cmfd_source_projection.cpp
//! \brief Implementation of external source projection onto CMFD mesh

#include "openmc/cmfd_source_projection.h"
#include "openmc/bank.h"
#include "openmc/error.h"
#include "openmc/mesh.h"
#include "openmc/message_passing.h"
#include "openmc/source.h"
#include <algorithm>
#include <cmath>
#include <fmt/core.h>

namespace openmc {
namespace cmfd {

int get_energy_bin_cmfd(double E, const std::vector<double>& egrid, int ng)
{
  // Clamp to grid bounds
  if (E < egrid[0]) {
    warning(
      fmt::format("External source energy {} eV below CMFD grid minimum {} eV. "
                  "Clamping to first group.",
        E, egrid[0]));
    return 0;
  }
  if (E >= egrid[ng]) {
    warning(
      fmt::format("External source energy {} eV above CMFD grid maximum {} eV. "
                  "Clamping to last group.",
        E, egrid[ng]));
    return ng - 1;
  }

  // Binary search for energy group containing E
  // We want index g such that egrid[g] <= E < egrid[g+1]
  int g = std::lower_bound(egrid.begin(), egrid.end(), E) - egrid.begin();

  // lower_bound returns iterator to first element >= E
  // So we need g-1 to get the bin where egrid[g-1] <= E < egrid[g]
  return std::max(0, std::min(ng - 1, g - 1));
}

int bin_source_site_to_cmfd(const SourceSite& site, StructuredMesh* mesh,
  const std::vector<double>& egrid, int ng)
{
  // Get spatial mesh bin
  int mesh_bin = mesh->get_bin(site.r);
  if (mesh_bin < 0) {
    return -1; // Outside mesh
  }

  // Get energy bin
  int energy_bin = get_energy_bin_cmfd(site.E, egrid, ng);

  // Return flat index: mesh_bin * ng + energy_bin
  // This matches the indexing used in CMFD: (i,j,k,g) -> (i*ny*nz*ng + j*nz*ng
  // + k*ng + g)
  return mesh_bin * ng + energy_bin;
}

xt::xtensor<double, 4> project_external_source_to_mesh(StructuredMesh* mesh,
  const std::vector<double>& egrid, int64_t n_sample_particles,
  const std::array<int, 4>& mesh_dims)
{
  if (!mesh) {
    fatal_error("CMFD mesh is null when projecting external source");
  }

  int nx = mesh_dims[0];
  int ny = mesh_dims[1];
  int nz = mesh_dims[2];
  int ng = mesh_dims[3];

  if (static_cast<int>(egrid.size()) != ng + 1) {
    fatal_error(fmt::format(
      "Energy grid size ({}) does not match ng+1 ({})", egrid.size(), ng + 1));
  }

  // SUBCRITICAL MULTIPLICATION MODE (Fixed-Source Fission Problem)
  // ============================================================
  // We solve the linear system (single iteration, no eigenvalue):
  //
  //     M*φ = F*φ + Q
  //
  // where:
  //   M = material-specific total interaction matrix
  //   F = fission production matrix (ν*Σ_f)
  //   Q = external source (computed by this function)
  //   φ = scalar flux (computed by CMFD solver)
  //
  // The external source Q is obtained by sampling the external source
  // distribution and binning onto the CMFD mesh. This projected source
  // is then passed to the CMFD solver as the RHS term.
  //
  // After the solver computes φ, the subcritical multiplication factor is:
  //
  //     k_eff = (total fission production from φ) / (total absorption of Q)
  //
  // This measures how many fissions are induced per neutron removed
  // (captured or leaking) from the external source.

  // Create flat array for accumulation
  int64_t total_bins = static_cast<int64_t>(nx) * ny * nz * ng;
  auto src_flat =
    xt::xtensor<double, 1>::from_shape({static_cast<std::size_t>(total_bins)});
  src_flat.fill(0.0);

  // Initialize random seed for reproducibility
  uint64_t seed = 12345 + mpi::rank * 1000;

  // Sample external source particles and accumulate weights in mesh bins
  for (int64_t i = 0; i < n_sample_particles; ++i) {
    SourceSite site = sample_external_source(&seed);

    // Bin to CMFD mesh
    int bin = bin_source_site_to_cmfd(site, mesh, egrid, ng);

    if (bin >= 0 && bin < total_bins) {
      src_flat(bin) += site.wgt;
    }
  }

#ifdef OPENMC_MPI
  // Reduce across all ranks to get global source distribution
  std::vector<double> src_reduced(total_bins);
  MPI_Reduce(src_flat.data(), src_reduced.data(), total_bins, MPI_DOUBLE,
    MPI_SUM, 0, mpi::intracomm);

  if (mpi::master) {
    std::copy(src_reduced.begin(), src_reduced.end(), src_flat.begin());
  }

  // Broadcast result to all ranks
  MPI_Bcast(src_flat.data(), total_bins, MPI_DOUBLE, 0, mpi::intracomm);
#endif

  // Normalize to unit integral
  // Q is the normalized source; the solver uses: M*φ = F*φ + Q
  double total = xt::sum(src_flat)();
  if (total > 0.0) {
    src_flat /= total;
  } else {
    warning(
      "External source projection resulted in zero total weight. "
      "Check that source particles are being generated inside CMFD mesh.");
  }

  // Reshape from flat to (nx, ny, nz, ng)
  std::vector<std::size_t> shape = {static_cast<std::size_t>(nx),
    static_cast<std::size_t>(ny), static_cast<std::size_t>(nz),
    static_cast<std::size_t>(ng)};
  auto src_reshaped = xt::reshape_view(src_flat, shape);

  // Print the projected source distribution for debugging
  if (mpi::master) {
    std::cout << "Projected external source distribution (normalized):\n";
    for (int i = 0; i < nx; ++i) {
      for (int j = 0; j < ny; ++j) {
        for (int k = 0; k < nz; ++k) {
          for (int g = 0; g < ng; ++g) {
            double value = src_reshaped(i, j, k, g);
            if (value > 0.0) {
              std::cout << fmt::format(
                "Mesh bin ({}, {}, {}, {}): {:.6e}\n", i, j, k, g, value);
            }
          }
        }
      }
    }
  }

  return xt::xtensor<double, 4>(src_reshaped);
}

} // namespace cmfd
} // namespace openmc