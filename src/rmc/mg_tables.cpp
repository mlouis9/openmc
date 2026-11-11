#include "openmc/rmc/mg_tables.h"

#include <unordered_map>

#include "openmc/error.h"
#include "openmc/hdf5_interface.h"
#include "openmc/material.h"
#include "openmc/random_ray/mg_data.h"
#include "openmc/vector.h"

namespace openmc {
namespace rmc {

void load_mg_tables(const std::string& field_file, RandomRayMGData& out)
{
  hid_t file = file_open(field_file, 'r');
  int version;
  read_attribute(file, "version", version);
  if (version < 2) {
    fatal_error(fmt::format("RMC field file {} has version {}; random ray in "
                            "a continuous-energy process needs version 2 or "
                            "later. Rerun the multigroup random ray solve.",
      field_file, version));
  }
  int G;
  read_attribute(file, "n_groups", G);
  vector<double> edges;
  read_dataset(file, "group_edges", edges);

  hid_t grp = open_group(file, "materials");
  vector<int32_t> ids, in_model;
  vector<double> sigma_t, sigma_s, nu_sigma_f, sigma_f, chi, kappa_fission;
  read_dataset(grp, "material_id", ids);
  read_dataset(grp, "in_model", in_model);
  read_dataset(grp, "sigma_t", sigma_t);
  read_dataset(grp, "sigma_s", sigma_s);
  read_dataset(grp, "nu_sigma_f", nu_sigma_f);
  read_dataset(grp, "sigma_f", sigma_f);
  read_dataset(grp, "chi", chi);
  read_dataset(grp, "kappa_fission", kappa_fission);
  close_group(grp);
  file_close(file);

  const int64_t M_file = ids.size();
  if (static_cast<int64_t>(edges.size()) != G + 1 ||
      static_cast<int64_t>(sigma_t.size()) != M_file * G ||
      static_cast<int64_t>(sigma_s.size()) != M_file * G * G) {
    fatal_error(fmt::format(
      "RMC field file {} has inconsistent table sizes.", field_file));
  }

  // File row of each material ID
  std::unordered_map<int32_t, int64_t> row_of_id;
  for (int64_t r = 0; r < M_file; ++r)
    row_of_id[ids[r]] = r;

  out.clear();
  out.n_groups = G;
  out.n_materials = model::materials.size();
  out.n_temperatures = 1;
  out.rev_energy_bins = edges;
  // Midpoints, by group index (group 0 has the highest energy), computed as
  // MgxsInterface does
  for (int g = 0; g < G; ++g) {
    out.energy_bin_avg.push_back(0.5 * (edges[G - g] + edges[G - g - 1]));
  }

  for (int m = 0; m < out.n_materials; ++m) {
    auto it = row_of_id.find(model::materials[m]->id());
    bool present = it != row_of_id.end() && in_model[it->second];
    out.exists_in_model.push_back(present);
    out.is_isotropic.push_back(1);
    // One temperature point; its value is never used to choose a table
    out.kTs.push_back(present ? vector<double> {0.0} : vector<double> {});

    int64_t fo = present ? it->second * G : 0; // file offset of the row
    for (int g_out = 0; g_out < G; ++g_out) {
      int ga_out = G - 1 - g_out; // ascending-energy index in the file
      out.sigma_t.push_back(present ? sigma_t[fo + ga_out] : 0.0);
      out.nu_sigma_f.push_back(present ? nu_sigma_f[fo + ga_out] : 0.0);
      out.sigma_f.push_back(present ? sigma_f[fo + ga_out] : 0.0);
      out.chi.push_back(present ? chi[fo + ga_out] : 0.0);
      out.kappa_fission.push_back(present ? kappa_fission[fo + ga_out] : 0.0);
      // The file stores sigma_s as [g_in][g_out] in ascending energy; the
      // solver stores it as [g_out][g_in] in descending energy.
      for (int g_in = 0; g_in < G; ++g_in) {
        int ga_in = G - 1 - g_in;
        out.sigma_s.push_back(
          present ? sigma_s[fo * G + ga_in * G + ga_out] : 0.0);
      }
    }
  }

  // Every material with tables must exist in this model
  for (int32_t id : ids) {
    if (model::material_map.find(id) == model::material_map.end()) {
      fatal_error(fmt::format("RMC field file {} has tables for material {}, "
                              "which is not in this model.",
        field_file, id));
    }
  }
}

} // namespace rmc
} // namespace openmc
