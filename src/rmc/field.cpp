#include "openmc/rmc/field.h"

#include <algorithm>
#include <cmath>
#include <unordered_map>

#include <fmt/format.h>

#include "openmc/cell.h"
#include "openmc/constants.h"
#include "openmc/error.h"
#include "openmc/hdf5_interface.h"
#include "openmc/material.h"
#include "openmc/mesh.h"
#include "openmc/random_lcg.h"
#include "openmc/random_ray/source_region.h" // hash_combine

namespace openmc {
namespace rmc {

namespace {

//! Read a whole 1D dataset. OpenMC's read_dataset() sizes the vector from
//! the first dimension only, which is why every array in these files is flat.
template<typename T>
vector<T> read_vec(hid_t loc, const char* name)
{
  vector<T> v;
  read_dataset(loc, name, v);
  return v;
}

} // namespace

size_t FrozenField::KeyHash::operator()(const Key& k) const
{
  size_t seed = 0;
  hash_combine(seed, k.cell);
  hash_combine(seed, k.instance);
  hash_combine(seed, k.mesh_bin);
  return seed;
}

size_t FrozenField::BaseHash::operator()(const Base& b) const
{
  size_t seed = 0;
  hash_combine(seed, b.cell);
  hash_combine(seed, b.instance);
  return seed;
}

FrozenField::FrozenField(const std::string& field_file,
  const std::string& lift_file, const std::string& volume_file)
{
  // ---- Field file: keys, materials, emission by tag ------------------------
  hid_t ff = file_open(field_file, 'r');
  int64_t R;
  read_attribute(ff, "n_groups", n_groups_);
  read_attribute(ff, "n_regions", R);
  group_edges_ = read_vec<double>(ff, "group_edges");
  if (!object_exists(ff, "domain_lower_left")) {
    fatal_error(fmt::format("RMC field: {} has no ray source box; rerun the "
                            "multigroup random ray solve (field version 3).",
      field_file));
  }
  auto box_ll = read_vec<double>(ff, "domain_lower_left");
  auto box_ur = read_vec<double>(ff, "domain_upper_right");
  box_lower_left_ = {box_ll[0], box_ll[1], box_ll[2]};
  box_upper_right_ = {box_ur[0], box_ur[1], box_ur[2]};
  hid_t rg = open_group(ff, "regions");
  auto cell_id = read_vec<int32_t>(rg, "cell_id");
  auto instance = read_vec<int32_t>(rg, "instance");
  auto mesh_id = read_vec<int32_t>(rg, "mesh_id");
  auto mesh_bin = read_vec<int64_t>(rg, "mesh_bin");
  auto material_id = read_vec<int32_t>(rg, "material_id");
  auto Q_s = read_vec<double>(rg, "Q_s");
  auto Q_f = read_vec<double>(rg, "Q_f");
  vector<double> phi;
  if (object_exists(rg, "phi"))
    phi = read_vec<double>(rg, "phi");
  close_group(rg);
  file_close(ff);

  const int G = n_groups_;
  if (static_cast<int64_t>(Q_s.size()) != R * G ||
      static_cast<int64_t>(Q_f.size()) != R * G ||
      static_cast<int64_t>(cell_id.size()) != R) {
    fatal_error("RMC field: array lengths do not match n_regions * n_groups.");
  }

  // ---- Lift file: within-group shapes per material --------------------------
  hid_t lf = file_open(lift_file, 'r');
  fine_edges_ = read_vec<double>(lf, "fine_edges");
  auto lift_ids = read_vec<int32_t>(lf, "material_ids");
  auto h_s = read_vec<double>(lf, "h_s");
  auto h_f = read_vec<double>(lf, "h_f");
  if (object_exists(lf, "h_phi"))
    lift_phi_ = read_vec<double>(lf, "h_phi");
  file_close(lf);
  field_file_ = field_file;
  n_fine_ = static_cast<int>(fine_edges_.size()) - 1;
  n_lift_materials_ = static_cast<int>(lift_ids.size());
  if (static_cast<int64_t>(h_s.size()) !=
      static_cast<int64_t>(n_lift_materials_) * n_fine_) {
    fatal_error("RMC field: lift arrays do not match materials * fine bins.");
  }
  lift_.resize(2 * h_s.size());
  std::copy(h_s.begin(), h_s.end(), lift_.begin());
  std::copy(h_f.begin(), h_f.end(), lift_.begin() + h_s.size());

  std::unordered_map<int32_t, int32_t> row_of_id;
  for (int32_t i = 0; i < n_lift_materials_; ++i)
    row_of_id[lift_ids[i]] = i;

  // ---- Volume file: normalization -------------------------------------------
  // An empty path loads the field unnormalized (scale 1, masses NaN). That is
  // how region volumes are computed in the first place.
  double scale = 1.0;
  vector<double> V;
  if (!volume_file.empty()) {
    hid_t vf = file_open(volume_file, 'r');
    read_attribute(vf, "scale", scale);
    V = read_vec<double>(vf, "V_A");
    file_close(vf);
    if (static_cast<int64_t>(V.size()) != R)
      fatal_error("RMC field: volume file does not match the field file.");
  }

  // ---- Group edges on the fine grid (for drawing from the shapes) ----------
  // The lowest group may start below the lift grid (often at 0); the lifts
  // are zero there, so the group starts at the grid's first bin
  group_fine_.assign(G + 1, -1);
  for (int g = 0; g <= G; ++g) {
    if (group_edges_[g] <= fine_edges_.front()) {
      group_fine_[g] = 0;
      continue;
    }
    auto it =
      std::lower_bound(fine_edges_.begin(), fine_edges_.end(), group_edges_[g]);
    if (it != fine_edges_.end() &&
        std::abs(*it - group_edges_[g]) <= 1e-12 * group_edges_[g])
      group_fine_[g] = static_cast<int>(it - fine_edges_.begin());
  }

  // ---- Assemble -------------------------------------------------------------
  volume_ = V;
  if (static_cast<int64_t>(phi.size()) == R * G) {
    flux_ = phi;
    for (double& x : flux_)
      x *= scale;
  }
  region_material_.assign(R, MATERIAL_VOID);
  region_cell_ = cell_id;
  region_instance_ = instance;
  region_bin_ = mesh_bin;
  lift_row_.assign(R, -1);
  region_mesh_.assign(R, -1);
  emission_.assign(2 * R * G, 0.0);
  for (int64_t r = 0; r < R; ++r) {
    key_to_region_[{cell_id[r], instance[r], mesh_bin[r]}] = r;

    // Mesh of the region's cell instance. Every region of one instance must
    // name the same mesh, since random ray applies at most one per instance.
    int32_t mesh_idx = -1;
    if (mesh_id[r] >= 0) {
      auto it = model::mesh_map.find(mesh_id[r]);
      if (it == model::mesh_map.end()) {
        fatal_error(
          fmt::format("RMC field: source-region mesh {} is not in this model.",
            mesh_id[r]));
      }
      mesh_idx = it->second;
    }
    auto [it_base, inserted] =
      mesh_of_base_.emplace(Base {cell_id[r], instance[r]}, mesh_idx);
    if (!inserted && it_base->second != mesh_idx) {
      fatal_error(fmt::format("RMC field: cell {} instance {} has regions on "
                              "different source-region meshes.",
        cell_id[r], instance[r]));
    }
    if (mesh_idx >= 0 && std::find(mesh_indices_.begin(), mesh_indices_.end(),
                           mesh_idx) == mesh_indices_.end()) {
      mesh_indices_.push_back(mesh_idx);
    }
    region_mesh_[r] = (mesh_bin[r] >= 0) ? mesh_idx : -1;

    if (material_id[r] >= 0) {
      auto it = row_of_id.find(material_id[r]);
      if (it == row_of_id.end()) {
        fatal_error(
          fmt::format("RMC field: no lift for material {}.", material_id[r]));
      }
      lift_row_[r] = it->second;
      auto im = model::material_map.find(material_id[r]);
      if (im != model::material_map.end())
        region_material_[r] = im->second;
    }

    for (int g = 0; g < G; ++g) {
      emission_[(0 * R + r) * G + g] = scale * Q_s[r * G + g];
      emission_[(1 * R + r) * G + g] = scale * Q_f[r * G + g];
      if (!V.empty()) {
        mass_[0] += V[r] * scale * Q_s[r * G + g];
        mass_[1] += V[r] * scale * Q_f[r * G + g];
      }
    }
  }
  if (V.empty())
    mass_[0] = mass_[1] = std::nan("");
  if (key_to_region_.size() != static_cast<size_t>(R))
    fatal_error("RMC field: duplicate region keys.");
}

int32_t FrozenField::mesh_of(int32_t cell_id, int32_t instance) const
{
  auto it = mesh_of_base_.find(Base {cell_id, instance});
  return (it == mesh_of_base_.end()) ? -2 : it->second;
}

int64_t FrozenField::region(
  int32_t cell_id, int32_t instance, int64_t mesh_bin) const
{
  auto it = key_to_region_.find({cell_id, instance, mesh_bin});
  return (it == key_to_region_.end()) ? -1 : it->second;
}

int64_t FrozenField::locate(const GeometryState& p) const
{
  // The same two steps as random ray's lookup_source_region_key(): the cell
  // instance decides which mesh (if any) subdivides it, and that mesh gives
  // the bin.
  Base base {model::cells[p.lowest_coord().cell()]->id_, p.cell_instance()};
  auto b = mesh_of_base_.find(base);
  if (b == mesh_of_base_.end())
    return -1; // a cell instance random ray never reached
  int64_t bin = (b->second >= 0) ? model::meshes[b->second]->get_bin(p.r()) : 0;
  auto it = key_to_region_.find({base.cell, base.instance, bin});
  return (it == key_to_region_.end()) ? -1 : it->second;
}

namespace {
//! Index i with edges[i] <= x < edges[i+1], or -1 outside [front, back).
int bin_index(const vector<double>& edges, double x)
{
  if (!(x >= edges.front() && x < edges.back()))
    return -1;
  return static_cast<int>(
           std::upper_bound(edges.begin(), edges.end(), x) - edges.begin()) -
         1;
}
} // namespace

double FrozenField::evaluate(int64_t region, Tag t, double E) const
{
  if (region < 0 || lift_row_[region] < 0)
    return 0.0;
  int g = bin_index(group_edges_, E);
  int i = bin_index(fine_edges_, E);
  if (g < 0 || i < 0)
    return 0.0;
  const int64_t ti = static_cast<int64_t>(t);
  const int64_t R = n_regions();
  double Q = emission_[(ti * R + region) * n_groups_ + g];
  double h = lift_[(ti * n_lift_materials_ + lift_row_[region]) * n_fine_ + i];
  return Q * h / (4.0 * PI);
}

int FrozenField::group(double E) const
{
  return bin_index(group_edges_, E);
}

double FrozenField::shape(Shape k, int64_t region, double E) const
{
  if (region < 0 || lift_row_[region] < 0)
    return 0.0;
  int i = bin_index(fine_edges_, E);
  if (i < 0)
    return 0.0;
  const int64_t row = lift_row_[region];
  if (k == Shape::phi) {
    if (lift_phi_.empty())
      fatal_error("RMC field: the lift file has no flux shape h_phi.");
    return lift_phi_[row * n_fine_ + i];
  }
  return lift_[(static_cast<int64_t>(k) * n_lift_materials_ + row) * n_fine_ +
               i];
}

namespace {
//! Mass of a shape over the fine bins [i0, i1)
double shape_mass(const double* h, const vector<double>& edges, int i0, int i1)
{
  double m = 0.0;
  for (int i = i0; i < i1; ++i)
    m += h[i] * (edges[i + 1] - edges[i]);
  return m;
}
} // namespace

double FrozenField::sample_shape(
  Shape k, int64_t region, int g, uint64_t* seed, double& density) const
{
  const double lo = std::max(group_edges_[g], fine_edges_.front());
  const double hi = group_edges_[g + 1];
  int i0 = group_fine_[g], i1 = group_fine_[g + 1];
  if (i0 < 0 || i1 < 0)
    fatal_error("RMC field: group edges are not on the lift grid.");
  const double* h = nullptr;
  if (region >= 0 && lift_row_[region] >= 0) {
    const int64_t row = lift_row_[region];
    h =
      (k == Shape::phi)
        ? &lift_phi_[row * n_fine_]
        : &lift_[(static_cast<int64_t>(k) * n_lift_materials_ + row) * n_fine_];
  }
  double m = h ? shape_mass(h, fine_edges_, i0, i1) : 0.0;
  if (!(m > 0.0)) {
    double E = lo * std::exp(std::log(hi / lo) * prn(seed));
    density = 1.0 / (E * std::log(hi / lo));
    return E;
  }
  double cut = prn(seed) * m, cum = 0.0;
  int i = i0;
  for (; i < i1 - 1; ++i) {
    cum += h[i] * (fine_edges_[i + 1] - fine_edges_[i]);
    if (cum > cut)
      break;
  }
  double E = fine_edges_[i] + prn(seed) * (fine_edges_[i + 1] - fine_edges_[i]);
  density = h[i] / m;
  return E;
}

double FrozenField::shape_draw_density(
  Shape k, int64_t region, int g, double E) const
{
  const double lo = std::max(group_edges_[g], fine_edges_.front());
  const double hi = group_edges_[g + 1];
  if (!(E >= lo && E < hi))
    return 0.0;
  int i0 = group_fine_[g], i1 = group_fine_[g + 1];
  if (i0 < 0 || i1 < 0)
    fatal_error("RMC field: group edges are not on the lift grid.");
  const double* h = nullptr;
  if (region >= 0 && lift_row_[region] >= 0) {
    const int64_t row = lift_row_[region];
    h =
      (k == Shape::phi)
        ? &lift_phi_[row * n_fine_]
        : &lift_[(static_cast<int64_t>(k) * n_lift_materials_ + row) * n_fine_];
  }
  double m = h ? shape_mass(h, fine_edges_, i0, i1) : 0.0;
  if (!(m > 0.0))
    return 1.0 / (E * std::log(hi / lo));
  return h[bin_index(fine_edges_, E)] / m;
}

} // namespace rmc
} // namespace openmc
