#include "openmc/rmc/forward.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "openmc/cell.h"
#include "openmc/geometry.h"
#include "openmc/material.h"
#include "openmc/nuclide.h"
#include "openmc/physics.h"
#include "openmc/random_lcg.h"
#include "openmc/rmc/channels.h"
#include "openmc/rmc/correction.h"
#include "openmc/rmc/production.h"
#include "openmc/rmc/residual.h"
#include "openmc/rmc/tracer.h"

namespace openmc {
namespace rmc {
namespace {

Direction isotropic(uint64_t* seed)
{
  double mu = 2.0 * prn(seed) - 1.0;
  double phi = 2.0 * PI * prn(seed);
  double st = std::sqrt(std::max(0.0, 1.0 - mu * mu));
  return {st * std::cos(phi), st * std::sin(phi), mu};
}

int draw(const vector<double>& weights, uint64_t* seed)
{
  double sum = 0.0;
  for (double x : weights)
    sum += x;
  if (!(sum > 0.0))
    return -1;
  double x = prn(seed) * sum;
  for (int i = 0; i < static_cast<int>(weights.size()); ++i) {
    x -= weights[i];
    if (x < 0.0)
      return i;
  }
  return weights.size() - 1;
}

struct HomogeneousPath {
  const FrozenField& field;
  CharacteristicTracer tracer;
  SegmentList segments;
  int32_t material;
  double sqrtkT, dm;

  explicit HomogeneousPath(const FrozenField& f) : field(f), tracer(f)
  {
    Position x = 0.5 * (f.box_lower_left() + f.box_upper_right());
    GeometryState p;
    p.init_from_r_u(x, {0.0, 0.0, 1.0});
    if (!exhaustive_find_cell(p) || f.locate(p) != 0)
      throw std::runtime_error("Homogeneous forward source box is not in region 0.");
    const auto& c = *model::cells[p.lowest_coord().cell()];
    material = c.material(p.cell_instance());
    sqrtkT = c.sqrtkT(p.cell_instance());
    dm = c.density_mult(p.cell_instance());
  }

  Position birth(uint64_t* seed)
  {
    Position x;
    auto lo = field.box_lower_left(), hi = field.box_upper_right();
    for (int j = 0; j < 3; ++j)
      x[j] = lo[j] + (hi[j] - lo[j]) * prn(seed);
    return x;
  }

  Position collision(Position x, Direction& u, double distance)
  {
    tracer.begin(x, u);
    for (;;) {
      tracer.extend(32, segments);
      for (int j = 0; j < segments.size(); ++j) {
        if (segments.material[j] != material || segments.region[j] != 0 ||
            segments.sqrtkT[j] != sqrtkT || segments.density_mult[j] != dm)
          throw std::runtime_error("Forward homogeneous kernel encountered a different region or material.");
        if (distance < segments.length[j]) {
          u = segments.direction[j];
          return segments.start[j] + distance * u;
        }
        distance -= segments.length[j];
      }
      if (segments.ends_in_vacuum || segments.size() == 0)
        throw std::runtime_error("Forward homogeneous kernel requires a fully reflective domain.");
    }
  }
};

double elastic_total(XsProbe& probe, const Material& mat, double dm)
{
  double sum = 0.0;
  for (int j = 0; j < mat.nuclide_.size(); ++j)
    sum += mat.atom_density(j, dm) * probe.micro_elastic(mat.nuclide_[j]);
  return sum;
}

int choose_nuclide(XsProbe& probe, const Material& mat, double dm, Tag tag,
  uint64_t* seed)
{
  vector<double> w;
  for (int j = 0; j < mat.nuclide_.size(); ++j) {
    int nuclide = mat.nuclide_[j];
    double xs = tag == Tag::s ? probe.micro_elastic(nuclide)
                             : probe.particle().neutron_xs(nuclide).nu_fission;
    w.push_back(mat.atom_density(j, dm) * xs);
  }
  int j = draw(w, seed);
  return j < 0 ? -1 : mat.nuclide_[j];
}

double macro_chi(XsProbe& probe, const Material& mat, double dm, double E)
{
  double sum = 0.0;
  for (int j = 0; j < mat.nuclide_.size(); ++j) {
    int nuclide = mat.nuclide_[j];
    double nf = probe.particle().neutron_xs(nuclide).nu_fission;
    if (nf > 0.0)
      sum += mat.atom_density(j, dm) * nf * fission_chi(nuclide, probe, E);
  }
  return sum;
}

} // namespace

void forward_homogeneous(const FrozenField& field, const MGTables& mg,
  Tag tag, int64_t n, const double a[2], double b, double k, uint64_t seed,
  SignedBank& out, double diagnostics[5], int part)
{
  if (field.n_regions() != 1 || !field.has_volumes())
    throw std::runtime_error("Forward homogeneous kernel needs one region with an exact volume.");
  if (n <= 0 || !(k > 0.0) || part < 0 || part > 4)
    throw std::runtime_error("Invalid forward homogeneous sample count, eigenvalue, or part.");
  std::fill(diagnostics, diagnostics + 5, 0.0);
  auto lo = field.box_lower_left(), hi = field.box_upper_right();
  double V = (hi.x - lo.x) * (hi.y - lo.y) * (hi.z - lo.z);
  if (std::abs(field.volume(0) - V) > 1e-10 * V)
    throw std::runtime_error("Homogeneous field volume must equal the source box volume.");
  HomogeneousPath path(field);
  const Material& mat = *model::materials[path.material];
  ChannelRegistry registry;
  for (int nuclide : mat.nuclide_) {
    const auto& nuc = *data::nuclides[nuclide];
    if (nuc.awr_ <= 1.0 || nuc.resonant_ || !nuc.index_inelastic_scatter_.empty())
      throw std::runtime_error("Forward homogeneous kernel supports stationary elastic targets and fission only.");
  }
  const int G = field.n_groups();
  const auto shape = tag == Tag::s ? FrozenField::Shape::s : FrozenField::Shape::f;
  XsProbe probe;
  vector<double> source(2 * G), row(G), low(G, 0.0);
  double source_mass = 0.0;
  for (int y = 0; y < 2; ++y) {
    for (int g = 0; g < G; ++g) {
      double q = a[y] * field.emission(0, static_cast<Tag>(y), g) * V;
      source[y * G + g] = std::abs(q);
      source_mass += std::abs(q);
      double st = mg.sigma_t(path.material, path.sqrtkT, g);
      if (!(st > 0.0))
        throw std::runtime_error("Homogeneous MG total cross sections must be positive.");
      for (int h = 0; h < G; ++h) {
        double production = tag == Tag::s
                              ? mg.nu_sigma_s(path.material, path.sqrtkT, g, h)
                              : mg.nu_sigma_f(path.material, path.sqrtkT, g) *
                                  mg.chi(path.material, path.sqrtkT, h) / k;
        low[h] += q * production / st;
      }
    }
  }
  for (int h = 0; h < G; ++h)
    low[h] -= b * field.emission(0, tag, h) * V;

  if (source_mass > 0.0 && part != 4) {
    for (int64_t i = 0; i < n; ++i) {
      uint64_t s = future_seed(static_cast<uint64_t>(i) * 1000003ULL + 43, seed);
      int yg = draw(source, &s), y = yg / G, g = yg % G;
      auto source_shape = y == 0 ? FrozenField::Shape::s : FrozenField::Shape::f;
      double dummy;
      double Ei = field.sample_shape(source_shape, 0, g, &s, dummy);
      double st = probe.total(path.material, Ei, path.sqrtkT, path.dm);
      double stm = mg.sigma_t(path.material, path.sqrtkT, g);
      double mg_prod = 0.0;
      for (int h = 0; h < G; ++h) {
        row[h] = tag == Tag::s ? mg.nu_sigma_s(path.material, path.sqrtkT, g, h)
                              : mg.chi(path.material, path.sqrtkT, h);
        mg_prod += row[h];
      }
      if (tag == Tag::f)
        mg_prod = mg.nu_sigma_f(path.material, path.sqrtkT, g);
      bool ce_component = prn(&s) < 0.5;
      double distance = -std::log(prn(&s)) / (ce_component ? st : stm);
      Direction ui = isotropic(&s);
      Position x = path.collision(path.birth(&s), ui, distance);
      Direction uo;
      double Eo;
      if (ce_component) {
        int nuclide = choose_nuclide(probe, mat, path.dm, tag, &s);
        if (nuclide < 0) {
          diagnostics[4] += 1.0;
          continue;
        }
        Particle& p = probe.scratch();
        p.u() = ui;
        p.E() = Ei;
        p.wgt() = 1.0;
        p.stream() = STREAM_TRACKING;
        p.seeds(STREAM_TRACKING) = s;
        if (tag == Tag::s) {
          elastic_scatter(nuclide, *data::nuclides[nuclide]->reactions_[0],
            ChannelRegistry::elastic_kT(nuclide, probe), p);
          Eo = p.E();
          uo = p.u();
        } else {
          SourceSite site;
          const auto& rx = sample_fission(nuclide, p);
          sample_fission_neutron(nuclide, rx, &site, p);
          Eo = site.E;
          uo = site.u;
        }
        s = p.seeds(STREAM_TRACKING);
      } else {
        int h = mg_prod > 0.0 ? draw(row, &s) : -1;
        if (h < 0) {
          diagnostics[4] += 1.0;
          continue;
        }
        Eo = field.sample_shape(shape, 0, h, &s, dummy);
        uo = isotropic(&s);
      }

      // The elastic law is compared after integrating out incident energy.
      // Every nuclide contributes its own image, even if another one sampled Eo.
      double ce = 0.0, qce = 0.0;
      if (tag == Tag::s) {
        double mu = std::clamp(ui.dot(uo), -1.0, 1.0);
        for (int j = 0; j < mat.nuclide_.size(); ++j) {
          int nuclide = mat.nuclide_[j];
          auto image = elastic_stationary(nuclide,
            ChannelRegistry::elastic_kT(nuclide, probe), Eo, mu);
          if (!(image.density > 0.0) || field.group(image.E_in) != g)
            continue;
          double h = field.shape(source_shape, 0, image.E_in);
          if (!(h > 0.0))
            continue;
          double sigt = probe.total(path.material, image.E_in, path.sqrtkT, path.dm);
          double sigel = mat.atom_density(j, path.dm) * probe.micro_elastic(nuclide);
          double total_el = elastic_total(probe, mat, path.dm);
          double term = h * std::exp(-sigt * distance) * sigel * image.density / (2.0 * PI);
          ce += term;
          if (total_el > 0.0)
            qce += term * sigt / total_el;
        }
      } else {
        // Incident energy remains a latent variable for ordinary densities.
        probe.total(path.material, Ei, path.sqrtkT, path.dm);
        double nf = probe.particle().macro_xs().nu_fission;
        double kernel = macro_chi(probe, mat, path.dm, Eo) / (4.0 * PI);
        ce = std::exp(-st * distance) * kernel / k;
        if (nf > 0.0)
          qce = std::exp(-st * distance) * st * kernel / nf;
      }
      int go = field.group(Eo);
      double mlaw = go < 0 ? 0.0 : field.shape(shape, 0, Eo) *
        (tag == Tag::s ? mg.nu_sigma_s(path.material, path.sqrtkT, g, go)
                      : mg_prod * mg.chi(path.material, path.sqrtkT, go)) /
        (4.0 * PI);
      double low_value = std::exp(-stm * distance) * mlaw;
      double qlow = mg_prod > 0.0 ? low_value * stm / mg_prod : 0.0;
      if (tag == Tag::f)
        low_value /= k;
      double proposal = 0.5 * (qce + qlow);
      if (!(proposal > 0.0) || !std::isfinite(proposal))
        throw std::runtime_error("Continuous forward sample has invalid mixture density.");
      double factor = std::copysign(source_mass / n, a[y]);
      diagnostics[0] += factor * ce / proposal;
      diagnostics[1] += factor * low_value / proposal;
      diagnostics[2] += std::abs(factor * (ce - low_value) / proposal);
      double value = part == 1 ? ce : part == 2 ? low_value : ce - low_value;
      double w = factor * value / proposal;
      if (w != 0.0)
        out.push_back({x, uo, Eo, w, tag});
    }
  }
  // Exact homogeneous low-order remainder, including arbitrary a,b,k.
  // This avoids assuming that an exported numerical field is a fixed point.
  if (part == 0 || part == 4) {
    vector<double> weights(G);
    double mass = 0.0;
    for (int h = 0; h < G; ++h) {
      weights[h] = std::abs(low[h]);
      mass += weights[h];
    }
    diagnostics[3] = mass;
    if (mass > 0.0) {
      for (int64_t i = 0; i < n; ++i) {
        uint64_t s = future_seed(static_cast<uint64_t>(i) * 1000033ULL + 97, seed);
        int h = draw(weights, &s);
        double density;
        double E = field.sample_shape(shape, 0, h, &s, density);
        auto x = path.birth(&s);
        auto u = isotropic(&s);
        out.push_back({x, u, E, std::copysign(mass / n, low[h]), tag});
      }
    }
  }
}

//==============================================================================
// Continuous correlated forward residual on any field (heterogeneous)
//==============================================================================

namespace {

struct PathSeg {
  double length;
  int32_t material;
  int64_t region;
  double sqrtkT, dm;
  Position start;
  Direction dir;
};

//! A forward path read lazily from the tracer, with the optical depth of a
//! given energy on demand.
struct ForwardPath {
  CharacteristicTracer tracer;
  SegmentList chunk;
  vector<PathSeg> seg;
  bool ended {false};

  explicit ForwardPath(const FrozenField& f) : tracer(f) {}

  void begin(Position x, Direction u)
  {
    tracer.begin(x, u);
    seg.clear();
    ended = false;
  }

  //! Make segment j available; false if the path ends before it
  bool ensure(size_t j)
  {
    while (seg.size() <= j && !ended) {
      tracer.extend(32, chunk);
      for (int i = 0; i < chunk.size(); ++i)
        seg.push_back({chunk.length[i], chunk.material[i], chunk.region[i],
          chunk.sqrtkT[i], chunk.density_mult[i], chunk.start[i],
          chunk.direction[i]});
      if (chunk.ends_in_vacuum || chunk.size() == 0)
        ended = true;
    }
    return seg.size() > j;
  }
};

//! Sigma_t of a few materials at one energy
struct TotalCache {
  int32_t mat[4] {-2, -2, -2, -2};
  double sqrtkT[4], dm[4], value[4];
  double E {-1.0};
  void reset(double e)
  {
    E = e;
    for (auto& m : mat)
      m = -2;
  }
  double get(XsProbe& probe, int32_t m, double kT, double d)
  {
    if (m == MATERIAL_VOID)
      return 0.0;
    for (int i = 0; i < 4; ++i)
      if (mat[i] == m && sqrtkT[i] == kT && dm[i] == d)
        return value[i];
    int i = 0;
    while (i < 3 && mat[i] != -2)
      ++i;
    mat[i] = m;
    sqrtkT[i] = kT;
    dm[i] = d;
    value[i] = probe.total(m, E, kT, d);
    return value[i];
  }
};

//! Optical depth to distance d along the path at energy E (CE)
double ce_depth(ForwardPath& path, double d, double E, XsProbe& probe, TotalCache& cache)
{
  cache.reset(E);
  double tau = 0.0, done = 0.0;
  for (size_t j = 0; done < d && path.ensure(j); ++j) {
    const auto& q = path.seg[j];
    double l = std::min(q.length, d - done);
    tau += cache.get(probe, q.material, q.sqrtkT, q.dm) * l;
    done += l;
  }
  return tau;
}

double scatter_production(XsProbe& probe, const ChannelRegistry& registry, int32_t m,
  double dm)
{
  const auto& mx = probe.particle().macro_xs();
  if (probe.E() < 1.0e6)
    return mx.total - mx.absorption;
  double prod = 0.0;
  for (const auto& c : registry.channels(m))
    if (c.kind != ChannelKind::fission)
      prod += ChannelRegistry::production(c, dm, probe);
  return prod;
}

double elastic_macro(XsProbe& probe, const Material& mat, double dm)
{
  double sum = 0.0;
  for (size_t j = 0; j < mat.nuclide_.size(); ++j)
    sum += mat.atom_density(j, dm) * probe.micro_elastic(mat.nuclide_[j]);
  return sum;
}

} // namespace

namespace {
int resolve_slot = 0;
}
ResolveLibrary& resolve_library(int slot)
{
  static ResolveLibrary lib[2];
  return lib[slot < 0 ? resolve_slot : slot];
}
void resolve_select(int slot)
{
  resolve_slot = slot;
}

namespace {
//! Flux per eV at E decoded from lane coefficients z of block b
double resolve_decode(const ResolveBlock& b, double E, const vector<double>& z)
{
  int j = static_cast<int>(std::upper_bound(b.edges.begin(), b.edges.end(), E) - b.edges.begin()) - 1;
  if (j < 0 || j >= b.n)
    return 0.0;
  if (b.U.empty()) {
    // band (subgroup) library: indicator lanes 1/sqrt(mass) on their bins
    if (b.band.empty())
      return 0.0;
    int k = b.band[j];
    return z[k] / std::sqrt(b.band_mass[k]) / E;
  }
  const double* u = &b.U[static_cast<size_t>(j) * b.L];
  double v = 0.0;
  for (int k = 0; k < b.L; ++k)
    v += u[k] * z[k];
  return v / E;
}
} // namespace

void forward_points(const FrozenField& field, const MGTables& mg, Tag tag, int part,
  int64_t n, const double a[2], double k, double lambda, uint64_t seed,
  ForwardPoints& out, double diagnostics[24])
{
  // The field's first-collision emission of tag `tag`, CE minus a
  // multigroup reconstruction, sampled as signed points (no bins). A source
  // neutron is drawn from the field (region and position by the emission
  // mass, source tag and group, energy from the source region's lift,
  // isotropic direction); a collision site on its path and an outgoing
  // state z = (path distance, E', Omega') are drawn from a mixture of the CE
  // and multigroup first-collision emissions; both densities are evaluated
  // at z, the CE elastic one with the incident energy integrated out through
  // the elastic images of every nuclide of the collision material.
  //
  // part 0: the density part, CE stationary elastic (tag s) or CE fission
  //   (tag f) against the multigroup emission times f (tag s: the elastic
  //   share of the CE scattering production of the material and group, any
  //   value in [0, 1] keeps the split exact; tag f: 1). w_ce and w_mg are
  //   the two terms of the signed weight (w = w_ce - w_mg).
  // part 1: CE non-elastic scattering production (inelastic, (n,xn)), drawn
  //   by OpenMC's scatter(), unpaired (w_ce only).
  // part 2: the multigroup scattering emission times 1 - f, unpaired (w_mg).
  // The multigroup reconstruction: Sigma_t,g, nu Sigma_s(g -> g') or
  // chi_g' nu Sigma_f,g, the collision region's lift within g', isotropic.
  // Fission emission includes 1 / k on both sides.
  const int64_t R = field.n_regions();
  const int G = field.n_groups();
  const auto& fe = field.fine_edges();
  std::fill(diagnostics, diagnostics + 24, 0.0);
  if (!field.has_shape(FrozenField::Shape::phi))
    throw std::runtime_error("forward_points needs the flux shape h_phi in the lift file.");
  out.clear();
  if (n <= 0)
    return;
  XsProbe probe;
  ChannelRegistry registry;
  const double sqrtkT0 = std::sqrt(K_BOLTZMANN * 293.6);
  const auto out_shape = tag == Tag::s ? FrozenField::Shape::s : FrozenField::Shape::f;

  // Source: region by emission mass, then (source tag, group)
  vector<double> wr(R, 0.0), qsum(R, 0.0);
  for (int64_t r = 0; r < R; ++r) {
    if (field.region_material(r) == MATERIAL_VOID)
      continue;
    for (int y = 0; y < 2; ++y)
      for (int g = 0; g < G; ++g)
        qsum[r] += std::abs(a[y]) * field.emission(r, static_cast<Tag>(y), g);
    wr[r] = qsum[r] * field.volume(r);
  }
  VoxelOptions vo;
  vo.mode = VoxelMode::tabulated;
  vo.weights = wr;
  VoxelProposal voxel(field, vo);

  // Elastic share f(g, m) of the CE scattering production, weighted with
  // the flux lift (or 1/E) of the first region of each material
  const int n_mat = model::materials.size();
  vector<double> f_el(static_cast<size_t>(n_mat) * G, 1.0);
  if (tag == Tag::s) {
    vector<int64_t> rep(n_mat, -1);
    for (int64_t r = 0; r < R; ++r) {
      int32_t m = field.region_material(r);
      if (m >= 0 && rep[m] < 0)
        rep[m] = r;
    }
    for (int32_t m = 0; m < n_mat; ++m) {
      if (rep[m] < 0)
        continue;
      const Material& mat = *model::materials[m];
      vector<double> num(G, 0.0), den(G, 0.0);
      for (size_t i = 0; i + 1 < fe.size(); ++i) {
        double Em = std::sqrt(std::max(fe[i], 1e-11) * fe[i + 1]);
        int g = field.group(Em);
        if (g < 0)
          continue;
        double wgt = field.has_shape(FrozenField::Shape::phi)
                       ? field.shape(FrozenField::Shape::phi, rep[m], Em)
                       : 1.0 / Em;
        wgt *= fe[i + 1] - fe[i];
        probe.total(m, Em, sqrtkT0, 1.0);
        num[g] += wgt * elastic_macro(probe, mat, 1.0);
        den[g] += wgt * scatter_production(probe, registry, m, 1.0);
      }
      for (int g = 0; g < G; ++g)
        f_el[static_cast<size_t>(m) * G + g] =
          den[g] > 0.0 ? std::clamp(num[g] / den[g], 0.0, 1.0) : 1.0;
    }
  }

  // P1 elastic transfer mubar(m, g, g') for the P1 multigroup partner
  // (part 4): incident energies from the flux lift of the material's first
  // region, nuclide by elastic cross section, OpenMC's elastic_scatter();
  // samples weighted by Sigma_el(E)
  vector<double> mubar(static_cast<size_t>(n_mat) * G * G, 0.0);
  if (part == 4 && tag == Tag::s) {
    vector<int64_t> rep(n_mat, -1);
    for (int64_t r = 0; r < R; ++r) {
      int32_t m = field.region_material(r);
      if (m >= 0 && rep[m] < 0)
        rep[m] = r;
    }
    vector<double> num(mubar.size(), 0.0), den(mubar.size(), 0.0);
    uint64_t sm = future_seed(7, seed);
    for (int32_t m = 0; m < n_mat; ++m) {
      if (rep[m] < 0)
        continue;
      const Material& mat = *model::materials[m];
      for (int g = 0; g < G; ++g)
        for (int i = 0; i < 4000; ++i) {
          double dd;
          double Ein = field.sample_shape(FrozenField::Shape::phi, rep[m], g, &sm, dd);
          probe.total(m, Ein, sqrtkT0, 1.0);
          Particle& p = probe.scratch();
          vector<double> wn;
          double el = 0.0;
          for (size_t j = 0; j < mat.nuclide_.size(); ++j) {
            wn.push_back(mat.atom_density(j, 1.0) * probe.micro_elastic(mat.nuclide_[j]));
            el += wn.back();
          }
          int jn = draw(wn, &sm);
          if (jn < 0)
            continue;
          int nuc = mat.nuclide_[jn];
          p.u() = {0.0, 0.0, 1.0};
          p.E() = Ein;
          p.E_last() = Ein;
          p.wgt() = 1.0;
          p.stream() = STREAM_TRACKING;
          p.seeds(STREAM_TRACKING) = sm;
          elastic_scatter(nuc, *data::nuclides[nuc]->reactions_[0],
            ChannelRegistry::elastic_kT(nuc, probe), p);
          sm = p.seeds(STREAM_TRACKING);
          int go = field.group(p.E());
          if (go < 0)
            continue;
          size_t key = (static_cast<size_t>(m) * G + g) * G + go;
          num[key] += el * p.u().z;
          den[key] += el;
        }
    }
    for (size_t i = 0; i < mubar.size(); ++i)
      mubar[i] = den[i] > 0.0 ? num[i] / den[i] : 0.0;
  }

  ForwardPath path(field);
  TotalCache cache;
  vector<double> mgrow(G);
  for (int64_t i = 0; i < n; ++i) {
    uint64_t s = future_seed(static_cast<uint64_t>(i) * 1000003ULL + 61, seed);
    int64_t r0;
    auto rd = voxel.sample(&s, r0);
    if (r0 < 0 || !(qsum[r0] > 0.0))
      continue;
    // (source tag, group) in the region
    double x = prn(&s) * qsum[r0];
    int y = 0, g = 0;
    for (int yy = 0, done = 0; yy < 2 && !done; ++yy)
      for (int gg = 0; gg < G; ++gg) {
        x -= std::abs(a[yy]) * field.emission(r0, static_cast<Tag>(yy), gg);
        y = yy;
        g = gg;
        if (x < 0.0) {
          done = 1;
          break;
        }
      }
    const auto src_shape = y == 0 ? FrozenField::Shape::s : FrozenField::Shape::f;
    double factor = std::copysign(qsum[r0] / rd.density / n, a[y]);
    double dummy;
    double Ei = field.sample_shape(src_shape, r0, g, &s, dummy);
    Direction u0 = isotropic(&s);
    path.begin(rd.value, u0);

    // Which component draws the collision and the outgoing state
    // part 3: the density part with the kinematic multigroup partner
    const bool kinematic = part == 3 || part == 4;
    const bool partner_diag = part == 4;
    const bool resolve = part >= 5 && part <= 8;
    const bool banded = part == 8 && tag == Tag::s;
    constexpr double lambda_p = 0.45;
    const int dpart = (kinematic || resolve) ? 0 : part;
    double u_comp = prn(&s);
    bool use_ce = dpart == 1 || (dpart == 0 && u_comp < lambda);
    // Third component (part 8): a band k of the proposal library by the
    // source's band masses a_k, the collision with the band's Sigma_t
    bool use_p = banded && !use_ce && u_comp < lambda + lambda_p;
    const ResolveLibrary& plib = resolve_library(1);
    // The source's band masses a_k = c_k sqrt(mass_k) (band library, slot 1)
    const vector<double>* src_p = nullptr;
    const vector<double>* cdf_p = nullptr;
    const ResolveBlock* b_p = nullptr;
    if (banded) {
      const int64_t key = ResolveLibrary::source_key(r0, y, g);
      auto it = plib.source.find(key);
      b_p = plib.get(field.region_material(r0), g);
      if (it != plib.source.end() && b_p && !b_p->band_mass.empty()) {
        src_p = &it->second;
        auto& cdf = plib.source_cdf[key];
        if (cdf.empty()) {
          double acc = 0.0;
          for (int k2 = 0; k2 < b_p->L; ++k2) {
            acc += std::max(0.0, (*src_p)[k2] * std::sqrt(b_p->band_mass[k2]));
            cdf.push_back(acc);
          }
        }
        cdf_p = &cdf;
      }
    }
    auto a_band = [&](int kb) {
      return src_p ? std::max(0.0, (*src_p)[kb] * std::sqrt(b_p->band_mass[kb])) : 0.0;
    };
    int k_p = -1;
    if (use_p) {
      if (!cdf_p || !(cdf_p->back() > 0.0)) {
        diagnostics[4] += 1.0;
        continue;
      }
      double x = prn(&s) * cdf_p->back();
      k_p = static_cast<int>(std::upper_bound(cdf_p->begin(), cdf_p->end(), x) - cdf_p->begin());
      k_p = std::min(k_p, b_p->L - 1);
    }
    auto band_sigma = [&](int32_t m, int kb) {
      const ResolveBlock* bb = plib.get(m, g);
      return bb ? bb->D[kb] : 0.0;
    };
    double tau_star = -std::log(prn(&s));
    // Walk to the collision with the chosen component's Sigma_t
    cache.reset(Ei);
    double tau = 0.0, dist = 0.0;
    size_t jc = 0;
    bool hit = false;
    for (size_t j = 0; path.ensure(j); ++j) {
      const auto& q = path.seg[j];
      double st = q.material == MATERIAL_VOID ? 0.0
                  : use_ce ? cache.get(probe, q.material, q.sqrtkT, q.dm)
                  : use_p  ? band_sigma(q.material, k_p)
                           : mg.sigma_t(q.material, q.sqrtkT, g);
      if (st * q.length >= tau_star - tau) {
        dist += (tau_star - tau) / st;
        jc = j;
        hit = true;
        break;
      }
      tau += st * q.length;
      dist += q.length;
    }
    if (!hit) {
      diagnostics[5] += 1.0;
      continue;
    }
    const PathSeg qc = path.seg[jc];
    double before = 0.0;
    for (size_t j = 0; j < jc; ++j)
      before += path.seg[j].length;
    Position xc = qc.start + (dist - before) * qc.dir;
    const int32_t mc = qc.material;
    const Material& mat = *model::materials[mc];
    // Multigroup optical depth to the collision (group g)
    double tau_mg = 0.0;
    for (size_t j = 0; j <= jc; ++j) {
      const auto& q = path.seg[j];
      double l = (j == jc) ? dist - before : q.length;
      if (q.material != MATERIAL_VOID)
        tau_mg += mg.sigma_t(q.material, q.sqrtkT, g) * l;
    }
    const double st_mg = mg.sigma_t(mc, qc.sqrtkT, g);

    double Eo = 0.0;
    Direction uo {0.0, 0.0, 1.0};
    double w_ce = 0.0, w_mg = 0.0;

    if (dpart == 1) {
      // CE non-elastic scattering, analog at Ei
      double st = probe.total(mc, Ei, qc.sqrtkT, qc.dm);
      double prod = scatter_production(probe, registry, mc, qc.dm);
      double el = elastic_macro(probe, mat, qc.dm);
      double nonel = prod - el;
      if (!(nonel > 0.0) || !(st > 0.0)) {
        diagnostics[4] += 1.0;
        continue;
      }
      // Nuclide and reaction among the non-elastic scatterings: draw by
      // scatter() until it is not elastic (rejection keeps the conditional
      // law exact)
      bool ok = false;
      for (int tries = 0; tries < 1000 && !ok; ++tries) {
        probe.total(mc, Ei, qc.sqrtkT, qc.dm);
        Particle& p = probe.scratch();
        double ts = 0.0;
        for (size_t j = 0; j < mat.nuclide_.size(); ++j) {
          const auto& nx = p.neutron_xs(mat.nuclide_[j]);
          ts += mat.atom_density(j, qc.dm) * (nx.total - nx.absorption);
        }
        double xx = prn(&s) * ts;
        int nuc = mat.nuclide_.back();
        for (size_t j = 0; j < mat.nuclide_.size(); ++j) {
          const auto& nx = p.neutron_xs(mat.nuclide_[j]);
          xx -= mat.atom_density(j, qc.dm) * (nx.total - nx.absorption);
          if (xx < 0.0) {
            nuc = mat.nuclide_[j];
            break;
          }
        }
        p.u() = qc.dir;
        p.E() = Ei;
        p.E_last() = Ei;
        p.wgt() = 1.0;
        p.stream() = STREAM_TRACKING;
        p.seeds(STREAM_TRACKING) = s;
        p.local_secondary_bank().clear();
        p.n_secondaries() = 0;
        scatter(p, nuc);
        s = p.seeds(STREAM_TRACKING);
        int mt = p.event_mt();
        double yld = p.wgt() * (1.0 + static_cast<double>(p.local_secondary_bank().size()));
        p.local_secondary_bank().clear();
        p.n_secondaries() = 0;
        if (mt == ELASTIC)
          continue;
        ok = true;
        Eo = p.E();
        uo = p.u();
        // Expected yield of the non-elastic channel: its production over
        // Sigma_t, with this sample's multiplicity relative to the mean
        // multiplicity folded into nonel (the draw is by cross section)
        double xs_nonel = 0.0;
        for (size_t j = 0; j < mat.nuclide_.size(); ++j) {
          const auto& nx = p.neutron_xs(mat.nuclide_[j]);
          xs_nonel += mat.atom_density(j, qc.dm) * (nx.total - nx.absorption);
        }
        xs_nonel -= el;
        w_ce = factor * xs_nonel / st * yld;
        out.add(nuc, mt, mc, field.group(Eo), w_ce);
      }
      if (!ok) {
        diagnostics[4] += 1.0;
        continue;
      }
    } else if (dpart == 2) {
      // Multigroup scattering times (1 - f), unpaired
      double out_sum = 0.0;
      for (int h = 0; h < G; ++h) {
        mgrow[h] = mg.nu_sigma_s(mc, qc.sqrtkT, g, h);
        out_sum += mgrow[h];
      }
      int h = draw(mgrow, &s);
      if (h < 0 || !(st_mg > 0.0)) {
        diagnostics[4] += 1.0;
        continue;
      }
      Eo = field.sample_shape(FrozenField::Shape::s, qc.region, h, &s, dummy);
      uo = isotropic(&s);
      w_mg = factor * out_sum / st_mg * (1.0 - f_el[static_cast<size_t>(mc) * G + g]);
      out.add(-1, 991, mc, field.group(Eo), w_mg);
    } else {
      // Density part: draw the outgoing state from the chosen component
      double mg_out = 0.0;
      for (int h = 0; h < G; ++h) {
        mgrow[h] = tag == Tag::s ? mg.nu_sigma_s(mc, qc.sqrtkT, g, h)
                                 : mg.chi(mc, qc.sqrtkT, h);
        mg_out += tag == Tag::s ? mgrow[h] : 0.0;
      }
      if (tag == Tag::f)
        mg_out = mg.nu_sigma_f(mc, qc.sqrtkT, g);
      double Ein_draw = Ei;
      if (use_p) {
        // incident energy uniform in lethargy over the band's bins
        const ResolveBlock* bb = plib.get(mc, g);
        if (!bb || bb->band_bins[k_p].empty()) {
          diagnostics[4] += 1.0;
          continue;
        }
        const auto& cdf = bb->band_cdf[k_p];
        double x = prn(&s) * cdf.back();
        int ib = static_cast<int>(std::upper_bound(cdf.begin(), cdf.end(), x) - cdf.begin());
        ib = std::min(ib, static_cast<int>(cdf.size()) - 1);
        int jb = bb->band_bins[k_p][ib];
        Ein_draw = bb->edges[jb] * std::pow(bb->edges[jb + 1] / bb->edges[jb], prn(&s));
      }
      if (use_ce || use_p) {
        const double Ei = Ein_draw;
        probe.total(mc, Ei, qc.sqrtkT, qc.dm);
        Particle& p = probe.scratch();
        vector<double> wn;
        for (size_t j = 0; j < mat.nuclide_.size(); ++j) {
          int nuc = mat.nuclide_[j];
          double xs = tag == Tag::s ? probe.micro_elastic(nuc) : p.neutron_xs(nuc).nu_fission;
          wn.push_back(mat.atom_density(j, qc.dm) * xs);
        }
        int jn = draw(wn, &s);
        if (jn < 0) {
          diagnostics[4] += 1.0;
          continue;
        }
        int nuc = mat.nuclide_[jn];
        p.u() = qc.dir;
        p.E() = Ei;
        p.E_last() = Ei;
        p.wgt() = 1.0;
        p.stream() = STREAM_TRACKING;
        p.seeds(STREAM_TRACKING) = s;
        if (tag == Tag::s) {
          elastic_scatter(nuc, *data::nuclides[nuc]->reactions_[0],
            ChannelRegistry::elastic_kT(nuc, probe), p);
          Eo = p.E();
          uo = p.u();
        } else {
          SourceSite site;
          const auto& rx = sample_fission(nuc, p);
          sample_fission_neutron(nuc, rx, &site, p);
          Eo = site.E;
          uo = site.u;
        }
        s = p.seeds(STREAM_TRACKING);
      } else {
        int h = mg_out > 0.0 ? draw(mgrow, &s) : -1;
        if (h < 0) {
          diagnostics[4] += 1.0;
          continue;
        }
        Eo = field.sample_shape(out_shape, qc.region, h, &s, dummy);
        uo = isotropic(&s);
      }
      // Densities at z = (dist, Eo, uo) given the source (x0, u0, y, g)
      double ce = 0.0, qce = 0.0;
      // Partner diagnostics (see forward.h): m1s, m1 = the CE kernel on the
      // multigroup-attenuated group flux with the source lift / the
      // collision region's flux lift; p0, p1 = the Legendre P0 / P0+P1
      // truncation of m1 in the incident-outgoing cosine; c0 = the
      // incident-energy average of m1 (fission)
      const double tm_c = std::exp(-tau_mg);
      double m1s = 0.0, m1 = 0.0, p0 = 0.0, p1 = 0.0, c0 = 0.0;
      vector<std::pair<int, double>> by_nuc; // CE terms by nuclide
      // RESOLVE partner: carry the source's lane coefficients along the path
      double rs = 0.0, rs1 = 0.0, rs3 = 0.0, rs7 = 0.0, q_band = 0.0;
      bool band_mode = false;
      const ResolveBlock* rb = nullptr;
      vector<double> zeta;
      if (resolve) {
        const auto& lib = resolve_library(0);
        auto it = lib.source.find(ResolveLibrary::source_key(r0, y, g));
        int cur = field.region_material(r0);
        const ResolveBlock* bs = lib.get(cur, g);
        if (it != lib.source.end() && bs && bs->U.empty() && !bs->band.empty()) {
          // band library: no lane propagation; each image's band is attenuated
          // on its own (band_flux below)
          band_mode = true;
          zeta = it->second;
          rb = lib.get(mc, g);
        } else if (it != lib.source.end()) {
          zeta = it->second;
          for (size_t j = 0; j <= jc && !zeta.empty(); ++j) {
            const auto& q = path.seg[j];
            if (q.material == MATERIAL_VOID)
              continue;
            if (q.material != cur) {
              auto ic = lib.cross.find(ResolveLibrary::cross_key(cur, q.material, g));
              const ResolveBlock* bt = lib.get(q.material, g);
              if (ic == lib.cross.end() || !bt) {
                zeta.clear();
                break;
              }
              vector<double> zn(bt->L, 0.0);
              const int Lf = static_cast<int>(zeta.size());
              for (int a = 0; a < bt->L; ++a)
                for (int b = 0; b < Lf; ++b)
                  zn[a] += ic->second[static_cast<size_t>(a) * Lf + b] * zeta[b];
              zeta.swap(zn);
              cur = q.material;
            }
            const ResolveBlock* bc = lib.get(cur, g);
            double l = (j == jc) ? dist - before : q.length;
            for (int k2 = 0; k2 < bc->L; ++k2)
              zeta[k2] *= std::exp(-bc->D[k2] * l);
          }
          if (!zeta.empty() && cur == mc)
            rb = lib.get(mc, g);
        }
      }
      // The partner's uncollided flux per eV at E (lanes or bands)
      auto partner_flux = [&](double E) {
        if (!rb)
          return 0.0;
        if (!band_mode)
          return resolve_decode(*rb, E, zeta);
        int jb = static_cast<int>(std::upper_bound(rb->edges.begin(), rb->edges.end(), E) - rb->edges.begin()) - 1;
        if (jb < 0 || jb >= rb->n)
          return 0.0;
        int kb = rb->band[jb];
        double tk = 0.0;
        const auto& lib0 = resolve_library(0);
        for (size_t j2 = 0; j2 <= jc; ++j2) {
          const auto& q = path.seg[j2];
          if (q.material == MATERIAL_VOID)
            continue;
          const ResolveBlock* bq = lib0.get(q.material, g);
          double l = (j2 == jc) ? dist - before : q.length;
          tk += (bq ? bq->D[kb] : 0.0) * l;
        }
        return zeta[kb] / std::sqrt(rb->band_mass[kb]) * std::exp(-tk) / E;
      };
      if (tag == Tag::s) {
        double mu = std::clamp(qc.dir.dot(uo), -1.0, 1.0);
        for (size_t j = 0; j < mat.nuclide_.size(); ++j) {
          int nuc = mat.nuclide_[j];
          probe.total(mc, Eo, qc.sqrtkT, qc.dm); // for elastic_kT
          auto image = elastic_stationary(nuc, ChannelRegistry::elastic_kT(nuc, probe), Eo, mu);
          if (!(image.density > 0.0) || field.group(image.E_in) != g)
            continue;
          double hsrc = field.shape(src_shape, r0, image.E_in);
          if (!(hsrc > 0.0))
            continue;
          double tau_i = ce_depth(path, dist, image.E_in, probe, cache);
          double st_c = probe.total(mc, image.E_in, qc.sqrtkT, qc.dm);
          double sel = mat.atom_density(j, qc.dm) * probe.micro_elastic(nuc);
          double el_tot = elastic_macro(probe, mat, qc.dm);
          double term = hsrc * std::exp(-tau_i) * sel * image.density / (2.0 * PI);
          ce += term;
          by_nuc.push_back({nuc, term});
          if (el_tot > 0.0)
            qce += term * st_c / el_tot;
          double kin = sel * image.density / (2.0 * PI);
          if (rb)
            rs += partner_flux(image.E_in) * kin;
          m1s += tm_c * hsrc * kin;
          m1 += tm_c * field.shape(FrozenField::Shape::phi, qc.region, image.E_in) * kin;
        }
        if (banded && src_p) {
          // density of the band component at z, over every image in group g
          const ResolveBlock* bb = plib.get(mc, g);
          for (size_t j = 0; bb && j < mat.nuclide_.size(); ++j) {
            int nuc = mat.nuclide_[j];
            probe.total(mc, Eo, qc.sqrtkT, qc.dm);
            auto image = elastic_stationary(nuc, ChannelRegistry::elastic_kT(nuc, probe), Eo, mu);
            if (!(image.density > 0.0) || field.group(image.E_in) != g)
              continue;
            int jb = static_cast<int>(std::upper_bound(bb->edges.begin(), bb->edges.end(), image.E_in) - bb->edges.begin()) - 1;
            if (jb < 0 || jb >= bb->n)
              continue;
            int kb = bb->band[jb];
            if (!(a_band(kb) > 0.0))
              continue;
            double tk = 0.0;
            for (size_t j2 = 0; j2 <= jc; ++j2) {
              const auto& q = path.seg[j2];
              if (q.material == MATERIAL_VOID)
                continue;
              double l = (j2 == jc) ? dist - before : q.length;
              tk += band_sigma(q.material, kb) * l;
            }
            probe.total(mc, image.E_in, qc.sqrtkT, qc.dm);
            double sel = mat.atom_density(j, qc.dm) * probe.micro_elastic(nuc);
            double el_tot = elastic_macro(probe, mat, qc.dm);
            // draw density of the incident energy in band kb [1/eV]: the bin's
            // share of the band's cumulative weight, uniform in lethargy
            const auto& cdf = bb->band_cdf[kb];
            int pos = static_cast<int>(std::lower_bound(bb->band_bins[kb].begin(),
                        bb->band_bins[kb].end(), jb) - bb->band_bins[kb].begin());
            double wj = cdf[pos] - (pos > 0 ? cdf[pos - 1] : 0.0);
            double dj = std::log(bb->edges[jb + 1] / bb->edges[jb]);
            double pE = wj / (cdf.back() * dj * image.E_in);
            if (el_tot > 0.0)
              q_band += a_band(kb) * bb->D[kb] * std::exp(-tk) * pE *
                        sel / el_tot * image.density / (2.0 * PI);
          }
        }
        if (rb && part != 7 && part != 8) {
          // Legendre moments of the RESOLVE partner in the lab cosine, to P7
          constexpr int NR = 96, NL = 8;
          probe.total(mc, Eo, qc.sqrtkT, qc.dm);
          vector<double> kTr(mat.nuclide_.size());
          for (size_t j = 0; j < mat.nuclide_.size(); ++j)
            kTr[j] = ChannelRegistry::elastic_kT(mat.nuclide_[j], probe);
          auto legendre = [](double x, double* P) {
            P[0] = 1.0;
            P[1] = x;
            for (int l = 2; l < NL; ++l)
              P[l] = ((2 * l - 1) * x * P[l - 1] - (l - 1) * P[l - 2]) / l;
          };
          double S[NL] = {0.0};
          double P[NL];
          for (int iq = 0; iq < NR; ++iq) {
            double mq = -1.0 + (iq + 0.5) * 2.0 / NR;
            legendre(mq, P);
            for (size_t j = 0; j < mat.nuclide_.size(); ++j) {
              int nuc = mat.nuclide_[j];
              auto image = elastic_stationary(nuc, kTr[j], Eo, mq);
              if (!(image.density > 0.0) || field.group(image.E_in) != g)
                continue;
              double dv = partner_flux(image.E_in);
              if (dv == 0.0)
                continue;
              probe.total(mc, image.E_in, qc.sqrtkT, qc.dm);
              double v = dv * mat.atom_density(j, qc.dm) * probe.micro_elastic(nuc) *
                         image.density / (2.0 * PI);
              for (int l = 0; l < NL; ++l)
                S[l] += 2.0 * PI * v * P[l] * 2.0 / NR;
            }
          }
          legendre(mu, P);
          double acc = 0.0;
          for (int l = 0; l < NL; ++l) {
            acc += (2 * l + 1) * S[l] * P[l] / (4.0 * PI);
            if (l == 1)
              rs1 = acc;
            if (l == 3)
              rs3 = acc;
          }
          rs7 = acc;
        }
        // Legendre moments S_l(E') = 2 pi int P_l(mu) m1(E', mu) dmu / tm_c
        // by midpoint quadrature in the lab cosine
        constexpr int NQ = 32;
        if (kinematic) {
        vector<double> kTn(mat.nuclide_.size());
        probe.total(mc, Eo, qc.sqrtkT, qc.dm);
        for (size_t j = 0; j < mat.nuclide_.size(); ++j)
          kTn[j] = ChannelRegistry::elastic_kT(mat.nuclide_[j], probe);
        double S0 = 0.0, S1 = 0.0;
        for (int iq = 0; iq < NQ; ++iq) {
          double mq = -1.0 + (iq + 0.5) * 2.0 / NQ;
          for (size_t j = 0; j < mat.nuclide_.size(); ++j) {
            int nuc = mat.nuclide_[j];
            auto image = elastic_stationary(nuc, kTn[j], Eo, mq);
            if (!(image.density > 0.0) || field.group(image.E_in) != g)
              continue;
            double hp = field.shape(FrozenField::Shape::phi, qc.region, image.E_in);
            if (!(hp > 0.0))
              continue;
            probe.total(mc, image.E_in, qc.sqrtkT, qc.dm);
            double v = hp * mat.atom_density(j, qc.dm) * probe.micro_elastic(nuc) *
                       image.density / (2.0 * PI);
            S0 += 2.0 * PI * v * 2.0 / NQ;
            S1 += 2.0 * PI * v * mq * 2.0 / NQ;
          }
        }
        p0 = tm_c * S0 / (4.0 * PI);
        p1 = tm_c * (S0 + 3.0 * mu * S1) / (4.0 * PI);
        }
      } else {
        double tau_c = ce_depth(path, dist, Ei, probe, cache);
        double st_c = probe.total(mc, Ei, qc.sqrtkT, qc.dm);
        double nf = probe.particle().macro_xs().nu_fission;
        double kernel = macro_chi(probe, mat, qc.dm, Eo) / (4.0 * PI);
        ce = std::exp(-tau_c) * kernel / k;
        for (size_t j = 0; j < mat.nuclide_.size(); ++j) {
          int nuc = mat.nuclide_[j];
          double nfj = probe.particle().neutron_xs(nuc).nu_fission;
          if (nfj > 0.0)
            by_nuc.push_back({nuc, std::exp(-tau_c) * mat.atom_density(j, qc.dm) * nfj *
                                     fission_chi(nuc, probe, Eo) / (4.0 * PI) / k});
        }
        if (nf > 0.0)
          qce = std::exp(-tau_c) * st_c * kernel / nf;
        double hs = field.shape(src_shape, r0, Ei);
        if (rb && hs > 0.0)
          rs = rs1 = rs3 = rs7 = partner_flux(Ei) / hs * kernel / k;
        m1s = tm_c * kernel / k;
        m1 = hs > 0.0 ? tm_c * field.shape(FrozenField::Shape::phi, qc.region, Ei) / hs * kernel / k : 0.0;
        p0 = p1 = m1;
        // Average of the kernel over E ~ h_phi in group g: the multigroup
        // nu Sigma_f,g times the nu Sigma_f-weighted spectrum of 16 draws
        constexpr int NE = 16;
        double dd, num = 0.0, den = 0.0;
        for (int ie = 0; ie < (partner_diag ? NE : 0); ++ie) {
          double Ej = field.sample_shape(FrozenField::Shape::phi, qc.region, g, &s, dd);
          probe.total(mc, Ej, qc.sqrtkT, qc.dm);
          num += macro_chi(probe, mat, qc.dm, Eo);
          den += probe.particle().macro_xs().nu_fission;
        }
        if (den > 0.0)
          c0 = tm_c * mg.nu_sigma_f(mc, qc.sqrtkT, g) * num / den / (4.0 * PI) / k;
      }
      int go = field.group(Eo);
      double low = 0.0, qlow = 0.0;
      if (go >= 0) {
        double hout = field.shape(out_shape, qc.region, Eo);
        double law = (tag == Tag::s ? mg.nu_sigma_s(mc, qc.sqrtkT, g, go)
                                    : mg_out * mg.chi(mc, qc.sqrtkT, go)) * hout / (4.0 * PI);
        double tm = std::exp(-tau_mg);
        low = tm * law;
        if (tag == Tag::s)
          low *= f_el[static_cast<size_t>(mc) * G + g];
        else
          low /= k;
        if (mg_out > 0.0)
          qlow = tm * st_mg * law / mg_out;
      }
      double proposal = banded ? lambda * qce + lambda_p * q_band + (1.0 - lambda - lambda_p) * qlow
                               : lambda * qce + (1.0 - lambda) * qlow;
      if (!(proposal > 0.0) || !std::isfinite(proposal)) {
        diagnostics[4] += 1.0;
        continue;
      }
      w_ce = factor * ce / proposal;
      w_mg = factor * (resolve ? (part == 6 ? rs3 : rs) : kinematic ? (tag == Tag::s ? p1 : m1) : low) / proposal;
      if (resolve) {
        const double fq = factor / proposal;
        diagnostics[17] += fq * std::abs(ce - rs);
        diagnostics[18] += fq * std::abs(ce - rs1);
        diagnostics[19] += fq * std::abs(ce - rs3);
        diagnostics[20] += fq * std::abs(ce - rs7);
      }
      const double fp = factor / proposal;
      if (tag == Tag::f)
        p0 = c0; // fission: the energy-shape closure uses the averaged kernel
      diagnostics[6] += fp * std::abs(ce - m1s);
      diagnostics[7] += fp * std::abs(ce - m1);
      diagnostics[8] += fp * std::abs(ce - p1);
      diagnostics[9] += fp * std::abs(ce - (tag == Tag::s ? p0 : m1));
      diagnostics[10] += fp * std::abs(p0 - low);
      diagnostics[11] += fp * std::abs(p1 - p0);
      diagnostics[12] += fp * std::abs(m1 - p1);
      diagnostics[13] += fp * m1;
      diagnostics[14] += fp * m1s;
      diagnostics[15] += fp * (p0 - low);
      if (part == 4 && tag == Tag::s && go >= 0) {
        double mu_io = std::clamp(qc.dir.dot(uo), -1.0, 1.0);
        double pn = low * (1.0 + 3.0 * mu_io *
                                   mubar[(static_cast<size_t>(mc) * G + g) * G + go]);
        diagnostics[16] += fp * std::abs(ce - pn);
      }
      for (const auto& t : by_nuc)
        out.add(t.first, tag == Tag::s ? 2 : 18, mc, go, fp * t.second);
      out.add(-1, tag == Tag::s ? 990 : 992, mc, go, w_mg);
    }
    diagnostics[0] += w_ce;
    diagnostics[1] += w_mg;
    diagnostics[2] += std::abs(w_ce - w_mg);
    diagnostics[3] += std::abs(w_ce) + std::abs(w_mg);
    out.r.push_back(xc);
    out.u.push_back(uo);
    out.E.push_back(Eo);
    out.region.push_back(qc.region);
    out.material.push_back(mc);
    out.w_ce.push_back(w_ce);
    out.w_mg.push_back(w_mg);
    out.group_in.push_back(g);
  }
}

//==============================================================================
// Band first-flight sweep: the closure of a band partner
//==============================================================================

void band_sweep(const FrozenField& field, int slot, int64_t n_rays, double dead,
  double active, uint64_t seed, vector<double>& Phi, vector<int>& offset, int n_sample)
{
  // With n_sample > 0, each ray carries only n_sample bands per group, drawn
  // stratified from the bands' source mass over cross section (sum of V q / s),
  // each tallied with weight 1/(n_sample p_k): unbiased, at the cost of G
  // n_sample bands per segment instead of all of them.
  // Long random rays (uniform start in the field's box, isotropic, reflected)
  // carry the uncollided flux of the field's emission (amplitudes 1) in every
  // band of the band library in slot, with the flat source of each region:
  // per segment psi_out = q/s + (psi_in - q/s) exp(-s l). After a dead
  // length, the segment integrals of psi are tallied per region, so that
  // Phi[r][i] = 4 pi V_box / (total active length) * sum int psi ds is the
  // region-integrated band flux. The local part of each segment integral,
  // (q/s) l, is replaced by its expectation (needs the exact region volumes):
  // a control variate that removes the track-length noise.
  const auto& lib = resolve_library(slot);
  const int G = lib.n_groups;
  const int64_t R = field.n_regions();
  offset.assign(G + 1, 0);
  for (int g = 0; g < G; ++g) {
    int L = 0;
    for (int m = 0; m < lib.n_mat; ++m)
      if (const ResolveBlock* b = lib.get(m, g))
        L = std::max(L, b->L);
    offset[g + 1] = offset[g] + L;
  }
  const int NL = offset[G];
  // band sources per region, and band cross sections per material
  vector<double> q(static_cast<size_t>(R) * NL, 0.0);
  for (int64_t r = 0; r < R; ++r) {
    int32_t m = field.region_material(r);
    if (m < 0)
      continue;
    for (int g = 0; g < G; ++g) {
      const ResolveBlock* b = lib.get(m, g);
      if (!b)
        continue;
      for (int y = 0; y < 2; ++y) {
        auto it = lib.source.find(ResolveLibrary::source_key(r, y, g));
        if (it == lib.source.end())
          continue;
        double Q = field.emission(r, static_cast<Tag>(y), g);
        for (int k = 0; k < b->L; ++k)
          q[static_cast<size_t>(r) * NL + offset[g] + k] +=
            Q * std::max(0.0, it->second[k] * std::sqrt(b->band_mass[k])) / (4.0 * PI);
      }
    }
  }
  vector<double> sig(static_cast<size_t>(lib.n_mat) * NL, 0.0);
  for (int m = 0; m < lib.n_mat; ++m)
    for (int g = 0; g < G; ++g)
      if (const ResolveBlock* b = lib.get(m, g))
        for (int k = 0; k < b->L; ++k)
          sig[static_cast<size_t>(m) * NL + offset[g] + k] = b->D[k];

  Phi.assign(static_cast<size_t>(R) * NL, 0.0);
  // band draw probabilities by group
  vector<double> pk(NL, 0.0), cdfk(NL, 0.0);
  if (n_sample > 0) {
    // by source mass times mean free path: the transient part (the only one
    // sampled) is largest for the weakly attenuated bands
    for (int64_t r = 0; r < R; ++r) {
      int32_t m = field.region_material(r);
      if (m < 0)
        continue;
      for (int i = 0; i < NL; ++i) {
        double sgi = sig[static_cast<size_t>(m) * NL + i];
        if (sgi > 0.0)
          pk[i] += q[static_cast<size_t>(r) * NL + i] * field.volume(r) / sgi;
      }
    }
    for (int g = 0; g < G; ++g) {
      double tot = 0.0;
      for (int i = offset[g]; i < offset[g + 1]; ++i)
        tot += pk[i];
      double acc = 0.0;
      for (int i = offset[g]; i < offset[g + 1]; ++i) {
        pk[i] = tot > 0.0 ? pk[i] / tot : 0.0;
        acc += pk[i];
        cdfk[i] = acc;
      }
    }
  }
  vector<int> active_bands;
  vector<double> band_w(NL, 1.0);
  CharacteristicTracer tracer(field);
  SegmentList seg;
  vector<double> psi(NL);
  const Position lo = field.box_lower_left(), hi = field.box_upper_right();
  auto advance = [&](int64_t r, int32_t m, double l, bool tally) {
    const double* sg = &sig[static_cast<size_t>(m) * NL];
    const double* qq = &q[static_cast<size_t>(r) * NL];
    double* ph = &Phi[static_cast<size_t>(r) * NL];
    for (int i : active_bands) {
      // tally only the transient part (psi - q/s)(1 - e)/s: the local part
      // (q/s) l has the known expectation added in closed form below
      double sl = sg[i] * l, integral;
      double e = std::exp(-sl), qs = qq[i] / sg[i];
      integral = (psi[i] - qs) * (-std::expm1(-sl)) / sg[i];
      psi[i] = qs + (psi[i] - qs) * e;
      if (tally)
        ph[i] += band_w[i] * integral;
    }
  };
  for (int64_t n = 0; n < n_rays; ++n) {
    uint64_t s = future_seed(static_cast<uint64_t>(n) * 7919ULL + 3, seed);
    Position x;
    for (int j = 0; j < 3; ++j)
      x[j] = lo[j] + (hi[j] - lo[j]) * prn(&s);
    tracer.begin(x, isotropic(&s));
    std::fill(psi.begin(), psi.end(), 0.0);
    active_bands.clear();
    if (n_sample <= 0) {
      for (int i = 0; i < NL; ++i)
        active_bands.push_back(i);
    } else {
      std::fill(band_w.begin(), band_w.end(), 0.0);
      for (int g = 0; g < G; ++g) {
        if (offset[g + 1] == offset[g] || !(cdfk[offset[g + 1] - 1] > 0.0))
          continue;
        double shift = prn(&s);
        for (int t = 0; t < n_sample; ++t) {
          double xq = (t + shift) / n_sample * cdfk[offset[g + 1] - 1];
          int i = static_cast<int>(std::upper_bound(cdfk.begin() + offset[g], cdfk.begin() + offset[g + 1], xq) - cdfk.begin());
          i = std::min(i, offset[g + 1] - 1);
          if (band_w[i] == 0.0)
            active_bands.push_back(i);
          band_w[i] += 1.0 / (n_sample * pk[i]);
        }
      }
    }
    double done = 0.0;
    const double total = dead + active;
    while (done < total) {
      tracer.extend(64, seg);
      if (seg.size() == 0)
        break;
      for (int j = 0; j < seg.size() && done < total; ++j) {
        double l = std::min(seg.length[j], total - done);
        int32_t m = seg.material[j];
        int64_t r = seg.region[j];
        if (m == MATERIAL_VOID || r < 0) {
          done += l;
          continue;
        }
        if (done + l <= dead) {
          advance(r, m, l, false);
        } else if (done >= dead) {
          advance(r, m, l, true);
        } else {
          advance(r, m, dead - done, false);
          advance(r, m, done + l - dead, true);
        }
        done += l;
      }
      if (seg.ends_in_vacuum)
        break;
    }
  }
  const double vbox = (hi.x - lo.x) * (hi.y - lo.y) * (hi.z - lo.z);
  const double scale = 4.0 * PI * vbox / (static_cast<double>(n_rays) * active);
  for (auto& v : Phi)
    v *= scale;
  // the local part, exactly: E[sum (q/s) l over region r] = (q/s) L V_r / V_box,
  // so its contribution is 4 pi V_r q / s
  for (int64_t r = 0; r < R; ++r) {
    int32_t m = field.region_material(r);
    if (m < 0)
      continue;
    for (int i = 0; i < NL; ++i) {
      double sgi = sig[static_cast<size_t>(m) * NL + i];
      if (sgi > 0.0)
        Phi[static_cast<size_t>(r) * NL + i] +=
          (n_sample > 0 ? 1.0 : 1.0) * 4.0 * PI * field.volume(r) * q[static_cast<size_t>(r) * NL + i] / sgi;
    }
  }
}

} // namespace rmc
} // namespace openmc
