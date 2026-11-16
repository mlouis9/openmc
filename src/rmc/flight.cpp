#include "openmc/rmc/flight.h"
#include "openmc/rmc/capi.h"
#include "openmc/rmc/field.h"

#include <algorithm>
#include <cmath>

#include <fmt/format.h>

#include "openmc/cell.h"
#include "openmc/constants.h"
#include "openmc/error.h"
#include "openmc/geometry.h"
#include "openmc/material.h"
#include "openmc/nuclide.h"
#include "openmc/physics.h"
#include "openmc/random_lcg.h"
#include "openmc/reaction.h"

namespace openmc {
namespace rmc {

double SignedBank::abs_weight() const
{
  double s = 0.0;
  for (double v : w)
    s += std::abs(v);
  return s;
}

double SignedBank::weight(Tag t) const
{
  double s = 0.0;
  for (int64_t i = 0; i < size(); ++i)
    if (tag[i] == static_cast<int>(t))
      s += w[i];
  return s;
}

void FlightTally::reset()
{
  flux.assign(materials.size() * (E_edges.size() - 1), 0.0);
}

FlightPropagator::FlightPropagator() {}

FlightPropagator::Collision FlightPropagator::collide(
  Direction u, double k, uint64_t* seed)
{
  Collision c;
  Particle& p = probe_.scratch();
  const double E_in = probe_.E();
  p.u() = u;
  p.E() = E_in;
  p.wgt() = 1.0;
  p.stream() = STREAM_TRACKING;
  p.seeds(STREAM_TRACKING) = *seed;
  p.local_secondary_bank().clear();
  p.n_secondaries() = 0;

  // Nuclide by Sigma_t,m / Sigma_t, as sample_nuclide()
  int i_nuc = sample_nuclide(p);
  c.nuclide = i_nuc;
  const auto& micro = p.neutron_xs(i_nuc);
  const Nuclide& nuc = *data::nuclides[i_nuc];
  double xi = prn(p.current_seed()) * micro.total;
  double sigma_s = micro.total - micro.absorption;
  if (xi < sigma_s) {
    // Scatter: OpenMC's own scatter() (elastic, S(alpha,beta), inelastic)
    size_t before = p.local_secondary_bank().size();
    scatter(p, i_nuc);
    size_t copies = p.local_secondary_bank().size() - before;
    c.kind = 1;
    c.mt = p.event_mt();
    c.factor = p.wgt() * (1.0 + static_cast<double>(copies));
    c.E = p.E();
    c.u = p.u();
  } else if (nuc.fissionable_ &&
             prn(p.current_seed()) * micro.absorption < micro.fission) {
    // Fission: one neutron, weight nu / k with OpenMC's tabulated
    // nu-fission, as create_fission_sites() uses
    const Reaction& rx = sample_fission(i_nuc, p);
    SourceSite site;
    sample_fission_neutron(i_nuc, rx, &site, p);
    c.kind = 2;
    c.mt = rx.mt_;
    c.factor = micro.nu_fission / micro.fission / k;
    c.E = site.E;
    c.u = site.u;
  }
  *seed = p.seeds(STREAM_TRACKING);
  p.local_secondary_bank().clear();
  p.n_secondaries() = 0;
  p.E() = E_in;
  return c;
}

FlightStats FlightPropagator::propagate(const SignedBank& in, int64_t n,
  double k, uint64_t seed, SignedBank& out, FlightTally* tally,
  FlightAux* aux, bool history)
{
  const bool region_tally =
    aux && !aux->region_fission.empty() && loaded_field;
  FlightStats st;
  const int64_t m = in.size();
  if (m == 0 || n <= 0)
    return st;

  // Cumulative |w| for parent selection
  vector<double> cdf(m);
  double c = 0.0;
  for (int64_t i = 0; i < m; ++i) {
    c += std::abs(in.w[i]);
    cdf[i] = c;
  }
  st.W_abs = c;
  if (!(c > 0.0))
    return st;
  const double unit = c / n;
  const int n_E = tally && tally->active() ? tally->E_edges.size() - 1 : 0;

  for (int64_t j = 0; j < n; ++j) {
    uint64_t s = future_seed(static_cast<uint64_t>(j) * 999983ULL + 1, seed);
    ++st.flights;

    // Parent by |w|
    double x = prn(&s) * c;
    int64_t i = std::min<int64_t>(
      std::upper_bound(cdf.begin(), cdf.end(), x) - cdf.begin(), m - 1);
    const SignedSite par = in[i];
    const double sign = (par.w < 0.0) ? -1.0 : 1.0;

    // Place the geometry particle
    g_.r() = par.r;
    g_.u() = par.u;
    g_.r_last() = par.r;
    g_.u_last() = par.u;
    g_.wgt() = 1.0;
    g_.surface() = SURFACE_NONE;
    g_.n_coord() = 1;
    if (!exhaustive_find_cell(g_)) {
      fatal_error(fmt::format("RMC flight: parent at ({}, {}, {}) is outside "
                              "the geometry.",
        par.r.x, par.r.y, par.r.z));
    }

    // State of the flight: energy and signed weight change at scatterings
    // (history mode); position and direction live in g_
    double E = par.E;
    double wgt = sign * unit;
    for (int64_t n_col = 0;; ++n_col) {
      if (n_col >= 1'000'000) {
        fatal_error("RMC flight: more than 10^6 collisions in one history.");
      }

      // Tally bin of the energy (constant during one flight segment)
      int e_bin = -1;
      if (n_E > 0 && E >= tally->E_edges.front() && E < tally->E_edges.back()) {
        e_bin = static_cast<int>(std::upper_bound(tally->E_edges.begin(),
                                   tally->E_edges.end(), E) -
                                 tally->E_edges.begin()) -
                1;
      }

      // Fly to the sampled optical depth
      double tau = -std::log(prn(&s));
      double path = 0.0;
      bool collided = false;
      for (int64_t crossings = 0; crossings < 100'000'000; ++crossings) {
        const Cell& cell = *model::cells[g_.lowest_coord().cell()];
        int32_t inst = g_.cell_instance();
        int32_t mat = cell.material(inst);
        double sigma =
          probe_.total(mat, E, cell.sqrtkT(inst), cell.density_mult(inst));
        g_.boundary() = distance_to_boundary(g_);
        double d_b = g_.boundary().distance();
        bool hit = sigma * d_b > tau;
        double d = hit ? tau / sigma : d_b;
        if (e_bin >= 0) {
          auto it =
            std::find(tally->materials.begin(), tally->materials.end(), mat);
          if (it != tally->materials.end())
            tally->flux[(it - tally->materials.begin()) * n_E + e_bin] +=
              wgt * d;
        }
        for (int lev = 0; lev < g_.n_coord(); ++lev)
          g_.coord(lev).r() += d * g_.coord(lev).u();
        path += d;
        if (hit) {
          collided = true;
          break;
        }
        tau -= sigma * d_b;
        g_.event_cross_surface();
        if (!g_.alive())
          break;
      }
      if (!collided) {
        ++st.leaked;
        break;
      }
      ++st.collisions;
      {
        const auto& xs = probe_.particle().macro_xs();
        if (xs.total > 0.0) {
          st.absorption += wgt * xs.absorption / xs.total;
          st.nu_fission += wgt * xs.nu_fission / xs.total;
        }
      }

      // Collision estimator of the fission rate by field region: the probe
      // holds the collision cell's cross sections at the flight's energy
      if (region_tally) {
        int64_t reg = loaded_field->locate(g_);
        const auto& xs = probe_.particle().macro_xs();
        if (reg >= 0 && xs.total > 0.0) {
          aux->region_fission[reg] += wgt * xs.fission / xs.total;
          aux->region_nu_fission[reg] += wgt * xs.nu_fission / xs.total;
        }
      }

      // The collision, at the probe's state (material of the collision cell)
      Collision col = collide(g_.u(), k, &s);
      if (col.kind == 0) {
        ++st.captured;
        break;
      }
      if (col.kind == 1) {
        ++st.scattered;
        st.multiplied += wgt * (col.factor - 1.0);
      }
      else
        ++st.fissions;
      if (col.factor == 0.0)
        break;

      if (history && col.kind == 1) {
        // Continue the history from the scattering: new energy, direction,
        // and weight (factor > 1 only for multiplying reactions)
        E = col.E;
        wgt *= col.factor;
        for (int lev = 0; lev < g_.n_coord(); ++lev)
          g_.coord(lev).u() = col.u;
        g_.u_last() = col.u;
        g_.r_last() = g_.r();
        g_.surface() = SURFACE_NONE;
        continue;
      }

      out.push_back({g_.r(), col.u, col.E, wgt * col.factor,
        col.kind == 1 ? Tag::s : Tag::f});
      if (aux && aux->children) {
        aux->parent.push_back(i);
        aux->path.push_back(path);
        aux->nuclide.push_back(col.nuclide);
        aux->E_in.push_back(E);
        Direction ui = g_.u();
        aux->u_in.insert(aux->u_in.end(), {ui.x, ui.y, ui.z});
        aux->mt.push_back(col.mt);
      }
      break;
    }
  }
  return st;
}

} // namespace rmc
} // namespace openmc
