//! \file bank.h
//! \brief Signed particle banks of residual Monte Carlo, as structure of
//! arrays.

#ifndef OPENMC_RMC_BANK_H
#define OPENMC_RMC_BANK_H

#include <cstdint>

#include "openmc/position.h"
#include "openmc/rmc/field.h"
#include "openmc/vector.h"

namespace openmc {
namespace rmc {

//! One signed site, by value (for readability where speed does not matter)
struct SignedSite {
  Position r;
  Direction u;
  double E;
  double w;
  Tag tag;
};

//==============================================================================
//! A bank of signed sites. Selection by |w|, sums by tag, and sorting touch
//! one or two arrays, so the fields are stored separately.
//==============================================================================

struct SignedBank {
  vector<double> x, y, z;    //!< position [cm]
  vector<double> ux, uy, uz; //!< direction
  vector<double> E;          //!< energy [eV]
  vector<double> w;          //!< signed weight
  vector<int> tag;           //!< 0 = s, 1 = f

  int64_t size() const { return static_cast<int64_t>(w.size()); }

  void clear()
  {
    for (auto* v : {&x, &y, &z, &ux, &uy, &uz, &E, &w})
      v->clear();
    tag.clear();
  }

  void push_back(const SignedSite& s)
  {
    x.push_back(s.r.x);
    y.push_back(s.r.y);
    z.push_back(s.r.z);
    ux.push_back(s.u.x);
    uy.push_back(s.u.y);
    uz.push_back(s.u.z);
    E.push_back(s.E);
    w.push_back(s.w);
    tag.push_back(static_cast<int>(s.tag));
  }

  SignedSite operator[](int64_t i) const
  {
    return {{x[i], y[i], z[i]}, {ux[i], uy[i], uz[i]}, E[i], w[i],
      static_cast<Tag>(tag[i])};
  }

  //! Sum of |w|
  double abs_weight() const;
  //! Sum of w over the sites of a tag
  double weight(Tag t) const;
};

} // namespace rmc
} // namespace openmc

#endif // OPENMC_RMC_BANK_H
