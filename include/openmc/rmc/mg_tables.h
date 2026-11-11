//! \file mg_tables.h
//! \brief Multigroup tables for random ray in a continuous-energy process.

#ifndef OPENMC_RMC_MG_TABLES_H
#define OPENMC_RMC_MG_TABLES_H

#include <string>

namespace openmc {

class RandomRayMGData;

namespace rmc {

//! Fill random ray's multigroup data from the material tables of a field
//! file (written by FlatSourceDomain::export_rmc_field(), version 2 or
//! later), so that random ray can run in a process that loaded
//! continuous-energy data. Materials are matched to model::materials by ID,
//! groups are reversed from the file's ascending order to the solver's
//! descending order, and every value is copied exactly, so the tables equal
//! the ones the multigroup run used. Requires an initialized model.
void load_mg_tables(const std::string& field_file, RandomRayMGData& out);

} // namespace rmc
} // namespace openmc

#endif // OPENMC_RMC_MG_TABLES_H
