"""Python side of the frozen field for multilevel RMC.

Reads the field file written by random ray, recomputes its emission sources
independently, builds the within-group lifts, and provides a slow reference
implementation of the field (locator, evaluator, region volumes) used to test
the C++ implementation. Definitions: docs/source/methods/rmc.rst,
methods_rmc_field.
"""
from dataclasses import dataclass

import h5py
import numpy as np

import openmc


# --------------------------------------------------------------------------
# Field file
# --------------------------------------------------------------------------

@dataclass
class FieldFile:
    """Contents of a field file. Groups ascend in energy; [R, G] arrays."""
    k_mg: float
    edges: np.ndarray
    cell_id: np.ndarray
    instance: np.ndarray
    mesh_id: np.ndarray
    mesh_bin: np.ndarray
    material_id: np.ndarray
    volume_rr: np.ndarray
    density_mult: np.ndarray
    phi: np.ndarray
    Q_s: np.ndarray
    Q_f: np.ndarray
    mat_ids: np.ndarray        # [M] material IDs of the MGXS tables
    sigma_t: np.ndarray        # [M, G]
    sigma_s: np.ndarray        # [M, G_in, G_out]
    nu_sigma_f: np.ndarray     # [M, G]
    chi: np.ndarray            # [M, G]

    @classmethod
    def load(cls, path):
        with h5py.File(path, 'r') as f:
            G, R = int(f.attrs['n_groups']), int(f.attrs['n_regions'])
            r, m = f['regions'], f['materials']
            M = len(m['material_id'])
            return cls(
                float(f.attrs['k_mg']), f['group_edges'][()],
                r['cell_id'][()], r['instance'][()], r['mesh_id'][()],
                r['mesh_bin'][()], r['material_id'][()], r['volume_rr'][()],
                r['density_mult'][()],
                r['phi'][()].reshape(R, G), r['Q_s'][()].reshape(R, G),
                r['Q_f'][()].reshape(R, G),
                m['material_id'][()], m['sigma_t'][()].reshape(M, G),
                m['sigma_s'][()].reshape(M, G, G), m['nu_sigma_f'][()].reshape(M, G),
                m['chi'][()].reshape(M, G))

    @property
    def n_regions(self):
        return len(self.cell_id)

    @property
    def n_groups(self):
        return len(self.edges) - 1

    def keys(self):
        return list(zip(self.cell_id.tolist(), self.instance.tolist(),
                        self.mesh_bin.tolist()))


def recompute_emission(field, mgxs_path, materials_by_id):
    """Q_s and Q_f from mgxs.h5 and the exported flux, independently of the
    C++ export. Returns [R, G] arrays in ascending group order."""
    lib = openmc.MGXSLibrary.from_hdf5(str(mgxs_path))
    by_name = {x.name: x for x in lib.xsdatas}
    R, G = field.phi.shape
    Q_s, Q_f = np.zeros((R, G)), np.zeros((R, G))
    for i in range(R):
        mid = int(field.material_id[i])
        if mid < 0:
            continue
        xs = _xsdata_for(by_name, materials_by_id[mid])
        dm = field.density_mult[i]
        phi_desc = field.phi[i, ::-1]                 # OpenMC order: high -> low
        # [g_in, g_out], descending. In OpenMC's MGXS format the scatter matrix
        # is already a production (nu-scatter) matrix; multiplicity_matrix
        # would recover the bare scattering matrix, which is not wanted here.
        S = xs.scatter_matrix[0][..., 0]
        # Non-fissile materials store None
        nsf = _per_group(xs.nu_fission, G)
        chi = _per_group(xs.chi, G)
        chi = np.nan_to_num(chi, nan=0.0)          # as FlatSourceDomain::flatten_xs()
        Q_s[i] = dm * (phi_desc @ S)[::-1]
        Q_f[i] = dm * (chi * (nsf @ phi_desc) / field.k_mg)[::-1]
    return Q_s, Q_f


def _per_group(data, G):
    """First-temperature group values; None (non-fissile) and scalars allowed."""
    v = data[0] if data else None
    if v is None:
        return np.zeros(G)
    return np.broadcast_to(np.asarray(v, dtype=float), (G,))


def _xsdata_for(by_name, material):
    """material_wise MGXS are named '<name>_<id>' or after the material."""
    for name in (f'{material.name}_{material.id}', material.name, f'mat_{material.id}'):
        if name in by_name:
            return by_name[name]
    raise KeyError(f'No MGXS for material {material.id} ({material.name}); '
                   f'library has {sorted(by_name)}')


# --------------------------------------------------------------------------
# Lifts
# --------------------------------------------------------------------------

def build_lifts(sp_path, fine_edges, group_edges, material_ids, out_path):
    """Write the lift file from the tallies of mg.run_lift_tallies().

    Heights are densities per eV, normalized so each group integrates to one
    over the part of it covered by the fine grid. Arrays are flat,
    material-major: h_t[m * F + i].
    """
    fine_edges = np.asarray(fine_edges, dtype=float)
    dE = np.diff(fine_edges)
    group_of_fine = np.searchsorted(group_edges, fine_edges[:-1], side='right') - 1
    out = {'h_s': [], 'h_f': [], 'h_phi': []}
    with openmc.StatePoint(sp_path) as sp:
        te, tp = sp.get_tally(name='lift_emission'), sp.get_tally(name='lift_flux')
        for mid in material_ids:
            for key, tally, score in (('h_s', te, 'nu-scatter'), ('h_f', te, 'nu-fission'),
                                      ('h_phi', tp, 'flux')):
                v = tally.get_values(scores=[score], filters=[openmc.MaterialFilter],
                                     filter_bins=[(int(mid),)]).ravel()
                out[key].append(_normalize_per_group(v / dE, dE, group_of_fine, fine_edges))
    with h5py.File(out_path, 'w') as f:
        f.attrs['version'] = 1
        f['fine_edges'] = fine_edges
        f['group_edges'] = np.asarray(group_edges, dtype=float)
        f['material_ids'] = np.asarray(material_ids, dtype=np.int32)
        for key, rows in out.items():
            f[key] = np.concatenate(rows)


def _normalize_per_group(h, dE, group_of_fine, fine_edges):
    """Each group integrates to 1; groups with no tallied content get 1/E."""
    h = np.array(h, dtype=float)
    mid = np.sqrt(fine_edges[:-1] * fine_edges[1:])
    for g in np.unique(group_of_fine):
        sel = group_of_fine == g
        mass = np.sum(h[sel] * dE[sel])
        if not mass > 0.0:
            h[sel] = 1.0 / mid[sel]
            mass = np.sum(h[sel] * dE[sel])
        h[sel] /= mass
    return h


# --------------------------------------------------------------------------
# Reference field
# --------------------------------------------------------------------------

class ReferenceField:
    """Slow, independent implementation of the frozen field.

    Must run inside a CE process started with openmc.lib.init(). Region
    location uses openmc.lib.find_cell and the source-region mesh bin;
    evaluation returns Q_t per unit volume, energy [eV], and solid angle.
    """

    def __init__(self, field, lift_path, mesh, scale=1.0):
        self.F, self.mesh, self.scale = field, mesh, scale
        self.key_to_region = {k: i for i, k in enumerate(field.keys())}
        with h5py.File(lift_path, 'r') as f:
            self.fine = f['fine_edges'][()]
            mids = f['material_ids'][()]
            n_fine = len(self.fine) - 1
            self.lift = {t: {int(m): f[f'h_{t}'][()].reshape(len(mids), n_fine)[i]
                             for i, m in enumerate(mids)} for t in ('s', 'f')}
        self._ll = np.asarray(mesh.lower_left, dtype=float)
        self._ur = np.asarray(mesh.upper_right, dtype=float)
        self._dim = np.asarray(mesh.dimension, dtype=int)

    def mesh_bin(self, xyz):
        """Regular-mesh bin, x fastest, or -1 outside the mesh."""
        w = (self._ur - self._ll) / self._dim
        ijk = np.floor((np.asarray(xyz, dtype=float) - self._ll) / w).astype(int)
        if np.any(ijk < 0) or np.any(ijk >= self._dim):
            return -1
        return int(ijk[0] + self._dim[0] * (ijk[1] + self._dim[1] * ijk[2]))

    def locate(self, xyz):
        cell, inst = openmc.lib.find_cell(xyz)
        return self.key_to_region.get((cell.id, inst, self.mesh_bin(xyz)), -1)

    def evaluate(self, region, tag, E):
        """Edge convention: bin i is [e_i, e_{i+1}); outside the grid gives 0."""
        if region < 0:
            return 0.0
        mid = int(self.F.material_id[region])
        if mid < 0:
            return 0.0
        if not (self.fine[0] <= E < self.fine[-1]) or not (self.F.edges[0] <= E < self.F.edges[-1]):
            return 0.0
        g = np.searchsorted(self.F.edges, E, side='right') - 1
        i = np.searchsorted(self.fine, E, side='right') - 1
        Q = (self.F.Q_s if tag == 's' else self.F.Q_f)[region, g]
        return self.scale * Q * self.lift[tag][mid][i] / (4.0 * np.pi)


def voxel_iter(mesh):
    """(i, j, k, lower corner, widths) for every voxel of a regular mesh."""
    ll = np.asarray(mesh.lower_left, dtype=float)
    dim = np.asarray(mesh.dimension, dtype=int)
    w = (np.asarray(mesh.upper_right, dtype=float) - ll) / dim
    for k in range(dim[2]):
        for j in range(dim[1]):
            for i in range(dim[0]):
                yield i, j, k, ll + w * np.array([i, j, k]), w


def region_volumes(ref, n_per_voxel, seed):
    """Region volumes [cm^3] from points uniform in each voxel."""
    rng = np.random.default_rng(seed)
    counts = np.zeros(ref.F.n_regions)
    V_vox = None
    for *_, lo, w in voxel_iter(ref.mesh):
        V_vox = float(np.prod(w))
        for x in lo + w * rng.random((n_per_voxel, 3)):
            R = ref.locate(x)
            if R >= 0:
                counts[R] += 1
    return V_vox * counts / n_per_voxel, counts


def field_scale(field, V):
    """Scale making <1, Q_f> = 1/k_mg with region volumes V."""
    return (1.0 / field.k_mg) / float(np.sum(V[:, None] * field.Q_f))


def write_volumes(path, VA, VB, counts_A, counts_B, scale, n_per_voxel, seeds):
    with h5py.File(path, 'w') as f:
        f.attrs['version'] = 1
        f.attrs['scale'] = scale
        f.attrs['n_per_voxel'] = n_per_voxel
        f.attrs['seeds'] = np.asarray(seeds)
        f['V_A'], f['V_B'] = VA, VB
        f['counts_A'], f['counts_B'] = counts_A, counts_B
