"""Python side of the frozen field for residual Monte Carlo (RMC).

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
    box: tuple                 # (lower_left, upper_right) of the ray source box

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
                m['chi'][()].reshape(M, G),
                (f['domain_lower_left'][()], f['domain_upper_right'][()]))

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

def as_meshes(meshes):
    """A list of meshes from one mesh or an iterable of meshes."""
    if isinstance(meshes, openmc.MeshBase):
        return [meshes]
    return list(meshes)


def regular_mesh_bin(mesh, xyz):
    """Regular-mesh bin, x fastest, or -1 outside the mesh."""
    ll = np.asarray(mesh.lower_left, dtype=float)
    dim = np.asarray(mesh.dimension, dtype=int)
    w = (np.asarray(mesh.upper_right, dtype=float) - ll) / dim
    ijk = np.floor((np.asarray(xyz, dtype=float) - ll) / w).astype(int)
    if np.any(ijk < 0) or np.any(ijk >= dim):
        return -1
    return int(ijk[0] + dim[0] * (ijk[1] + dim[1] * ijk[2]))


class ReferenceField:
    """Slow, independent implementation of the frozen field.

    Must run inside a CE process started with openmc.lib.init(). A point is
    located as random ray does it: find the cell instance with
    openmc.lib.find_cell, look up which source-region mesh (if any) the field
    file assigns to that instance, and take the bin in that mesh. Evaluation
    returns Q_t per unit volume, energy [eV], and solid angle.

    meshes holds the source-region meshes (one mesh or an iterable); the field
    file names the one each region uses by ID.
    """

    def __init__(self, field, lift_path, meshes, scale=1.0):
        self.F, self.scale = field, scale
        self.meshes = {m.id: m for m in as_meshes(meshes)}
        self.key_to_region = {k: i for i, k in enumerate(field.keys())}
        # Mesh ID of each cell instance (-1 for none); one per instance
        self.mesh_of_base = {}
        for c, i, mid in zip(field.cell_id.tolist(), field.instance.tolist(),
                             field.mesh_id.tolist()):
            if self.mesh_of_base.setdefault((c, i), mid) != mid:
                raise ValueError(f'cell {c} instance {i} has two meshes')
        missing = {m for m in self.mesh_of_base.values() if m >= 0} - set(self.meshes)
        if missing:
            raise ValueError(f'meshes {sorted(missing)} not given')
        with h5py.File(lift_path, 'r') as f:
            self.fine = f['fine_edges'][()]
            mids = f['material_ids'][()]
            n_fine = len(self.fine) - 1
            self.lift = {t: {int(m): f[f'h_{t}'][()].reshape(len(mids), n_fine)[i]
                             for i, m in enumerate(mids)} for t in ('s', 'f')}

    def region_mesh_id(self, region):
        """ID of the mesh whose voxels contain the region, or -1 (no mesh, or
        outside it), as FrozenField::region_mesh()."""
        F = self.F
        if F.mesh_bin[region] < 0:
            return -1
        return int(F.mesh_id[region])

    def locate(self, xyz):
        cell, inst = openmc.lib.find_cell(xyz)
        mid = self.mesh_of_base.get((cell.id, inst))
        if mid is None:
            return -1
        b = regular_mesh_bin(self.meshes[mid], xyz) if mid >= 0 else 0
        return self.key_to_region.get((cell.id, inst, b), -1)

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


def region_volumes(ref, n_per_voxel, seed, n_box=0):
    """Region volumes [cm^3] by locating uniform points.

    A region is counted only by its own sampling domain: n_per_voxel points
    in every voxel of its mesh, or n_box points in the ray source box for
    regions without a mesh (or outside it). Returns (volumes, hit counts).
    """
    rng = np.random.default_rng(seed)
    R = ref.F.n_regions
    owner = np.array([ref.region_mesh_id(r) for r in range(R)])
    counts = np.zeros(R)
    V_dom = np.zeros(R)
    n_dom = np.zeros(R)

    def sample(lo, w, n, mesh_id):
        for x in lo + w * rng.random((n, 3)):
            r = ref.locate(x)
            if r >= 0 and owner[r] == mesh_id:
                counts[r] += 1

    for mid in sorted({m for m in owner if m >= 0}):
        for *_, lo, w in voxel_iter(ref.meshes[mid]):
            sample(lo, w, n_per_voxel, mid)
        V_dom[owner == mid] = float(np.prod(w))
        n_dom[owner == mid] = n_per_voxel
    if np.any(owner < 0):
        if n_box <= 0:
            raise ValueError('regions without a mesh need n_box > 0')
        lo, hi = (np.asarray(v, dtype=float) for v in ref.F.box)
        sample(lo, hi - lo, n_box, -1)
        V_dom[owner < 0] = float(np.prod(hi - lo))
        n_dom[owner < 0] = n_box
    return V_dom * counts / n_dom, counts


def field_scale(field, V):
    """Scale making <1, Q_f> = 1/k_mg with region volumes V."""
    return (1.0 / field.k_mg) / float(np.sum(V[:, None] * field.Q_f))


def write_volumes(path, VA, VB, counts_A, counts_B, scale, n_per_voxel, seeds,
                  n_box=0):
    with h5py.File(path, 'w') as f:
        f.attrs['version'] = 2
        f.attrs['scale'] = scale
        f.attrs['n_per_voxel'] = n_per_voxel
        f.attrs['n_box'] = n_box
        f.attrs['seeds'] = np.asarray(seeds)
        f['V_A'], f['V_B'] = VA, VB
        f['counts_A'], f['counts_B'] = counts_A, counts_B
