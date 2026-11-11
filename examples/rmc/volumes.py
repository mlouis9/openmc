"""Exact source-region volumes for the verification problems.

Random ray's prescribed volume estimator (``volume_estimator = 'prescribed'``)
reads source-region volumes from an HDF5 file keyed by (cell ID, instance,
mesh bin). For the problems in problems.py those volumes have closed forms:
each source region is the intersection of a rectangular mesh voxel with
either the inside or the outside of the fuel cylinder, extruded in z.

The functions here compute them and write the file. They are independent of
OpenMC's geometry routines, so comparing them with point-location volumes
checks both.
"""
import math

import h5py
import numpy as np

import problems


def _disk_strip_integral(t0, t1, r, y):
    r"""Integral over t in [t0, t1] (within [-r, r]) of the length of the
    chord {Y : Y <= y, X = t} inside the disk of radius r at the origin.

    The chord at abscissa t spans [-s, s] with s = sqrt(r^2 - t^2), so its
    part below y has length clamp(y + s, 0, 2 s).
    """
    def S(t):
        # Antiderivative of s(t) = sqrt(r^2 - t^2)
        t = min(max(t, -r), r)
        return 0.5 * (t * math.sqrt(max(r * r - t * t, 0.0))
                      + r * r * math.asin(t / r))

    def integrate(a, b, mode):
        if b <= a:
            return 0.0
        if mode == 'full':     # 2 s(t)
            return 2.0 * (S(b) - S(a))
        if mode == 'partial':  # y + s(t)
            return y * (b - a) + (S(b) - S(a))
        return 0.0             # 'none'

    if y >= r:
        return integrate(t0, t1, 'full')
    if y <= -r:
        return 0.0
    # |t| <= w: s(t) >= |y|, chord crosses Y = y; |t| > w: s(t) < |y|
    w = math.sqrt(r * r - y * y)
    outer = 'full' if y >= 0.0 else 'none'
    total = 0.0
    for a, b, mode in ((-r, -w, outer), (-w, w, 'partial'), (w, r, outer)):
        total += integrate(max(a, t0), min(b, t1), mode)
    return total


def disk_quadrant_area(x, y, r):
    """Area of the disk of radius r at the origin with X <= x and Y <= y."""
    x = min(max(x, -r), r)
    return _disk_strip_integral(-r, x, r, y)


def circle_rectangle_area(cx, cy, r, x0, x1, y0, y1):
    """Area of the intersection of a disk (centre (cx, cy), radius r) with the
    rectangle [x0, x1] x [y0, y1], by inclusion-exclusion of quadrant areas."""
    a, b = x0 - cx, x1 - cx
    c, d = y0 - cy, y1 - cy
    return (disk_quadrant_area(b, d, r) - disk_quadrant_area(a, d, r)
            - disk_quadrant_area(b, c, r) + disk_quadrant_area(a, c, r))


def pincell_volumes(mesh, radius=problems.R_FUEL, center=(0.0, 0.0),
                    fuel_cell=problems.CELL_FUEL, mod_cell=problems.CELL_MOD):
    """Exact source-region volumes of a z-extruded pin cell under a regular
    mesh with one voxel in z.

    Returns (cell_id, instance, mesh_bin, volume) arrays, two rows per voxel
    (fuel, then moderator), including zero volumes. Mesh bins are numbered
    with x fastest, as in OpenMC's RegularMesh.
    """
    nx, ny, nz = mesh.dimension
    if nz != 1:
        raise ValueError('pincell_volumes expects one mesh voxel in z.')
    lo = np.asarray(mesh.lower_left, dtype=float)
    hi = np.asarray(mesh.upper_right, dtype=float)
    w = (hi - lo) / np.array([nx, ny, nz])
    height = w[2]

    cell_id, instance, mesh_bin, volume = [], [], [], []
    for iy in range(ny):
        for ix in range(nx):
            b = ix + nx * iy
            x0, y0 = lo[0] + ix * w[0], lo[1] + iy * w[1]
            a_fuel = circle_rectangle_area(center[0], center[1], radius,
                                           x0, x0 + w[0], y0, y0 + w[1])
            a_fuel = min(max(a_fuel, 0.0), w[0] * w[1])
            for cid, area in ((fuel_cell, a_fuel),
                              (mod_cell, w[0] * w[1] - a_fuel)):
                cell_id.append(cid)
                instance.append(0)
                mesh_bin.append(b)
                volume.append(area * height)
    return (np.array(cell_id, dtype=np.int32), np.array(instance, dtype=np.int32),
            np.array(mesh_bin, dtype=np.int64), np.array(volume, dtype=float))


def write_source_region_volumes(path, cell_id, instance, mesh_bin, volume):
    """Write a source-region volume file for the prescribed volume estimator."""
    with h5py.File(path, 'w') as f:
        f.attrs['filetype'] = np.bytes_('source_region_volumes')
        f.attrs['version'] = 1
        f.create_dataset('cell_id', data=np.asarray(cell_id, dtype=np.int32))
        f.create_dataset('instance', data=np.asarray(instance, dtype=np.int32))
        f.create_dataset('mesh_bin', data=np.asarray(mesh_bin, dtype=np.int64))
        f.create_dataset('volume', data=np.asarray(volume, dtype=float))
