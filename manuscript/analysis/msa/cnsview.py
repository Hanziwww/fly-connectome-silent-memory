"""Display transform for the unfolded CNS view: brain in frontal view, ventral nerve cord (VNC) hung below it.

MaleCNS coordinates are in micrometres: +x is the fly's left, +y is ventral and +z is posterior. The brain is
shown in frontal view (X = x, Y = y, looking along +z), so the fly's left is on the image right. Points
posterior to the neck (z > NECK_Z) are rotated 90 degrees about the mediolateral axis through the neck
(y = NECK_Y, z = NECK_Z), so the VNC hangs below the brain in ventral view:

    X = x,  Y = NECK_Y + (z - NECK_Z),  depth = NECK_Z - (y - NECK_Y)

Each side is rigid and the two sides meet on the axis, like a dissected CNS mount. It is a display convention.
NECK_Z is the posterior limit of the brain neuropil meshes and NECK_Y the centre of the cervical connective
there; P10 re-measures both from the meshes and records them in figure_data/fd_display_transform.csv.
"""
import numpy as np

NECK_Z = 416.0
NECK_Y = 415.5


def unfold(xyz, vnc=None):
    """Return (n x 3 array of X, Y, depth in micrometres, boolean mask of VNC-side points). The side follows z unless
    a mask is given (whole meshes keep one side, so no triangle is stretched across the hinge)."""
    p = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    vnc = z > NECK_Z if vnc is None else np.broadcast_to(np.asarray(vnc, bool), z.shape)
    out = np.empty_like(p)
    out[:, 0] = x
    out[:, 1] = np.where(vnc, NECK_Y + (z - NECK_Z), y)
    out[:, 2] = np.where(vnc, NECK_Z - (y - NECK_Y), z)
    return out, vnc


def unfold_vectors(v, vnc):
    """Rotate direction vectors (normals) with the same rule; vnc is the mask returned by unfold()."""
    d = np.asarray(v, dtype=np.float64).reshape(-1, 3)
    out = d.copy()
    out[vnc, 1] = d[vnc, 2]
    out[vnc, 2] = -d[vnc, 1]
    return out
