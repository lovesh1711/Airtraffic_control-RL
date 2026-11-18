"""Closest-approach cache and polygon helpers in a local NM frame (x = north, y = east)."""

import numpy as np
import bluesky as bs

from core.observations import _cpa

_CACHE = {"key": None, "table": None}


def cpa_table(ac_idx):
    """[(idx, tcpa_s, dcpa_m, dist_nm), ...] for every other aircraft.

    The observation and the reward both need this on every step, so it is
    cached on (sim time, number of aircraft, own index).
    """
    key = (float(bs.sim.simt), int(bs.traf.ntraf), int(ac_idx))
    if _CACHE["key"] == key:
        return _CACHE["table"]

    lat0 = bs.traf.lat[ac_idx]
    lon0 = bs.traf.lon[ac_idx]
    table = []
    for j in range(bs.traf.ntraf):
        if j == ac_idx:
            continue
        tcpa, dcpa_m = _cpa(ac_idx, j)
        _, dist_nm = bs.tools.geo.kwikqdrdist(lat0, lon0, bs.traf.lat[j], bs.traf.lon[j])
        table.append((j, tcpa, dcpa_m, dist_nm))

    _CACHE["key"] = key
    _CACHE["table"] = table
    return table


def clear():
    # the sim clock can rewind on reset, so keys from the last episode may collide
    _CACHE["key"] = None
    _CACHE["table"] = None


def sector_segments(own_lat, own_lon, vertices):
    """Polygon edges (A, B) in the ownship's local NM frame."""
    verts = np.asarray(vertices, dtype=float)
    x = (verts[:, 0] - own_lat) * 60.0
    y = (verts[:, 1] - own_lon) * 60.0 * np.cos(np.radians(own_lat))
    A = np.column_stack((x, y))
    return A, np.roll(A, -1, axis=0)


def _cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def ray_exit_distance(A, B, u, eps=1e-9):
    """Distance along the unit vector u to the first edge crossing (inf if none)."""
    d = B - A
    denom = _cross(np.broadcast_to(u, d.shape), d)
    ok = np.abs(denom) > eps
    if not ok.any():
        return np.inf
    safe = np.where(ok, denom, 1.0)
    t = _cross(A, d) / safe
    s = _cross(A, np.broadcast_to(u, A.shape)) / safe
    hit = ok & (t > 0.0) & (s >= 0.0) & (s <= 1.0)
    return float(t[hit].min()) if hit.any() else np.inf


def nearest_edge(A, B, eps=1e-12):
    """(distance, unit vector) from the ownship to the closest edge point."""
    d = B - A
    dd = (d * d).sum(axis=1)
    proj = np.clip(-(A * d).sum(axis=1) / np.maximum(dd, eps), 0.0, 1.0)
    closest = A + proj[:, None] * d
    dist = np.linalg.norm(closest, axis=1)
    k = int(np.argmin(dist))
    r = float(dist[k])
    return r, (closest[k] / r if r > eps else np.array([1.0, 0.0]))


def point_in_polygon(A, B):
    """Crossing-number test for the origin."""
    ay, by = A[:, 1], B[:, 1]
    straddle = (ay > 0.0) != (by > 0.0)
    if not straddle.any():
        return False
    ax, bx = A[:, 0], B[:, 0]
    dy = np.where(straddle, by - ay, 1.0)
    xint = ax + (-ay) * (bx - ax) / dy
    return bool(np.count_nonzero(straddle & (xint > 0.0)) % 2)


def ray_miss_distance(A, B, u, eps=1e-12):
    """(miss, along): closest approach of the forward ray to the polygon, and where.

    Zero miss if the ray goes through it. This is the static version of
    (dcpa, tcpa).
    """
    if np.isfinite(ray_exit_distance(A, B, u)):
        return 0.0, 0.0
    # no intersection, so the minimum is at a corner or at the ray origin
    corners = np.vstack((A, B))
    t = np.clip(corners @ u, 0.0, None)
    d_corner = np.linalg.norm(corners - t[:, None] * u, axis=1)

    d = B - A
    dd = (d * d).sum(axis=1)
    proj = np.clip(-(A * d).sum(axis=1) / np.maximum(dd, eps), 0.0, 1.0)
    closest = A + proj[:, None] * d
    d_edge = np.linalg.norm(closest, axis=1)

    dists = np.concatenate((d_corner, d_edge))
    alongs = np.concatenate((t, np.clip(closest @ u, 0.0, None)))
    k = int(np.argmin(dists))
    return float(dists[k]), float(alongs[k])
