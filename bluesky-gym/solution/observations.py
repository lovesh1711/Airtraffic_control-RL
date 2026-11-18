"""Observation components: intruders, obstacles, sector boundary and route context.

Compared with the components in core/observations.py:
- positions are normalised by sector scale instead of 1000 km,
- empty slots are padded with "nothing there" values plus a validity mask
  (zeros would read as an aircraft sitting on top of us),
- closest-approach / time-to-entry / time-to-exit features are given directly.
"""

import numpy as np
import bluesky as bs
from gymnasium import spaces

from core.observations import _bound_angle
from core.scenario import resample_perimeter
from solution.geometry import (cpa_table, sector_segments, ray_exit_distance,
                               nearest_edge, point_in_polygon, ray_miss_distance)

NM2KM = 1.852


class RouteObservation:
    """Seven scalars about the route the action is anchored to (body frame):
    direction and range to the current anchor, direction of the next leg,
    remaining route length, and time slack before the 3000 s limit."""

    KEYS = ("route_anchor_cos", "route_anchor_sin", "route_anchor_distance",
            "route_next_cos", "route_next_sin", "route_remaining", "route_time_slack")

    def __init__(self, dist_norm=100.0, route_norm=200.0, slack_norm=1000.0):
        self.dist_norm = dist_norm
        self.route_norm = route_norm
        self.slack_norm = slack_norm

    def space(self):
        return {k: spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float64)
                for k in self.KEYS}

    def observe(self, ac_id, anchor_ll, next_ll, remaining_nm, slack_s):
        i = bs.traf.id2idx(ac_id)
        lat, lon, hdg = bs.traf.lat[i], bs.traf.lon[i], bs.traf.hdg[i]
        qa, da = bs.tools.geo.kwikqdrdist(lat, lon, anchor_ll[0], anchor_ll[1])
        if next_ll is None:
            qn = qa
        else:
            qn, _ = bs.tools.geo.kwikqdrdist(anchor_ll[0], anchor_ll[1], next_ll[0], next_ll[1])
        ra, rn = np.radians(qa - hdg), np.radians(qn - hdg)
        one = lambda v: np.array([float(v)])
        return {
            "route_anchor_cos": one(np.cos(ra)),
            "route_anchor_sin": one(np.sin(ra)),
            "route_anchor_distance": one(min(da, 2 * self.dist_norm) / self.dist_norm),
            "route_next_cos": one(np.cos(rn)),
            "route_next_sin": one(np.sin(rn)),
            "route_remaining": one(min(remaining_nm, 2 * self.route_norm) / self.route_norm),
            "route_time_slack": one(np.clip(slack_s / self.slack_norm, -1.0, 3.0)),
        }


def _plan_positions(latlon, speed_nm_s, terminal, times, ref_lat, ref_lon):
    """Positions (len(times), 2) in a local NM frame when flying a polyline.

    latlon[0] is the current position. Past the last point, a terminal plan
    (our aircraft, removed at its goal) gives NaN and a non-terminal one (a
    scripted intruder) keeps going along its last leg.
    """
    pts = np.asarray(latlon, dtype=float)
    xy = np.column_stack(((pts[:, 0] - ref_lat) * 60.0,
                          (pts[:, 1] - ref_lon) * 60.0 * np.cos(np.radians(ref_lat))))
    d = times * speed_nm_s
    if len(xy) == 1:
        return np.repeat(xy, len(times), axis=0)
    seg = np.diff(xy, axis=0)
    L = np.hypot(seg[:, 0], seg[:, 1])
    cum = np.concatenate(([0.0], np.cumsum(L)))
    out = np.column_stack((np.interp(d, cum, xy[:, 0]), np.interp(d, cum, xy[:, 1])))
    beyond = d > cum[-1]
    if beyond.any():
        if terminal:
            out[beyond] = np.nan
        else:
            k = np.flatnonzero(L > 1e-9)
            u = seg[k[-1]] / L[k[-1]] if len(k) else np.zeros(2)
            out[beyond] = xy[-1] + (d[beyond] - cum[-1])[:, None] * u
    return out


def _closest_along(own, other, times):
    """(min distance, time of it) between two sampled trajectories.

    Exact within each sample interval for piecewise-linear relative motion.
    NaN samples are skipped.
    """
    r = other - own
    r0, r1 = r[:-1], r[1:]
    ok = np.isfinite(r0).all(axis=1) & np.isfinite(r1).all(axis=1)
    if not ok.any():
        return np.inf, np.inf
    r0, r1 = r0[ok], r1[ok]
    t0, t1 = times[:-1][ok], times[1:][ok]
    dr = r1 - r0
    dd = (dr * dr).sum(axis=1)
    s = np.clip(-(r0 * dr).sum(axis=1) / np.maximum(dd, 1e-12), 0.0, 1.0)
    p = r0 + s[:, None] * dr
    dist = np.hypot(p[:, 0], p[:, 1])
    k = int(np.argmin(dist))
    return float(dist[k]), float(t0[k] + s[k] * (t1[k] - t0[k]))


class RiskIntruderObservation:
    """The n most urgent other aircraft, with closest-approach features.

    intent: also give tcpa/dcpa when both aircraft fly their plans
        (closest_fn(own_id, other_id) -> (dmin_nm, t_min_s)).
    share: multi-agent only, what each neighbour is currently commanding
        (share_fn(own_id, other_id) -> (turn, dspd, give_way, dmin_nm, t_min_s)).
    """

    SHARE_KEYS = ("cmd_turn", "cmd_dspd", "give_way", "tcpa_cmd", "dcpa_cmd")

    def __init__(self, n, pos_norm=100_000.0, spd_norm=150.0, dist_norm=100.0,
                 tcpa_norm=600.0, dcpa_norm=50_000.0, intent=False, closest_fn=None,
                 share=False, share_fn=None):
        self.n = n
        self.pos_norm = pos_norm
        self.spd_norm = spd_norm
        self.dist_norm = dist_norm
        self.tcpa_norm = tcpa_norm
        self.dcpa_norm = dcpa_norm
        self.intent = bool(intent)
        self.closest_fn = closest_fn
        self.share = bool(share)
        self.share_fn = share_fn

    def space(self):
        box = lambda lo, hi: spaces.Box(lo, hi, shape=(self.n,), dtype=np.float64)
        out = {
            "x_r": box(-np.inf, np.inf),
            "y_r": box(-np.inf, np.inf),
            "vx_r": box(-np.inf, np.inf),
            "vy_r": box(-np.inf, np.inf),
            "cos_track": box(-1.0, 1.0),
            "sin_track": box(-1.0, 1.0),
            "intruder_distance": box(-np.inf, np.inf),
            "tcpa": box(-np.inf, np.inf),
            "dcpa": box(-np.inf, np.inf),
            "intruder_valid": box(0.0, 1.0),
        }
        if self.intent:
            out["tcpa_plan"] = box(-np.inf, np.inf)
            out["dcpa_plan"] = box(-np.inf, np.inf)
        if self.share:
            for k in self.SHARE_KEYS:
                out[k] = box(-np.inf, np.inf)
        return out

    def _rank(self, table):
        """Most urgent first: converging soon and close. Diverging traffic is
        ranked by range behind every converging one."""
        if not table:
            return []
        keys = []
        for _j, tcpa, dcpa, dist_nm in table:
            if not np.isfinite(tcpa):
                key = self.tcpa_norm + dist_nm
            else:
                key = min(tcpa, self.tcpa_norm) + dcpa / (self.dcpa_norm / self.tcpa_norm)
            keys.append(float(key))
        return [table[i] for i in np.argsort(keys)]

    def observe(self, ac_id):
        ac_idx = bs.traf.id2idx(ac_id)
        own_lat = bs.traf.lat[ac_idx]
        own_lon = bs.traf.lon[ac_idx]
        own_hdg = bs.traf.hdg[ac_idx]
        own_hdg_rad = np.deg2rad(own_hdg)
        own_tas = bs.traf.tas[ac_idx]
        vn_own = np.cos(own_hdg_rad) * own_tas
        ve_own = np.sin(own_hdg_rad) * own_tas

        order = self._rank(cpa_table(ac_idx))[:self.n]

        x_r = np.zeros(self.n)
        y_r = np.zeros(self.n)
        vx_r = np.zeros(self.n)
        vy_r = np.zeros(self.n)
        cos_track = np.zeros(self.n)
        sin_track = np.zeros(self.n)
        distance = np.full(self.n, self.dist_norm)
        tcpa_a = np.full(self.n, self.tcpa_norm)
        dcpa_a = np.full(self.n, self.dcpa_norm)
        valid = np.zeros(self.n)

        for slot, (j, tcpa, dcpa_m, dist_nm) in enumerate(order):
            brg, _ = bs.tools.geo.kwikqdrdist(own_lat, own_lon, bs.traf.lat[j], bs.traf.lon[j])
            dist_m = dist_nm * NM2KM * 1000.0

            int_hdg_rad = np.deg2rad(bs.traf.hdg[j])
            vn_r = np.cos(int_hdg_rad) * bs.traf.tas[j] - vn_own
            ve_r = np.sin(int_hdg_rad) * bs.traf.tas[j] - ve_own

            # body frame: x along our nose, y to starboard
            rel_rad = np.deg2rad(brg - own_hdg)
            x_r[slot] = dist_m * np.cos(rel_rad)
            y_r[slot] = dist_m * np.sin(rel_rad)
            vx_r[slot] = vn_r * np.cos(own_hdg_rad) + ve_r * np.sin(own_hdg_rad)
            vy_r[slot] = -vn_r * np.sin(own_hdg_rad) + ve_r * np.cos(own_hdg_rad)

            track = np.arctan2(vy_r[slot], vx_r[slot])
            cos_track[slot] = np.cos(track)
            sin_track[slot] = np.sin(track)
            distance[slot] = min(dist_nm, self.dist_norm)
            tcpa_a[slot] = self.tcpa_norm if not np.isfinite(tcpa) else min(tcpa, self.tcpa_norm)
            dcpa_a[slot] = min(dcpa_m, self.dcpa_norm)
            valid[slot] = 1.0

        extra = {}
        if self.intent:
            tcpa_p = np.full(self.n, self.tcpa_norm)
            dcpa_p = np.full(self.n, self.dcpa_norm)
            if self.closest_fn is not None:
                for slot, (j, _t, _d, dist_nm) in enumerate(order):
                    dmin, tmin = self.closest_fn(ac_id, bs.traf.id[j])
                    if not np.isfinite(dmin):
                        continue
                    if tmin <= 1e-6:            # closest now = diverging
                        tmin = self.tcpa_norm
                    tcpa_p[slot] = min(tmin, self.tcpa_norm)
                    dcpa_p[slot] = min(dmin * NM2KM * 1000.0, self.dcpa_norm)
            extra = {"tcpa_plan": tcpa_p / self.tcpa_norm,
                     "dcpa_plan": dcpa_p / self.dcpa_norm}
        if self.share:
            turn, dspd, give = np.zeros(self.n), np.zeros(self.n), np.zeros(self.n)
            tcpa_c = np.full(self.n, self.tcpa_norm)
            dcpa_c = np.full(self.n, self.dcpa_norm)
            if self.share_fn is not None:
                for slot, (j, _t, _d, _r) in enumerate(order):
                    t_, s_, g_, dmin, tmin = self.share_fn(ac_id, bs.traf.id[j])
                    turn[slot] = np.clip(t_, -2.0, 2.0)
                    dspd[slot] = np.clip(s_, -2.0, 2.0)
                    give[slot] = g_
                    if np.isfinite(dmin):
                        tcpa_c[slot] = self.tcpa_norm if tmin <= 1e-6 else min(tmin, self.tcpa_norm)
                        dcpa_c[slot] = min(dmin * NM2KM * 1000.0, self.dcpa_norm)
            extra.update({"cmd_turn": turn, "cmd_dspd": dspd, "give_way": give,
                          "tcpa_cmd": tcpa_c / self.tcpa_norm,
                          "dcpa_cmd": dcpa_c / self.dcpa_norm})

        return {**extra,
                "x_r": x_r / self.pos_norm,
                "y_r": y_r / self.pos_norm,
                "vx_r": vx_r / self.spd_norm,
                "vy_r": vy_r / self.spd_norm,
                "cos_track": cos_track,
                "sin_track": sin_track,
                "intruder_distance": distance / self.dist_norm,
                "tcpa": tcpa_a / self.tcpa_norm,
                "dcpa": dcpa_a / self.dcpa_norm,
                "intruder_valid": valid}


class ThreatObstacleObservation:
    """Obstacles as their true polygons: signed clearance, time to entry along
    the current track, closest miss, size, position, and a validity mask.
    Slots are ordered by threat (soonest entry, then clearance)."""

    def __init__(self, n, dist_norm=100.0, radius_norm=25.0, t_norm=600.0, polygons=None):
        self.n = n
        self.dist_norm = dist_norm
        self.radius_norm = radius_norm
        self.t_norm = t_norm
        # read at observe time: the base reset builds its first observation
        # before our own reset code runs
        self.polygons = polygons
        self.last_order = []

    def space(self):
        box = lambda lo, hi: spaces.Box(lo, hi, shape=(self.n,), dtype=np.float64)
        return {
            "obstacle_x": box(-np.inf, np.inf),
            "obstacle_y": box(-np.inf, np.inf),
            "obstacle_radius": box(0.0, np.inf),
            "obstacle_clearance": box(-np.inf, np.inf),
            "obstacle_t_enter": box(-np.inf, np.inf),
            "obstacle_miss": box(-np.inf, np.inf),
            "obstacle_inside": box(0.0, 1.0),
            "obstacle_valid": box(0.0, 1.0),
        }

    def _row(self, verts, own_lat, own_lon, own_hdg, spd_nm_s):
        A, B = sector_segments(own_lat, own_lon, verts)
        psi = np.deg2rad(own_hdg)
        u = np.array([np.cos(psi), np.sin(psi)])

        inside = point_in_polygon(A, B)
        edge_dist, _ = nearest_edge(A, B)
        clearance = -edge_dist if inside else edge_dist

        if inside:
            t_enter = 0.0
            miss = 0.0
        else:
            hit = ray_exit_distance(A, B, u)
            miss, along = ray_miss_distance(A, B, u)
            if np.isfinite(hit):
                t_enter = hit / spd_nm_s
            else:
                t_enter = min(along / spd_nm_s, self.t_norm)

        area = 0.5 * abs(float(np.sum(A[:, 0] * B[:, 1] - B[:, 0] * A[:, 1])))
        size = float(np.sqrt(area / np.pi))          # area-equivalent radius

        centroid = A.mean(axis=0)
        x_body = float(centroid @ u)
        y_body = float(-centroid[0] * u[1] + centroid[1] * u[0])
        return float(t_enter), float(clearance), float(miss), size, x_body, y_body, bool(inside)

    def observe(self, ac_id, *_unused):
        ac_idx = bs.traf.id2idx(ac_id)
        own_lat = bs.traf.lat[ac_idx]
        own_lon = bs.traf.lon[ac_idx]
        own_hdg = bs.traf.hdg[ac_idx]
        spd_nm_s = max(bs.traf.tas[ac_idx] / (NM2KM * 1000.0), 1e-6)

        polys = self.polygons() if self.polygons is not None else []
        rows = [(k,) + self._row(verts, own_lat, own_lon, own_hdg, spd_nm_s)
                for k, verts in enumerate(polys)]
        rows.sort(key=lambda r: (min(r[1], self.t_norm), r[2]))
        order = rows[:self.n]
        self.last_order = [r[0] for r in order]

        x = np.zeros(self.n)
        y = np.zeros(self.n)
        radius_a = np.zeros(self.n)
        clearance_a = np.full(self.n, self.dist_norm)
        t_enter_a = np.full(self.n, self.t_norm)
        miss_a = np.full(self.n, self.dist_norm)
        inside_a = np.zeros(self.n)
        valid = np.zeros(self.n)

        for slot, (k, t_enter, clearance, miss, size, px, py, inside) in enumerate(order):
            x[slot] = px
            y[slot] = py
            radius_a[slot] = size
            clearance_a[slot] = np.clip(clearance, -self.dist_norm, self.dist_norm)
            t_enter_a[slot] = min(t_enter, self.t_norm)
            miss_a[slot] = np.clip(miss, -self.dist_norm, self.dist_norm)
            inside_a[slot] = 1.0 if inside else 0.0
            valid[slot] = 1.0

        return {
            "obstacle_x": x / self.dist_norm,
            "obstacle_y": y / self.dist_norm,
            "obstacle_radius": radius_a / self.radius_norm,
            "obstacle_clearance": clearance_a / self.dist_norm,
            "obstacle_t_enter": t_enter_a / self.t_norm,
            "obstacle_miss": miss_a / self.dist_norm,
            "obstacle_inside": inside_a,
            "obstacle_valid": valid,
        }


class ThreatBoundaryObservation:
    """The base twelve perimeter points plus signed margin to the sector edge,
    time to exit along the current track and direction to the nearest edge."""

    def __init__(self, n=12, distance_norm=170.0, margin_norm=50.0, t_norm=600.0):
        self.n = n
        self.distance_norm = distance_norm
        self.margin_norm = margin_norm
        self.t_norm = t_norm

    def space(self):
        box = lambda lo, hi, k: spaces.Box(lo, hi, shape=(k,), dtype=np.float64)
        return {
            "sector_point_distance": box(-np.inf, np.inf, self.n),
            "sector_point_cos_bearing": box(-1.0, 1.0, self.n),
            "sector_point_sin_bearing": box(-1.0, 1.0, self.n),
            "inside_sector": box(0.0, 1.0, 1),
            "sector_margin": box(-np.inf, np.inf, 1),
            "sector_t_exit": box(-np.inf, np.inf, 1),
            "sector_near_cos": box(-1.0, 1.0, 1),
            "sector_near_sin": box(-1.0, 1.0, 1),
        }

    def observe(self, ac_id, sector_vertices, inside):
        ac_idx = bs.traf.id2idx(ac_id)
        own_lat = bs.traf.lat[ac_idx]
        own_lon = bs.traf.lon[ac_idx]
        own_hdg = bs.traf.hdg[ac_idx]
        psi = np.deg2rad(own_hdg)

        boundary = resample_perimeter(sector_vertices, self.n)
        distances = np.zeros(self.n)
        cos_bearing = np.zeros(self.n)
        sin_bearing = np.zeros(self.n)
        for i in range(self.n):
            qdr, dist = bs.tools.geo.kwikqdrdist(own_lat, own_lon, boundary[i, 0], boundary[i, 1])
            rel = np.deg2rad(_bound_angle(own_hdg - qdr))
            distances[i] = dist * NM2KM
            cos_bearing[i] = np.cos(rel)
            sin_bearing[i] = np.sin(rel)

        A, B = sector_segments(own_lat, own_lon, sector_vertices)
        u = np.array([np.cos(psi), np.sin(psi)])
        edge_nm, direction = nearest_edge(A, B)
        margin = edge_nm if inside else -edge_nm

        spd_nm_s = max(bs.traf.tas[ac_idx] / (NM2KM * 1000.0), 1e-6)
        exit_nm = ray_exit_distance(A, B, u)
        if not inside:
            t_exit = 0.0
        elif np.isfinite(exit_nm):
            t_exit = exit_nm / spd_nm_s
        else:
            t_exit = self.t_norm

        near_cos = direction[0] * np.cos(psi) + direction[1] * np.sin(psi)
        near_sin = -direction[0] * np.sin(psi) + direction[1] * np.cos(psi)

        return {
            "sector_point_distance": distances / self.distance_norm,
            "sector_point_cos_bearing": cos_bearing,
            "sector_point_sin_bearing": sin_bearing,
            "inside_sector": np.array([1.0 if inside else 0.0]),
            "sector_margin": np.array(
                [np.clip(margin, -self.margin_norm, self.margin_norm) / self.margin_norm]),
            "sector_t_exit": np.array([min(t_exit, self.t_norm) / self.t_norm]),
            "sector_near_cos": np.array([near_cos]),
            "sector_near_sin": np.array([near_sin]),
        }
