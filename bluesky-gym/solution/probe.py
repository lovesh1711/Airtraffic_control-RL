"""Manoeuvre probe: what each candidate command would do over the next 240 s.

For 27 candidates (9 heading offsets x 3 speeds) we fly the aircraft forward
with BlueSky's own 1 s kinematics (speed ramp, turn rate from bank angle,
position update), against the predicted tracks of the other aircraft, and
report per candidate: closest approach, time to and seconds of loss of
separation, of restricted-area entry and of sector exit. The policy sees the
candidates as a set (own branch in solution/encoder.py) and still chooses the
action itself.

Scripted intruders follow their remaining waypoints; our other aircraft hold
their current command and disappear at their goal.
"""

import numpy as np
from gymnasium import spaces
from matplotlib.path import Path

import bluesky as bs
from bluesky.tools.aero import Rearth, g0, kts, vcas2tas

from solution.observations import _plan_positions

NM = 1852.0


DEG_NM = np.radians(1.0) * Rearth / NM          # NM per degree, BlueSky's Earth


def flat_nm(lat_a, lon_a, lat_b, lon_b):
    """kwikqdrdist's distance in NM (the formula the metrics use), broadcasting."""
    dlat = np.radians(lat_b - lat_a)
    dlon = np.radians(((lon_b - lon_a) + 180.0) % 360.0 - 180.0)
    c = np.cos(np.radians(lat_a + lat_b) * 0.5)
    return Rearth * np.sqrt(dlat * dlat + dlon * dlon * c * c) / NM


def near_nm2(lat_a, lon_a, lat_b, lon_b, cos_a):
    """Squared flat_nm with the cosine taken at a's latitude (much cheaper;
    under 0.1% different within the 5 NM that matters)."""
    dlat = lat_b - lat_a
    dlon = (lon_b - lon_a) * cos_a
    return (dlat * dlat + dlon * dlon) * (DEG_NM * DEG_NM)


def kinematic_latlon(lat0, lon0, hdg0, tas0, hdg_cmd, tas_cmd, axmax, bank_rad, n,
                     eps=0.01):
    """Positions under steady (heading, TAS) commands, stepping like BlueSky.

    Each of BlueSky's per-second update rules has a closed form: speed ramps
    at axmax, heading turns at g*tan(bank)/tas using the updated speed, and
    position integrates lat then lon. hdg_cmd and tas_cmd are (C,); returns
    (lat, lon), each (C, n + 1) for t = 0..n s.
    """
    hdg_cmd = np.atleast_1d(np.asarray(hdg_cmd, dtype=float))
    tas_cmd = np.atleast_1d(np.asarray(tas_cmd, dtype=float))
    k = np.arange(1, n + 1, dtype=float)[None, :]
    dv = (tas_cmd - tas0)[:, None]
    tas = tas0 + np.sign(dv) * np.minimum(axmax * k, np.abs(dv))            # (C, n)
    rate = np.degrees(g0 * np.tan(bank_rad) / np.maximum(tas, eps))
    dh = ((hdg_cmd - hdg0 + 180.0) % 360.0 - 180.0)[:, None]
    hdg = hdg0 + np.sign(dh) * np.minimum(np.cumsum(rate, axis=1), np.abs(dh))
    r = np.radians(hdg)
    lat = lat0 + np.cumsum(np.degrees(tas * np.cos(r) / Rearth), axis=1)
    lon = lon0 + np.cumsum(np.degrees(tas * np.sin(r) / np.cos(np.radians(lat)) / Rearth),
                           axis=1)
    C = hdg_cmd.shape[0]
    return (np.concatenate((np.full((C, 1), lat0), lat), axis=1),
            np.concatenate((np.full((C, 1), lon0), lon), axis=1))


def _first(mask, n):
    """First True column per row, ``n`` where none."""
    return np.where(mask.any(axis=1), mask.argmax(axis=1), n)


class ManoeuvreProbeObservation:
    """Predicted consequences of each candidate command.

    n_headings odd so a0 = 0 is a candidate; speeds are the a1 values paired
    with every heading; containment is tested every geo_every seconds.
    """

    FEATURES = ("probe_a0", "probe_a1", "probe_dmin", "probe_t_los", "probe_los",
                "probe_t_obstacle", "probe_obstacle", "probe_t_exit", "probe_outside")
    MASK_KEY = "probe_valid"

    def __init__(self, n_headings=9, speeds=(-1.0, 0.0, 1.0), d_heading=45.0,
                 horizon_s=240, los_nm=5.0, dmin_norm=10.0, geo_every=2):
        if n_headings % 2 == 0:
            raise ValueError(f"n_headings must be odd, got {n_headings}")
        self.n_headings = int(n_headings)
        self.speeds = tuple(float(s) for s in speeds)
        self.d_heading = float(d_heading)
        self.horizon = int(horizon_s)
        self.los_nm = float(los_nm)
        self.dmin_norm = float(dmin_norm)
        self.geo_every = int(geo_every)
        a0, a1 = np.meshgrid(np.linspace(-1.0, 1.0, self.n_headings),
                             np.asarray(self.speeds), indexing="ij")
        self.a0, self.a1 = a0.ravel(), a1.ravel()
        self.n = self.a0.shape[0]

    def space(self):
        shape = (self.n,)
        out = {k: spaces.Box(-np.inf, np.inf, shape=shape, dtype=np.float64)
               for k in self.FEATURES}
        out[self.MASK_KEY] = spaces.Box(0.0, 1.0, shape=shape, dtype=np.float64)
        return out

    def benign(self):
        """No prediction available (aircraft already removed): nothing ahead."""
        one = np.ones(self.n)
        return {"probe_a0": self.a0.copy(), "probe_a1": self.a1.copy(),
                "probe_dmin": one * 3.0, "probe_t_los": one.copy(), "probe_los": one * 0.0,
                "probe_t_obstacle": one.copy(), "probe_obstacle": one * 0.0,
                "probe_t_exit": one.copy(), "probe_outside": one * 0.0,
                self.MASK_KEY: one * 0.0}

    def commands(self, hdg, anchor_qdr, turn_guard_deg, target_kt):
        """``(heading_deg, cas_kt)`` per candidate, as the action would command."""
        h = (anchor_qdr + self.a0 * self.d_heading) % 360.0
        if turn_guard_deg is not None and np.isfinite(anchor_qdr):
            off = (anchor_qdr - hdg + 180.0) % 360.0 - 180.0
            if abs(off) > turn_guard_deg:
                h = np.full_like(h, (hdg + np.sign(off) * turn_guard_deg) % 360.0)
        return h, np.array([target_kt(a) for a in self.a1])

    def observe(self, own, traffic, obstacles, sector, goal=None, capture_nm=None):
        """own: lat, lon, hdg, tas, alt, axmax, bank, hdg_cmd (C,), cas_cmd_kt (C,).
        traffic: (lat, lon), each (n + 1, J), NaN once an aircraft is gone.
        Nothing after the ownship reaches its goal counts."""
        n = self.horizon
        tas_cmd = vcas2tas(own["cas_cmd_kt"] * kts, own["alt"])
        lat, lon = kinematic_latlon(own["lat"], own["lon"], own["hdg"], own["tas"],
                                    own["hdg_cmd"], tas_cmd, own["axmax"], own["bank"], n)
        alive = np.ones(lat.shape, dtype=bool)
        if goal is not None:
            cap = flat_nm(lat, lon, goal[0], goal[1]) < capture_nm
            k = _first(cap[:, 1:], n)                       # capture at step k + 1
            alive = np.arange(n + 1)[None, :] <= (k + 1)[:, None]

        # --- traffic -------------------------------------------------------
        tl, tn = traffic
        if tl.size:
            la, lo = lat[:, 1:, None], lon[:, 1:, None]
            d2 = near_nm2(la, lo, tl[None, 1:], tn[None, 1:], np.cos(np.radians(la)))  # (C, T, J)
            d2 = np.where(np.isfinite(d2) & alive[:, 1:, None], d2, np.inf)
            dmin = np.sqrt(d2.min(axis=(1, 2)))
            los = d2 < self.los_nm * self.los_nm
            t_los = _first(los.any(axis=2), n)
            los_s = los.sum(axis=(1, 2))
        else:
            dmin = np.full(self.n, np.inf)
            t_los = np.full(self.n, n)
            los_s = np.zeros(self.n)

        # --- geometry: the metric's own containment test on lat/lon ---------
        g = slice(0, n + 1, self.geo_every)
        pts = np.column_stack((lat[:, g].ravel(), lon[:, g].ravel()))
        shape = lat[:, g].shape
        inside = np.zeros(shape, dtype=bool)
        for poly in obstacles or []:
            if poly is not None and len(poly) >= 3:
                inside |= Path(np.asarray(poly, dtype=float)).contains_points(pts).reshape(shape)
        if sector is not None and len(sector) >= 3:
            outside = ~Path(np.asarray(sector, dtype=float)).contains_points(pts).reshape(shape)
        else:
            outside = np.zeros(shape, dtype=bool)
        live_g = alive[:, g]
        inside &= live_g
        outside &= live_g
        m = shape[1]

        h = float(n)
        return {
            "probe_a0": self.a0.copy(),
            "probe_a1": self.a1.copy(),
            "probe_dmin": np.clip(dmin / self.dmin_norm, 0.0, 3.0),
            "probe_t_los": (t_los + 1.0 * (t_los < n)) / h,
            "probe_los": np.minimum(los_s / h, 3.0),
            "probe_t_obstacle": np.minimum(_first(inside, m) * self.geo_every / h, 1.0),
            "probe_obstacle": inside.sum(axis=1) * self.geo_every / h,
            "probe_t_exit": np.minimum(_first(outside, m) * self.geo_every / h, 1.0),
            "probe_outside": outside.sum(axis=1) * self.geo_every / h,
            self.MASK_KEY: np.ones(self.n),
        }


class ManoeuvreProbeMixin:
    """Env side of the probe and of the multi-agent shared-intent features.

    Each aircraft's predicted future is cached for the current sim second, so
    in multi-agent it is built once and shared by every ownship.
    """

    def _configure_probe(self, probe=False, share_intent=False):
        self.probe_obs = (ManoeuvreProbeObservation(d_heading=self.d_heading,
                                                    los_nm=self.intrusion_distance)
                          if probe else None)
        self.share_intent = bool(share_intent)
        if self.probe_obs is not None and getattr(self, "speed_mode", None) != "anchored":
            raise ValueError("the manoeuvre probe needs speed_mode='anchored': its "
                             "speed candidates are the targets a1 selects")
        self._future_key, self._futures = None, {}

    # -- futures ------------------------------------------------------------
    def _controlled(self):
        return getattr(self, "possible_agents", None) or [self.agent]

    def _capture_nm(self):
        return self.distance_margin / 1.852            # km -> NM

    def _future(self, acid, n):
        """(lat, lon), each (n + 1,): where acid will be, NaN once it is gone."""
        key = (self._plan_epoch, float(bs.sim.simt))
        if key != self._future_key:
            self._future_key, self._futures = key, {}
        hit = self._futures.get((acid, n))
        if hit is not None:
            return hit
        j = bs.traf.id2idx(acid)
        if j < 0:
            out = (np.full(n + 1, np.nan), np.full(n + 1, np.nan))
        elif acid in self._controlled():
            lat, lon = kinematic_latlon(
                float(bs.traf.lat[j]), float(bs.traf.lon[j]), float(bs.traf.hdg[j]),
                float(bs.traf.tas[j]), bs.traf.aporasas.hdg[j], bs.traf.aporasas.tas[j],
                float(bs.traf.perf.axmax[j]), self._bank(j), n)
            lat, lon = lat[0].copy(), lon[0].copy()
            g = self._goal_latlon(acid)
            cap = flat_nm(lat, lon, g[0], g[1]) < self._capture_nm()
            if cap[1:].any():
                k = int(np.argmax(cap[1:])) + 2           # gone after the capture step
                lat[k:], lon[k:] = np.nan, np.nan
            out = (lat, lon)
        else:
            plan = self._intent_plan(acid)
            ref_lat, ref_lon = float(bs.traf.lat[j]), float(bs.traf.lon[j])
            xy = _plan_positions(plan[0], plan[1] / NM, plan[2],
                                 np.arange(n + 1, dtype=float), ref_lat, ref_lon)
            out = (ref_lat + xy[:, 0] / 60.0,
                   ref_lon + xy[:, 1] / (60.0 * np.cos(np.radians(ref_lat))))
        self._futures[(acid, n)] = out
        return out

    @staticmethod
    def _bank(j):
        phi = float(bs.traf.ap.turnphi[j])
        eps = float(bs.traf.eps[j])
        return phi if phi > eps * eps else float(bs.traf.ap.bankdef[j])

    def _traffic_futures(self, own, n):
        others = [a for a in bs.traf.id if a != own]
        if not others:
            return np.empty((n + 1, 0)), np.empty((n + 1, 0))
        f = [self._future(a, n) for a in others]
        return (np.column_stack([x[0] for x in f]), np.column_stack([x[1] for x in f]))

    # -- the probe ------------------------------------------------------------
    def _probe_observe(self, ac_id):
        p = self.probe_obs
        i = bs.traf.id2idx(ac_id)
        qdr = self._anchor_qdr(ac_id)
        if i < 0 or not np.isfinite(qdr):
            return p.benign()
        hdg = float(bs.traf.hdg[i])
        h_cmd, cas_cmd = p.commands(hdg, qdr, self.goal_heading_action.turn_guard_deg,
                                    self.speed_action.target_kt)
        own = dict(lat=float(bs.traf.lat[i]), lon=float(bs.traf.lon[i]), hdg=hdg,
                   tas=float(bs.traf.tas[i]), alt=float(bs.traf.alt[i]),
                   axmax=float(bs.traf.perf.axmax[i]), bank=self._bank(i),
                   hdg_cmd=h_cmd, cas_cmd_kt=cas_cmd)
        return p.observe(own, self._traffic_futures(ac_id, p.horizon),
                         self._obstacle_polygons(), self._sector_vertices(),
                         goal=self._goal_latlon(ac_id), capture_nm=self._capture_nm())

    # -- shared intent (multi-agent) --------------------------------------------
    SHARE_HORIZON = 300

    def _intent_row(self, own, other):
        """(cmd_turn, cmd_dspd, give_way, dmin_nm, t_min_s) for one neighbour.

        cmd_turn/cmd_dspd: how the neighbour is already turning / changing speed.
        give_way: difference in time slack, so the aircraft with more time to
        spare yields. dmin, t_min: closest approach if both hold their commands.
        """
        if bs.traf.id2idx(own) < 0 or bs.traf.id2idx(other) < 0:
            return 0.0, 0.0, 0.0, np.inf, np.inf
        return self._intent_table(own).get(other, (0.0, 0.0, 0.0, np.inf, np.inf))

    def _intent_table(self, own):
        """_intent_row for every neighbour of own, in one vectorised pass."""
        key = ("intent", own)
        self._slack_of(own)                           # refreshes the per-second cache
        hit = self._futures.get(key)
        if hit is not None:
            return hit
        n = self.SHARE_HORIZON
        others = [a for a in bs.traf.id if a != own]
        out = {}
        if others:
            j = np.asarray(bs.traf.id2idx(others))
            turn = ((bs.traf.aporasas.hdg[j] - bs.traf.hdg[j] + 180.0) % 360.0 - 180.0) / 90.0
            dspd = (bs.traf.aporasas.tas[j] - bs.traf.tas[j]) / (60.0 * kts)
            la, lo = self._future(own, n)
            tl, tn = self._traffic_futures(own, n)            # (n + 1, J), same order
            d2 = near_nm2(la[:, None], lo[:, None], tl, tn, np.cos(np.radians(la))[:, None])
            d2 = np.where(np.isfinite(d2), d2, np.inf)
            k = np.argmin(d2, axis=0)
            dmin = np.sqrt(d2[k, np.arange(len(others))])
            mine = self._slack_of(own)
            controlled = self._controlled()
            for c, a in enumerate(others):
                give = (float(np.clip((mine - self._slack_of(a)) / 1000.0, -1.0, 1.0))
                        if a in controlled else 0.0)
                out[a] = (float(turn[c]), float(dspd[c]), give, float(dmin[c]), float(k[c]))
        self._futures[key] = out
        return out

    def _slack_of(self, acid):
        key = ("slack", acid)
        fk = (self._plan_epoch, float(bs.sim.simt))
        if fk != self._future_key:
            self._future_key, self._futures = fk, {}
        if key not in self._futures:
            _, _, remaining = self._route_progress(acid)
            self._futures[key] = float(self._time_slack(acid, remaining))
        return self._futures[key]
