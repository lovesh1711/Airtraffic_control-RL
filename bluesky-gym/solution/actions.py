"""Action mapping: heading relative to a route anchor, speed relative to nominal.

The shipped heading action is relative to the *current* heading, so a=0 keeps
whatever deviation the aircraft has. Here a0 is an offset from the bearing to
an anchor (the goal, or the next waypoint of a planned route around the
obstacles), so a0=0 always means "fly the sensible default" and every
deviation is temporary. a1 picks an absolute target speed around nominal.
"""

import numpy as np
import bluesky as bs

from core.tools import kwikqdrdist


class GoalReferencedHeadingAction:
    """HDG = bearing to anchor + a0 * d_heading.

    turn_guard_deg: if the anchor is further than this off the nose, keep
    turning the short way towards it. Without it a +-1 policy can alternate
    between commands on either side of the reciprocal and never finish a turn.
    """

    def __init__(self, d_heading=45.0, turn_guard_deg=None):
        self.d_heading = d_heading
        self.turn_guard_deg = turn_guard_deg

    def execute(self, ac_id, action, goal_lat, goal_lon):
        idx = bs.traf.id2idx(ac_id)
        if idx < 0:
            return
        qdr, _ = kwikqdrdist(bs.traf.lat[idx], bs.traf.lon[idx], goal_lat, goal_lon)
        new_hdg = (qdr + float(action) * self.d_heading) % 360.0
        if self.turn_guard_deg is not None and np.isfinite(qdr):
            hdg = float(bs.traf.hdg[idx])
            off = (qdr - hdg + 180.0) % 360.0 - 180.0
            if abs(off) > self.turn_guard_deg:
                new_hdg = (hdg + np.sign(off) * self.turn_guard_deg) % 360.0
        if not np.isfinite(new_hdg):
            # BlueSky drops "HDG ... nan", so fall back to the anchor bearing
            new_hdg = qdr % 360.0
            if not np.isfinite(new_hdg):
                return
        if new_hdg < 1e-4:
            new_hdg = 0.0       # "1e-05" is not accepted by the HDG command either
        bs.stack.stack(f"HDG {ac_id} {new_hdg}")


class AnchoredSpeedAction:
    """Absolute target speed: a1=0 nominal, +1 speed_max_kt, -1 speed_min_kt.

    The shipped speed action adds a1 * 6.67 kt to the current speed, so the
    policy ends up pinned at the envelope limit; a target speed can be held.
    """

    MPS2KT = 1.0 / 0.514444

    def __init__(self, nominal_kt, speed_min_kt=200.0, speed_max_kt=316.8):
        if not speed_min_kt < nominal_kt <= speed_max_kt:
            raise ValueError(f"need speed_min_kt < nominal ({nominal_kt:.1f}) "
                             f"<= speed_max_kt, got {speed_min_kt}, {speed_max_kt}")
        self.nominal_kt = float(nominal_kt)
        self.speed_min_kt = float(speed_min_kt)
        self.speed_max_kt = float(speed_max_kt)

    def target_kt(self, a1):
        a1 = float(np.clip(a1, -1.0, 1.0)) if np.isfinite(a1) else 0.0
        if a1 >= 0.0:
            return self.nominal_kt + a1 * (self.speed_max_kt - self.nominal_kt)
        return self.nominal_kt + a1 * (self.nominal_kt - self.speed_min_kt)

    def execute(self, ac_id, action):
        if bs.traf.id2idx(ac_id) < 0:
            return
        bs.stack.stack(f"SPD {ac_id} {self.target_kt(action):.2f}")


class GoalReferencedActionMixin:
    """Replaces _get_action. Mix in before the env class; needs _goal_latlon(ac_id)."""

    # plan-based closest approach is solved on this grid (exact between samples)
    PLAN_TIMES = np.arange(0.0, 605.0, 10.0)

    def _configure_actions(self, d_heading=None, route_mode="goal",
                           route_clearance_nm=2.5, route_switch_nm=4.0,
                           route_budget_frac=0.8, route_budget_hard_frac=None,
                           route_least_penetration=False, route_lateral_offset_nm=0.0,
                           speed_mode="relative", speed_min_kt=200.0, speed_max_kt=316.8,
                           turn_guard_deg=None):
        self.goal_heading_action = GoalReferencedHeadingAction(
            d_heading=self.d_heading if d_heading is None else d_heading,
            turn_guard_deg=turn_guard_deg)
        if speed_mode not in ("relative", "anchored"):
            raise ValueError(f"speed_mode must be 'relative' or 'anchored', got {speed_mode!r}")
        self.speed_mode = speed_mode
        if speed_mode == "anchored":
            self.speed_action = AnchoredSpeedAction(
                nominal_kt=self.ac_spd * AnchoredSpeedAction.MPS2KT,
                speed_min_kt=speed_min_kt, speed_max_kt=speed_max_kt)
        self._action_now = {}
        self._action_prev = {}
        self._plan_epoch, self._plan_key = 0, None
        self._plan_traj, self._plan_pair = {}, {}

        if route_mode not in ("goal", "visibility"):
            raise ValueError(f"route_mode must be 'goal' or 'visibility', got {route_mode!r}")
        self.route_mode = route_mode
        self.route_switch_nm = float(route_switch_nm)
        self._route = {}
        self._route_i = {}
        self._route_pull = {}
        self._planner = None
        if route_mode == "visibility":
            from solution.route import RoutePlanner
            # route length budgets as fractions of the distance flyable in the
            # time limit (ac_spd is in m/s)
            self._planner = RoutePlanner(
                clearance_nm=route_clearance_nm,
                budget_nm=route_budget_frac * self.ac_spd * self.episode_time_limit / 1852.0,
                budget_hard_nm=(None if route_budget_hard_frac is None else
                                route_budget_hard_frac * self.ac_spd
                                * self.episode_time_limit / 1852.0),
                least_penetration=route_least_penetration,
                lateral_offset_nm=route_lateral_offset_nm)

    def _anchor_latlon(self, ac_id):
        if self.route_mode == "goal" or self._planner is None:
            return self._goal_latlon(ac_id)
        return self._route_anchor(ac_id)

    def _anchor_peek(self, ac_id):
        """Current anchor without advancing the route (for observations)."""
        if self.route_mode == "goal" or self._planner is None:
            return self._goal_latlon(ac_id)
        wps = self._route.get(ac_id)
        if wps is None or not len(wps):
            return self._goal_latlon(ac_id)
        i = min(self._route_i.get(ac_id, 0), len(wps) - 1)
        return float(wps[i][0]), float(wps[i][1])

    def _anchor_qdr(self, ac_id):
        idx = bs.traf.id2idx(ac_id)
        if idx < 0:
            return float("nan")
        alat, alon = self._anchor_peek(ac_id)
        qdr, _ = kwikqdrdist(bs.traf.lat[idx], bs.traf.lon[idx], alat, alon)
        return float(qdr)

    def _intent_plan(self, acid):
        """(points, tas_m_s, terminal): where acid will fly if nobody deviates.

        Our aircraft follow their remaining route and disappear at the goal;
        scripted intruders follow their remaining waypoints and keep going.
        """
        j = bs.traf.id2idx(acid)
        if j < 0:
            return None
        pos = (float(bs.traf.lat[j]), float(bs.traf.lon[j]))
        tas = float(bs.traf.tas[j])
        controlled = getattr(self, "possible_agents", None) or [self.agent]
        if acid in controlled:
            wps = self._route.get(acid) if self._planner is not None else None
            if wps is not None and len(wps):
                i = min(self._route_i.get(acid, 0), len(wps) - 1)
                rest = [tuple(map(float, w)) for w in wps[i:]]
            else:
                rest = [tuple(map(float, self._goal_latlon(acid)))]
            return [pos] + rest, tas, True
        routes = {r.ac_id: r.waypoints[1:] for r in getattr(self.scenario, "intruder_routes", [])}
        legs = routes.get(acid)
        if legs is None:
            return [pos], tas, False
        rt = bs.traf.ap.route[j]
        if rt.nwp > 0:
            if not bs.traf.swlnav[j]:
                rest = []
            else:
                rest = [(float(a), float(b)) for a, b in
                        zip(rt.wplat[max(rt.iactwp, 0):], rt.wplon[max(rt.iactwp, 0):])]
        else:
            # waypoints are only added on the first sim step
            rest = [tuple(map(float, w)) for w in legs]
        if not rest:
            hdg = np.radians(float(bs.traf.hdg[j]))
            ahead = (pos[0] + np.cos(hdg) / 60.0,
                     pos[1] + np.sin(hdg) / (60.0 * np.cos(np.radians(pos[0]))))
            return [pos, ahead], tas, False
        return [pos] + rest, tas, False

    def _route_progress(self, ac_id):
        """(anchor, next waypoint or None, remaining route NM) for RouteObservation."""
        i = bs.traf.id2idx(ac_id)
        pos = (float(bs.traf.lat[i]), float(bs.traf.lon[i]))
        goal = tuple(map(float, self._goal_latlon(ac_id)))
        wps = self._route.get(ac_id) if self._planner is not None else None
        if wps is None or not len(wps):
            return goal, None, float(kwikqdrdist(*pos, *goal)[1])
        k = min(self._route_i.get(ac_id, 0), len(wps) - 1)
        pts = [pos] + [tuple(map(float, w)) for w in wps[k:]]
        remaining = sum(float(kwikqdrdist(*a, *b)[1]) for a, b in zip(pts[:-1], pts[1:]))
        nxt = pts[2] if len(pts) > 2 else None
        return pts[1], nxt, remaining

    def _time_slack(self, ac_id, remaining_nm):
        left = self.episode_time_limit - self.metrics[ac_id]["flight_time"]
        return left - remaining_nm * 1852.0 / self.ac_spd

    def _plan_closest(self, own, other):
        """(dmin_nm, t_at_min) if both aircraft fly their plans, cached per sim second."""
        from solution.observations import _plan_positions, _closest_along
        key = (self._plan_epoch, float(bs.sim.simt))
        if key != self._plan_key:
            self._plan_key, self._plan_traj, self._plan_pair = key, {}, {}
        pair = self._plan_pair.get((own, other))
        if pair is not None:
            return pair
        trajs = []
        for acid in (own, other):
            tr = self._plan_traj.get((own, acid))
            if tr is None:
                plan = self._intent_plan(acid)
                i = bs.traf.id2idx(own)
                if plan is None or i < 0:
                    self._plan_pair[(own, other)] = (np.inf, np.inf)
                    return np.inf, np.inf
                tr = _plan_positions(plan[0], plan[1] / 1852.0, plan[2], self.PLAN_TIMES,
                                     float(bs.traf.lat[i]), float(bs.traf.lon[i]))
                self._plan_traj[(own, acid)] = tr
            trajs.append(tr)
        out = _closest_along(trajs[0], trajs[1], self.PLAN_TIMES)
        self._plan_pair[(own, other)] = out
        return out

    def _reset_routes(self):
        """Plan one route per aircraft (call from reset)."""
        self._plan_epoch += 1
        self._route = {}
        self._route_i = {}
        self._route_pull = {}
        if self._planner is None:
            return
        sector = self._sector_vertices()
        if sector is None or len(sector) < 3:
            return
        ref = tuple(np.mean(np.asarray(sector, dtype=float), axis=0))
        self._planner.reset(self._obstacle_polygons(), sector, ref)
        for ac_id in self._route_agents():
            idx = bs.traf.id2idx(ac_id)
            if idx < 0:
                continue
            wps = self._planner.plan((bs.traf.lat[idx], bs.traf.lon[idx]), self._goal_latlon(ac_id))
            if len(wps):
                self._route[ac_id] = wps
                self._route_i[ac_id] = 0
                # an offset route must not shortcut back across the centreline
                # into the opposite lane, so skip waypoints only at the clearance
                # the centreline was planned with
                self._route_pull[ac_id] = (self._planner.last_centreline_nm
                                           if self._planner.last_offset_nm else None)

    def _route_agents(self):
        return [self.agent]

    def _route_anchor(self, ac_id):
        """Next route waypoint. Skips ahead when a later waypoint is already in
        clear view, or when the current one is within route_switch_nm."""
        goal = self._goal_latlon(ac_id)
        wps = self._route.get(ac_id)
        if wps is None or not len(wps):
            return goal
        idx = bs.traf.id2idx(ac_id)
        if idx < 0:
            return goal
        pos = (bs.traf.lat[idx], bs.traf.lon[idx])

        i = self._route_i.get(ac_id, 0)
        last = len(wps) - 1
        pull = self._route_pull.get(ac_id)
        while i < last and self._planner.visible(pos, wps[i + 1], pull):
            i += 1
        while i < last and kwikqdrdist(pos[0], pos[1], wps[i][0], wps[i][1])[1] < self.route_switch_nm:
            i += 1
        self._route_i[ac_id] = i
        return wps[i]

    def _get_action(self, ac_id, action):
        a = np.asarray(action, dtype=float).copy()
        self._action_prev[ac_id] = self._action_now.get(ac_id, np.zeros_like(a))
        self._action_now[ac_id] = a

        anchor_lat, anchor_lon = self._anchor_latlon(ac_id)
        self.goal_heading_action.execute(ac_id, a[0], anchor_lat, anchor_lon)
        self.speed_action.execute(ac_id, a[1])
