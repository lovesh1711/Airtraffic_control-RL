"""Shaped training reward, shared by the single- and multi-agent envs.

The scored metrics come from the env's own _update_metrics / _get_info, which
are never touched here; this is only the training signal.

Besides the base per-second penalties it has:
- potential-based progress towards the goal,
- separation, obstacle and boundary costs that fire *before* a violation
  (current proximity plus predicted closest approach along the plans),
- a steep barrier just outside 5 NM,
- one-off charges when an intrusion, restricted-area entry or sector exit starts,
- a penalty for timing out without reaching the goal,
- small costs on heading deviation and on changing it.
"""

import numpy as np
import bluesky as bs
import bluesky_gym.envs.common.functions as fn
from shapely.geometry import Polygon, Point

from core.tools import kwikqdrdist
from solution.geometry import (cpa_table, clear as clear_cpa_cache, sector_segments,
                               ray_exit_distance, nearest_edge, point_in_polygon,
                               ray_miss_distance)

NM2KM = 1.852
# decision interval (s) the per-step weights are tuned for
REFERENCE_ACTION_FREQUENCY = 10.0


class ShapedRewardMixin:
    """Needs _goal_latlon(ac_id), _baseline_of(ac_id) and _obstacle_polygons()."""

    def _configure_shaping(self, w_progress=0.5, w_separation=-1.2,
                           w_obstacle=-1.20, w_boundary=-1.10, w_step=-0.08,
                           w_action=-0.40, w_smooth=-0.25,
                           warn_nm=12.0, tcpa_horizon=300.0,
                           obstacle_clearance_nm=8.0, boundary_margin_nm=10.0,
                           w_miss=0.0, w_intrusion_event=0.0,
                           w_restricted_event=0.0, w_exit_event=0.0,
                           w_sep_barrier=0.0, sep_barrier_nm=6.5,
                           separation_mode="straight"):
        # "straight": predicted conflicts assume both aircraft hold their
        # velocity; "plan": both fly their plans (route / scripted waypoints)
        if separation_mode not in ("straight", "plan"):
            raise ValueError(f"separation_mode must be 'straight' or 'plan', got {separation_mode!r}")
        self.separation_mode = separation_mode
        if w_sep_barrier and not sep_barrier_nm > 5.0:
            raise ValueError("sep_barrier_nm must exceed the 5 NM intrusion distance")
        self.w_progress = w_progress
        self.w_separation = w_separation
        self.w_obstacle = w_obstacle
        self.w_boundary = w_boundary
        self.w_step = w_step
        self.w_action = w_action
        self.w_smooth = w_smooth
        self.w_miss = w_miss
        self.w_intrusion_event = w_intrusion_event
        self.w_restricted_event = w_restricted_event
        self.w_exit_event = w_exit_event
        self.w_sep_barrier = w_sep_barrier
        self.sep_barrier_nm = sep_barrier_nm
        self.warn_nm = warn_nm
        self.tcpa_horizon = tcpa_horizon
        self.obstacle_clearance_nm = obstacle_clearance_nm
        self.boundary_margin_nm = boundary_margin_nm
        self._prev_goal_dist = {}
        self._sector_poly_nm = None

    def _reset_shaping(self):
        clear_cpa_cache()
        self._prev_goal_dist = {}
        centre = np.array(self.scenario.center)
        verts = [fn.latlong_to_nm(centre, np.array(v)) for v in self.scenario.sector]
        self._sector_poly_nm = Polygon(verts)
        self._sector_verts = list(self.scenario.sector)

    def _own_nm(self, idx):
        centre = np.array(self.scenario.center)
        return fn.latlong_to_nm(centre, np.array([bs.traf.lat[idx], bs.traf.lon[idx]]))

    def _plan_cpa(self, idx, j):
        """(tcpa, dcpa_m) if both aircraft fly their plans; inf if diverging."""
        dmin, tmin = self._plan_closest(bs.traf.id[idx], bs.traf.id[j])
        if not np.isfinite(dmin) or tmin <= 1e-6:
            return np.inf, np.inf
        return tmin, dmin * 1852.0

    def _separation_cost(self, idx):
        cost = 0.0
        span = max(self.warn_nm - self.intrusion_distance, 1e-6)
        for j, tcpa, dcpa_m, d_nm in cpa_table(idx):
            if d_nm < self.warn_nm:
                cost += (self.warn_nm - d_nm) / span
            if self.separation_mode == "plan":
                tcpa, dcpa_m = self._plan_cpa(idx, j)
            if np.isfinite(tcpa) and 0.0 < tcpa < self.tcpa_horizon:
                dcpa_nm = dcpa_m / 1000.0 / NM2KM
                if dcpa_nm < self.warn_nm:
                    urgency = 1.0 - tcpa / self.tcpa_horizon
                    cost += urgency * (self.warn_nm - dcpa_nm) / self.warn_nm
        return cost

    def _barrier_cost(self, idx):
        """Quadratic in how far the current and the predicted distance reach
        into [5 NM, sep_barrier_nm]."""
        lo, hi = self.intrusion_distance, self.sep_barrier_nm
        band = lambda d: min(1.0, max(0.0, (hi - d) / (hi - lo))) ** 2
        cost = 0.0
        for j, tcpa, dcpa_m, d_nm in cpa_table(idx):
            cost += band(d_nm)
            if self.separation_mode == "plan":
                tcpa, dcpa_m = self._plan_cpa(idx, j)
            if np.isfinite(tcpa) and 0.0 < tcpa < self.tcpa_horizon:
                urgency = 1.0 - tcpa / self.tcpa_horizon
                cost += urgency * band(dcpa_m / 1000.0 / NM2KM)
        return cost

    def _obstacle_cost(self, idx):
        """Proximity ramp inside obstacle_clearance_nm, plus a term for the
        current track entering (or passing close to) an obstacle soon."""
        cost = 0.0
        own_lat, own_lon = bs.traf.lat[idx], bs.traf.lon[idx]
        psi = np.deg2rad(bs.traf.hdg[idx])
        u = np.array([np.cos(psi), np.sin(psi)])
        spd_nm_s = max(bs.traf.tas[idx] / (NM2KM * 1000.0), 1e-6)

        for verts in self._obstacle_polygons():
            A, B = sector_segments(own_lat, own_lon, verts)
            inside = point_in_polygon(A, B)
            edge_dist, _ = nearest_edge(A, B)
            margin = -edge_dist if inside else edge_dist
            if margin < self.obstacle_clearance_nm:
                cost += (self.obstacle_clearance_nm - max(margin, 0.0)) / self.obstacle_clearance_nm

            if inside:
                t_enter, miss = 0.0, 0.0
            else:
                hit = ray_exit_distance(A, B, u)
                if np.isfinite(hit):
                    t_enter, miss = hit / spd_nm_s, 0.0
                else:
                    miss, along = ray_miss_distance(A, B, u)
                    if miss >= self.obstacle_clearance_nm:
                        continue
                    t_enter = along / spd_nm_s

            if t_enter < self.tcpa_horizon:
                urgency = 1.0 - t_enter / self.tcpa_horizon
                depth = self.obstacle_clearance_nm - miss
                cost += urgency * depth / self.obstacle_clearance_nm
        return cost

    def _boundary_cost(self, idx):
        """Proximity to the sector edge plus a term for the current track
        leaving the sector soon; grows with distance once outside."""
        if self._sector_poly_nm is None:
            return 0.0
        p = Point(self._own_nm(idx))
        d_edge = p.distance(self._sector_poly_nm.exterior)
        if not self._sector_poly_nm.contains(p):
            return 1.0 + d_edge / self.boundary_margin_nm

        cost = 0.0
        if d_edge < self.boundary_margin_nm:
            cost += (self.boundary_margin_nm - d_edge) / self.boundary_margin_nm

        psi = np.deg2rad(bs.traf.hdg[idx])
        u = np.array([np.cos(psi), np.sin(psi)])
        A, B = sector_segments(bs.traf.lat[idx], bs.traf.lon[idx], self._sector_verts)
        exit_nm = ray_exit_distance(A, B, u)
        if np.isfinite(exit_nm):
            spd_nm_s = max(bs.traf.tas[idx] / (NM2KM * 1000.0), 1e-6)
            t_exit = exit_nm / spd_nm_s
            if t_exit < self.tcpa_horizon:
                cost += 1.0 - t_exit / self.tcpa_horizon
        return cost

    def _get_reward(self, ac_id):
        m = self.metrics[ac_id]
        base = self._baseline_of(ac_id)
        idx = bs.traf.id2idx(ac_id)
        glat, glon = self._goal_latlon(ac_id)

        qdr, dist_nm = kwikqdrdist(bs.traf.lat[idx], bs.traf.lon[idx], glat, glon)
        drift = abs(np.deg2rad(fn.bound_angle_positive_negative_180(bs.traf.hdg[idx] - qdr)))

        reached = m["waypoint_reached"] - base["waypoint_reached"]
        d_intrusion = m["intrusion_time"] - base["intrusion_time"]
        d_restricted = m["time_in_restricted_area"] - base["time_in_restricted_area"]
        d_outside = m["time_outside_sector"] - base["time_outside_sector"]

        prev = self._prev_goal_dist.get(ac_id, dist_nm)
        progress = prev - dist_nm
        self._prev_goal_dist[ac_id] = dist_nm

        # state costs are charged per decision, so scale them with the decision
        # interval to keep the trade-off per second fixed
        k = self.action_frequency / REFERENCE_ACTION_FREQUENCY

        reward = (self.reach_reward * reached
                  + self.w_progress * progress
                  + k * self.drift_penalty * drift
                  + self.intrusion_penalty * d_intrusion
                  + self.restricted_area_penalty * d_restricted
                  + self.sector_exit_penalty * d_outside
                  + k * self.w_step)

        reward += k * self.w_separation * self._separation_cost(idx)
        if self.w_sep_barrier:
            reward += k * self.w_sep_barrier * self._barrier_cost(idx)
        if self.w_intrusion_event:
            reward += self.w_intrusion_event * (m["intrusion_events"] - base["intrusion_events"])
        if self.w_restricted_event:
            reward += self.w_restricted_event * (m["restricted_area_events"]
                                                 - base["restricted_area_events"])
        if self.w_exit_event:
            reward += self.w_exit_event * (m["sector_exit_events"] - base["sector_exit_events"])
        reward += k * self.w_obstacle * self._obstacle_cost(idx)
        reward += k * self.w_boundary * self._boundary_cost(idx)

        a = getattr(self, "_action_now", {}).get(ac_id)
        if a is not None:
            reward += k * self.w_action * abs(float(a[0]))
            ap = getattr(self, "_action_prev", {}).get(ac_id)
            if ap is not None:
                reward += self.w_smooth * abs(float(a[0]) - float(ap[0]))
        if (self.w_miss and not m["waypoint_reached"]
                and m["flight_time"] >= self.episode_time_limit):
            reward += self.w_miss
        return reward
