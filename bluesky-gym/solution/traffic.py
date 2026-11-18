"""Conflict-timed intruder traffic, used for single-agent training only.

The scenario generator only asks an intruder's route to pass within 15 NM of
the agent's straight start-goal line; when the intruder gets there is random,
so most intruders cross far from the agent. Here intruders are re-timed so
they reach the crossing at about the moment the agent does (+- jitter_s),
which gives more of the close encounters the policy still gets wrong.

Only the spawn point along the intruder's own route changes (and, if that is
not enough, its speed within the original 120-180 m/s); routes, sector,
obstacles and the agent are untouched. Evaluation always uses the original
generator.
"""

import math

from shapely.geometry import LineString, Point
from shapely.ops import nearest_points

from core.scenario import ScenarioGenerator, _nm_to_latlon


class ConflictTimedScenarioGenerator(ScenarioGenerator):
    """retime_frac: chance each intruder is tried (only ~a quarter can be
    re-timed at all). min_lead_s / min_spawn_nm skip encounters that would
    start at or right next to the agent's spawn."""

    def __init__(self, retime_frac=1.0, jitter_s=60.0, min_lead_s=120.0,
                 min_spawn_nm=15.0, **kwargs):
        super().__init__(**kwargs)
        self.retime_frac = float(retime_frac)
        self.jitter_s = float(jitter_s)
        self.min_lead_s = float(min_lead_s)
        self.min_spawn_nm = float(min_spawn_nm)
        self.n_routes = 0
        self.n_retimed = 0

    def _to_nm(self, latlon):
        c = self.center
        return ((latlon[0] - c[0]) * 60.0,
                (latlon[1] - c[1]) * 60.0 * math.cos(math.radians(c[0])))

    def _generate_routes(self, rng, sector_poly, obstacle_polys, start, goal):
        routes = super()._generate_routes(rng, sector_poly, obstacle_polys, start, goal)
        agent_line = LineString([start, goal])
        for route in routes:
            self.n_routes += 1
            if rng.random() < self.retime_frac and self._retime(route, agent_line, start, rng):
                self.n_retimed += 1
        return routes

    def _retime(self, route, agent_line, start, rng):
        """Move the spawn (and if needed the speed) so the intruder meets the
        agent's line with the agent. Leaves the route unchanged if it can't."""
        pts = [self._to_nm(p) for p in route.waypoints]
        line = LineString(pts)
        hit = line.intersection(agent_line)
        if hit.is_empty:
            on_route, on_agent = nearest_points(line, agent_line)
        else:
            cands = list(getattr(hit, "geoms", [hit]))
            on_route = min((g for g in cands if g.geom_type == "Point"),
                           key=line.project, default=None)
            if on_route is None:
                return False
            on_agent = on_route
        t_agent = (agent_line.project(on_agent) / (self.agent_speed / 1852.0)
                   + rng.uniform(-self.jitter_s, self.jitter_s))
        if t_agent < self.min_lead_s:
            return False
        s_x = line.project(on_route)
        speed = route.speed
        s0 = s_x - (speed / 1852.0) * t_agent      # distance along the route at t = 0
        if s0 < 0.0:
            # would arrive too early even from its spawn: slow it down, if allowed
            speed = s_x / t_agent * 1852.0
            if not self.intruder_speed_range[0] <= speed <= self.intruder_speed_range[1]:
                return False
            s0 = 0.0
        if s0 >= line.length - 1.0:
            return False
        spawn = line.interpolate(s0)
        if spawn.distance(Point(start)) < self.min_spawn_nm:
            return False
        rest = [p for p in pts[1:] if line.project(Point(p)) > s0 + 1e-6]
        route.waypoints = ([_nm_to_latlon(self.center, (spawn.x, spawn.y))]
                           + [_nm_to_latlon(self.center, p) for p in rest])
        route.speed = float(speed)
        return True
