"""Visibility-graph route planner around the obstacle polygons.

The action is an offset from the bearing to the next waypoint of this route,
so a0 = 0 means "follow a clear path" instead of "fly straight through
whatever is in the way". Obstacles and the sector are static and known at
reset, so planning once per episode is enough.

Graph nodes are the start, the goal and every obstacle corner pushed outward;
two nodes are connected when the straight leg between them keeps a clearance
from every obstacle and stays inside the sector. Dijkstra gives the shortest
route. Everything is numpy in a flat NM frame (x = north, y = east).
"""

import heapq

import numpy as np


def to_nm(latlon, ref_lat, ref_lon):
    a = np.atleast_2d(np.asarray(latlon, dtype=float))
    x = (a[:, 0] - ref_lat) * 60.0
    y = (a[:, 1] - ref_lon) * 60.0 * np.cos(np.radians(ref_lat))
    return np.column_stack((x, y))


def to_latlon(xy, ref_lat, ref_lon):
    a = np.atleast_2d(np.asarray(xy, dtype=float))
    lat = ref_lat + a[:, 0] / 60.0
    lon = ref_lon + a[:, 1] / (60.0 * np.cos(np.radians(ref_lat)))
    return np.column_stack((lat, lon))


def polygon_edges(poly):
    A = np.asarray(poly, dtype=float)
    return A, np.roll(A, -1, axis=0)


def point_inside(p, A, B):
    a, b = A - p, B - p
    straddle = (a[:, 1] > 0.0) != (b[:, 1] > 0.0)
    if not straddle.any():
        return False
    dy = np.where(straddle, b[:, 1] - a[:, 1], 1.0)
    xint = a[:, 0] + (-a[:, 1]) * (b[:, 0] - a[:, 0]) / dy
    return bool(np.count_nonzero(straddle & (xint > 0.0)) % 2)


def point_to_edges_dist(A, B, p):
    d = B - A
    dd = (d * d).sum(axis=1)
    t = np.clip(((p - A) * d).sum(axis=1) / np.maximum(dd, 1e-12), 0.0, 1.0)
    return np.linalg.norm(A + t[:, None] * d - p, axis=1)


def corners_to_seg_dist(p, q, corners):
    d = q - p
    dd = float(d @ d)
    t = np.clip(((corners - p) @ d) / max(dd, 1e-12), 0.0, 1.0)
    return np.linalg.norm(p + t[:, None] * d - corners, axis=1)


def segments_cross(p, q, A, B):
    """Proper crossing only: touching at an endpoint does not count."""
    r = q - p
    s = B - A
    denom = r[0] * s[:, 1] - r[1] * s[:, 0]
    qp = A - p
    ok = np.abs(denom) > 1e-12
    if not ok.any():
        return False
    safe = np.where(ok, denom, 1.0)
    t = (qp[:, 0] * s[:, 1] - qp[:, 1] * s[:, 0]) / safe
    u = (qp[:, 0] * r[1] - qp[:, 1] * r[0]) / safe
    return bool(np.any(ok & (t > 1e-9) & (t < 1.0 - 1e-9) & (u > 1e-9) & (u < 1.0 - 1e-9)))


def seg_clear_of_polygon(p, q, A, B, clearance):
    """Segment does not cross the polygon, is not inside it, and keeps clearance."""
    if segments_cross(p, q, A, B):
        return False
    if point_inside(0.5 * (p + q), A, B):
        return False
    if clearance <= 0.0:
        return True
    if point_to_edges_dist(A, B, p).min() < clearance:
        return False
    if point_to_edges_dist(A, B, q).min() < clearance:
        return False
    return bool(corners_to_seg_dist(p, q, A).min() >= clearance)


# vectorised versions over M segments at once (same arithmetic as above)

def segments_cross_many(P, Q, A, B):
    r = Q - P
    s = B - A
    denom = r[:, None, 0] * s[None, :, 1] - r[:, None, 1] * s[None, :, 0]
    qp = A[None, :, :] - P[:, None, :]
    ok = np.abs(denom) > 1e-12
    safe = np.where(ok, denom, 1.0)
    t = (qp[..., 0] * s[None, :, 1] - qp[..., 1] * s[None, :, 0]) / safe
    u = (qp[..., 0] * r[:, None, 1] - qp[..., 1] * r[:, None, 0]) / safe
    return np.any(ok & (t > 1e-9) & (t < 1.0 - 1e-9) & (u > 1e-9) & (u < 1.0 - 1e-9), axis=1)


def points_inside_many(X, A, B):
    a = A[None, :, :] - X[:, None, :]
    b = B[None, :, :] - X[:, None, :]
    straddle = (a[..., 1] > 0.0) != (b[..., 1] > 0.0)
    dy = np.where(straddle, b[..., 1] - a[..., 1], 1.0)
    xint = a[..., 0] + (-a[..., 1]) * (b[..., 0] - a[..., 0]) / dy
    return np.count_nonzero(straddle & (xint > 0.0), axis=1) % 2 == 1


def points_to_edges_min_dist(X, A, B):
    d = B - A
    dd = (d * d).sum(axis=1)
    t = np.clip(((X[:, None, :] - A[None, :, :]) * d[None, :, :]).sum(axis=2)
                / np.maximum(dd, 1e-12)[None, :], 0.0, 1.0)
    foot = A[None, :, :] + t[..., None] * d[None, :, :]
    return np.linalg.norm(foot - X[:, None, :], axis=2).min(axis=1)


def corners_to_segs_min_dist(P, Q, corners):
    d = Q - P
    dd = (d * d).sum(axis=1)
    t = np.clip(((corners[None, :, :] - P[:, None, :]) * d[:, None, :]).sum(axis=2)
                / np.maximum(dd, 1e-12)[:, None], 0.0, 1.0)
    foot = P[:, None, :] + t[..., None] * d[:, None, :]
    return np.linalg.norm(foot - corners[None, :, :], axis=2).min(axis=1)


def segs_clear_of_polygon_many(P, Q, A, B, clearance):
    ok = ~segments_cross_many(P, Q, A, B)
    ok &= ~points_inside_many(0.5 * (P + Q), A, B)
    if clearance <= 0.0:
        return ok
    ok &= points_to_edges_min_dist(P, A, B) >= clearance
    ok &= points_to_edges_min_dist(Q, A, B) >= clearance
    ok &= corners_to_segs_min_dist(P, Q, A) >= clearance
    return ok


def offset_polygon(poly, margin):
    """Corners pushed `margin` outward (miter offset, capped at sharp spikes).

    Pushing away from the centroid does not work for the long, stretched
    obstacles: corners at the ends of the long axis barely move away from the
    long edges and the graph comes out disconnected.
    """
    P = np.asarray(poly, dtype=float)
    if len(P) < 3:
        return P.copy()
    area2 = float(np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1]))
    sign = 1.0 if area2 > 0.0 else -1.0

    edge = np.roll(P, -1, axis=0) - P
    nrm = np.column_stack((edge[:, 1], -edge[:, 0])) * sign
    nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    n2 = nrm
    n1 = np.roll(nrm, 1, axis=0)
    b = n1 + n2
    denom = 1.0 + (n1 * n2).sum(axis=1)
    step = np.where(denom[:, None] > 1e-6, b / np.maximum(denom, 1e-6)[:, None], b)
    length = np.linalg.norm(step, axis=1, keepdims=True)
    step = step * np.minimum(1.0, 4.0 / np.maximum(length, 1e-12))
    return P + margin * step


class RoutePlanner:
    """Shortest obstacle-free route from a position to the goal.

    clearance_nm      distance kept from the obstacles (relaxed step by step
                      if nothing fits)
    sector_margin_nm  distance kept inside the sector boundary
    node_margin_nm    extra outward push for the corner nodes
    budget_nm         longest acceptable route
    budget_hard_nm    longer route allowed only if nothing fits budget_nm
    least_penetration if no clear route fits, take the one that spends the
                      least distance inside obstacles instead of the straight line
    lateral_offset_nm fly every route this far to the right of the shortest
                      path, so opposite directions round an obstacle in
                      separate lanes instead of head-on at its corners
    """

    # corner shifts tried, as multiples of lateral_offset_nm. Only a corner
    # bending away from its obstacle (the outer lane) has room above 1.
    OFFSET_STEPS = (5.0 / 3.0, 4.0 / 3.0, 1.0, 2.0 / 3.0, 1.0 / 3.0, 0.0)

    def __init__(self, clearance_nm=2.5, sector_margin_nm=2.0, node_margin_nm=1.5,
                 budget_nm=None, budget_hard_nm=None, least_penetration=False,
                 penetration_weights=(30.0, 10.0, 3.0), lateral_offset_nm=0.0):
        self.clearance_nm = float(clearance_nm)
        self.sector_margin_nm = float(sector_margin_nm)
        self.node_margin_nm = float(node_margin_nm)
        self.budget_nm = None if budget_nm is None else float(budget_nm)
        self.budget_hard_nm = None if budget_hard_nm is None else float(budget_hard_nm)
        self.least_penetration = bool(least_penetration)
        self.penetration_weights = tuple(float(w) for w in penetration_weights)
        self.lateral_offset_nm = float(lateral_offset_nm)
        self.ref = None
        self.obstacles = []
        self.sector = None
        self.n_plans = self.n_direct = self.n_hard_budget = 0
        self.n_least_pen = self.n_offset = 0
        self.last_clearance = None
        self.last_length_nm = None
        self.last_penetration_nm = None
        self.last_offset_nm = 0.0
        self.last_centreline_nm = None

    def reset(self, obstacles, sector, ref):
        """Cache the episode's obstacles and sector in a fixed NM frame about ref."""
        self.ref = ref
        self.obstacles = []
        self.sector = None
        if obstacles is None or ref is None:
            return
        lat, lon = ref
        for poly in obstacles:
            if poly is not None and len(poly) >= 3:
                self.obstacles.append(polygon_edges(to_nm(poly, lat, lon)))
        if sector is not None and len(sector) >= 3:
            self.sector = polygon_edges(to_nm(sector, lat, lon))

    def _legal(self, p, q, clearance=None):
        c = self.clearance_nm if clearance is None else clearance
        for A, B in self.obstacles:
            if not seg_clear_of_polygon(p, q, A, B, c):
                return False
        if self.sector is not None:
            A, B = self.sector
            if segments_cross(p, q, A, B):
                return False
            if not point_inside(0.5 * (p + q), A, B):
                return False
            m = self.sector_margin_nm
            if m > 0.0 and min(point_to_edges_dist(A, B, p).min(),
                               point_to_edges_dist(A, B, q).min(),
                               corners_to_seg_dist(p, q, A).min()) < m:
                return False
        return True

    def _legal_matrix(self, nodes, clearance):
        """L[i, j] for i < j: is the leg nodes[i] -> nodes[j] legal?"""
        n = len(nodes)
        I, J = np.triu_indices(n, k=1)
        P, Q = nodes[I], nodes[J]
        ok = np.ones(len(I), dtype=bool)
        for A, B in self.obstacles:
            ok &= segs_clear_of_polygon_many(P, Q, A, B, clearance)
        if self.sector is not None:
            A, B = self.sector
            ok &= ~segments_cross_many(P, Q, A, B)
            ok &= points_inside_many(0.5 * (P + Q), A, B)
            m = self.sector_margin_nm
            if m > 0.0:
                ok &= np.minimum(np.minimum(points_to_edges_min_dist(P, A, B),
                                            points_to_edges_min_dist(Q, A, B)),
                                 corners_to_segs_min_dist(P, Q, A)) >= m
        L = np.zeros((n, n), dtype=bool)
        L[I, J] = ok
        return L

    def _nodes(self, start, goal, clearance):
        margin = clearance + self.node_margin_nm
        cand = [offset_polygon(A, margin) for A, _ in self.obstacles]
        head = np.asarray([start, goal], dtype=float)
        if not cand:
            return head
        X = np.vstack(cand)
        ok = np.ones(len(X), dtype=bool)
        for A, B in self.obstacles:
            ok &= ~points_inside_many(X, A, B)
            ok &= points_to_edges_min_dist(X, A, B) >= clearance
        if self.sector is not None:
            A, B = self.sector
            ok &= points_inside_many(X, A, B)
            ok &= points_to_edges_min_dist(X, A, B) >= self.sector_margin_nm
        return np.vstack((head, X[ok]))

    def _shortest(self, nodes, clearance):
        """Dijkstra from node 0 to node 1; None if the goal is cut off."""
        n = len(nodes)
        legal = self._legal_matrix(nodes, clearance)
        adj = [[] for _ in range(n)]
        for i, j in zip(*np.nonzero(legal)):
            w = float(np.linalg.norm(nodes[i] - nodes[j]))
            adj[i].append((j, w))
            adj[j].append((i, w))

        dist = [np.inf] * n
        prev = [-1] * n
        dist[0] = 0.0
        pq = [(0.0, 0)]
        seen = set()
        while pq:
            d, u = heapq.heappop(pq)
            if u in seen:
                continue
            seen.add(u)
            if u == 1:
                break
            for v, w in adj[u]:
                if d + w < dist[v]:
                    dist[v] = d + w
                    prev[v] = u
                    heapq.heappush(pq, (dist[v], v))

        if not np.isfinite(dist[1]):
            return None
        path, u = [], 1
        while u != 0:
            path.append(nodes[u])
            u = prev[u]
        path.reverse()
        return np.asarray(path)

    def _offset_path(self, start, path, d):
        """Shift the corners of `path` to the right of travel.

        Each corner takes the largest of OFFSET_STEPS * d that keeps the legs
        legal; start and goal stay put. Returns (path, mean shift) or None.
        """
        P = np.vstack((start[None, :], path))
        if len(P) < 3:
            return path, 0.0
        u = np.diff(P, axis=0)
        L = np.linalg.norm(u, axis=1)
        if np.any(L < 1e-6):
            return None
        u = u / L[:, None]
        right = np.stack((-u[:, 1], u[:, 0]), axis=1)
        m = right[:-1] + right[1:]
        norm = np.linalg.norm(m, axis=1)
        cos_half = np.maximum(0.5 * norm, 0.5)
        bisector = np.where(norm[:, None] > 1e-9, m / np.maximum(norm, 1e-9)[:, None], 0.0)
        shift = d * bisector / cos_half[:, None]
        Q = P.copy()
        moved = 0.0
        for k in range(1, len(P) - 1):
            for f in self.OFFSET_STEPS:
                q = P[k] + f * shift[k - 1]
                if (np.dot(q - Q[k - 1], P[k] - P[k - 1]) > 0.0
                        and np.dot(P[k + 1] - q, P[k + 1] - P[k]) > 0.0
                        and self._legal(Q[k - 1], q) and self._legal(q, P[k + 1])):
                    Q[k] = q
                    moved += f * d
                    break
            else:
                return None
        return Q[1:], moved / (len(P) - 2)

    def plan(self, start_latlon, goal_latlon):
        """Waypoints (lat, lon) ending at the goal; empty if no route was found."""
        self.n_plans += 1
        self.last_clearance = None
        self.last_offset_nm = 0.0
        self.last_centreline_nm = None
        if self.ref is None:
            return np.zeros((0, 2))
        lat, lon = self.ref
        start = to_nm(start_latlon, lat, lon)[0]
        goal = to_nm(goal_latlon, lat, lon)[0]

        if self._legal(start, goal):
            self.n_direct += 1
            self.last_clearance = self.clearance_nm
            return to_latlon(goal[None, :], lat, lon)

        limit = np.inf if self.budget_nm is None else self.budget_nm

        # Clearance is a preference and length a constraint: take the widest
        # rung whose route fits the budget, else remember the shortest one.
        # With an offset, the shared centreline is first planned wider so the
        # shifted corners still keep the clearance.
        c, d = self.clearance_nm, self.lateral_offset_nm
        rungs = [(c, 0.0), (0.5 * c, 0.0), (0.25 * c, 0.0), (0.0, 0.0)]
        if d > 0.0:
            rungs[:0] = [(c + d, d), (c + 0.5 * d, d), (c, d)]
        best = None
        for clearance, offset in rungs:
            nodes = self._nodes(start, goal, clearance)
            if len(nodes) <= 2:
                continue
            path = self._shortest(nodes, clearance)
            if path is None:
                continue
            centreline = clearance
            if offset:
                shifted = self._offset_path(start, path, offset)
                if shifted is None or shifted[1] <= 0.0:
                    continue
                path, offset = shifted
                clearance = self.clearance_nm
            legs = np.vstack((start[None, :], path))
            length = float(np.linalg.norm(np.diff(legs, axis=0), axis=1).sum())
            if best is None or length < best[0]:
                best = (length, clearance, path, offset, centreline)
            if length <= limit:
                break

        if (best is not None and best[0] > limit and self.budget_hard_nm is not None
                and best[0] <= self.budget_hard_nm):
            limit = self.budget_hard_nm
            self.n_hard_budget += 1
        if best is not None and best[0] <= limit:
            length, clearance, path, offset, centreline = best
            self.last_clearance = clearance
            self.last_length_nm = length
            self.last_offset_nm = offset
            self.last_centreline_nm = centreline
            if offset:
                self.n_offset += 1
            return to_latlon(path, lat, lon)

        if self.least_penetration:
            cap = limit if self.budget_hard_nm is None else max(limit, self.budget_hard_nm)
            found = self._least_penetration(start, goal, cap)
            if found is not None:
                path, length, pen = found
                self.n_least_pen += 1
                self.last_length_nm = length
                self.last_penetration_nm = pen
                self.last_clearance = 0.0
                return to_latlon(path, lat, lon)

        # no route: the caller flies straight at the goal
        return np.zeros((0, 2))

    def _penetration_many(self, P, Q, samples=64):
        """Length of each leg P[k] -> Q[k] inside the obstacles (midpoint rule)."""
        L = np.linalg.norm(Q - P, axis=1)
        if not self.obstacles or not len(P):
            return np.zeros(len(P))
        t = (np.arange(samples) + 0.5) / samples
        X = (P[:, None, :] + t[None, :, None] * (Q - P)[:, None, :]).reshape(-1, 2)
        inside = np.zeros(len(X), dtype=bool)
        for A, B in self.obstacles:
            inside |= points_inside_many(X, A, B)
        return L * inside.reshape(len(P), samples).mean(axis=1)

    def _least_penetration(self, start, goal, limit):
        """Shortest path where every NM inside an obstacle costs extra
        (weights from strict to lenient); first one within `limit` wins."""
        for margin in (1.0, 0.0):
            found = self._least_penetration_at(start, goal, limit, margin)
            if found is not None:
                return found
        return None

    def _least_penetration_at(self, start, goal, limit, margin):
        cand = [offset_polygon(A, self.node_margin_nm) for A, _ in self.obstacles]
        X = np.vstack(cand) if cand else np.zeros((0, 2))
        if self.sector is not None and len(X):
            X = X[points_inside_many(X, *self.sector)]
            if margin > 0.0 and len(X):
                X = X[points_to_edges_min_dist(X, *self.sector) >= margin]
        nodes = np.vstack((start[None, :], goal[None, :], X))
        n = len(nodes)
        I, J = np.triu_indices(n, k=1)
        P, Q = nodes[I], nodes[J]
        ok = np.ones(len(I), dtype=bool)
        if self.sector is not None:
            A, B = self.sector
            ok &= ~segments_cross_many(P, Q, A, B)
            ok &= points_inside_many(0.5 * (P + Q), A, B)
            if margin > 0.0:
                ok &= np.minimum(np.minimum(points_to_edges_min_dist(P, A, B),
                                            points_to_edges_min_dist(Q, A, B)),
                                 corners_to_segs_min_dist(P, Q, A)) >= margin
        length = np.linalg.norm(Q - P, axis=1)
        pen = self._penetration_many(P, Q)
        for w in self.penetration_weights:
            C = np.full((n, n), np.inf)
            cost = np.where(ok, length + w * pen, np.inf)
            C[I, J] = cost
            C[J, I] = cost
            dist = np.full(n, np.inf)
            dist[0] = 0.0
            prev = np.full(n, -1)
            done = np.zeros(n, dtype=bool)
            for _ in range(n):
                u = int(np.argmin(np.where(done, np.inf, dist)))
                if not np.isfinite(dist[u]) or u == 1:
                    break
                done[u] = True
                nd = dist[u] + C[u]
                better = (nd < dist) & ~done
                dist[better] = nd[better]
                prev[better] = u
            if not np.isfinite(dist[1]):
                continue
            idx = [1]
            while idx[-1] != 0:
                idx.append(int(prev[idx[-1]]))
            pts = nodes[idx[::-1]]
            total = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
            if total <= limit:
                p_tot = float(self._penetration_many(pts[:-1], pts[1:]).sum())
                return pts[1:], total, p_tot
        return None

    def visible(self, a_latlon, b_latlon, clearance=None):
        if self.ref is None:
            return True
        lat, lon = self.ref
        return self._legal(to_nm(a_latlon, lat, lon)[0], to_nm(b_latlon, lat, lon)[0], clearance)

    def min_clearance(self, path_latlon):
        """Smallest distance from a polyline to any obstacle (0 if it enters one)."""
        if self.ref is None or not self.obstacles or len(path_latlon) < 2:
            return np.inf
        lat, lon = self.ref
        pts = to_nm(path_latlon, lat, lon)
        worst = np.inf
        for p, q in zip(pts[:-1], pts[1:]):
            for A, B in self.obstacles:
                if segments_cross(p, q, A, B) or point_inside(0.5 * (p + q), A, B):
                    return 0.0
                worst = min(worst,
                            point_to_edges_dist(A, B, p).min(),
                            point_to_edges_dist(A, B, q).min(),
                            corners_to_seg_dist(p, q, A).min())
        return float(worst)
