"""Where do the multi-agent intrusions happen? And what does the lateral offset change?

Run from bluesky-gym/:

    python ../results/analysis/route_conflicts.py diagnose   # the corner table
    python ../results/analysis/route_conflicts.py offset     # planned conflicts, offset 0 vs 3

Both regenerate the validation scenarios (eval seed 7777) without the simulator
and replay every aircraft's PLANNED route at constant speed (150 m/s, removed at
the 5 km capture radius). It's only a proxy, since the policy deviates from its route
and changes speed, but it shows where the planner puts aircraft together.

``diagnose`` matches each pair of MA aircraft that lost separation in a
validation run (two aircraft of one episode with the same intrusion seconds) to
its planned-route geometry. Result for ma_2531101 (validation/val_ma_2534660, 300
episodes): 2419 planned-route conflicts against 1276 for straight lines, ~97%
resolved by the policy; head-on at an obstacle corner (< 8 NM) 760 conflicts,
5.4% unresolved, 48% of the paired intrusion seconds.

``offset`` plans every route with and without the lateral offset: 3000/3000
routes identical at offset 0 to the planner without the option; at 3 NM
pairwise conflicts 2419 -> 1874, head-on 1865 -> 1311, route +6.2%, least
penetration routes unchanged, minimum clearance 2.50 NM.
"""
import collections
import json
import sys

import numpy as np

sys.path.insert(0, ".")
from gymnasium.utils import seeding                       # noqa: E402
from core.scenario import ScenarioGenerator               # noqa: E402
from solution.route import RoutePlanner, to_nm            # noqa: E402

VAL = "../results/validation/val_ma_2534660/val.json"
MODEL = "ma_2531101"
FLY = 150.0 * 3000 / 1852.0        # NM flyable in the time limit
V_NM = 150.0 / 1852.0              # NM per second
CAP = 5.0 / 1.852                  # goal capture radius, NM


def planner(offset=0.0):
    # the submitted MA planner: budget 0.8, longer routes up to 0.85, least penetration
    return RoutePlanner(clearance_nm=2.5, budget_nm=0.8 * FLY, budget_hard_nm=0.85 * FLY,
                        least_penetration=True, lateral_offset_nm=offset)


def track(poly):
    """Positions at 1 s steps along a polyline until within CAP of its end."""
    seg = np.diff(poly, axis=0)
    L = np.linalg.norm(seg, axis=1)
    cum = np.r_[0, np.cumsum(L)]
    s = np.arange(0, max(0.0, cum[-1] - CAP) / V_NM + 1) * V_NM
    i = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(L) - 1)
    f = (s - cum[i]) / np.where(L[i] > 0, L[i], 1)
    return poly[i] + seg[i] * f[:, None], cum[-1]


def routes(pl, sc, ref):
    out = {}
    for a in sc.agents:
        wps = pl.plan(a.start, a.goal)
        st = to_nm([a.start], *ref)
        poly = np.vstack([st, to_nm(wps if len(wps) else [a.goal], *ref)])
        out[a.ac_id] = (track(poly), poly)
    return out


def closest(a, b):
    n = min(len(a), len(b))
    d = np.linalg.norm(a[:n] - b[:n], axis=1)
    return d, int(d.argmin())


def geometry(a, b, k, corners):
    va = a[min(k + 1, len(a) - 1)] - a[max(k - 1, 0)]
    vb = b[min(k + 1, len(b) - 1)] - b[max(k - 1, 0)]
    ang = np.degrees(np.arccos(np.clip(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9), -1, 1)))
    kind = "same dir (<45)" if ang < 45 else "crossing (45-135)" if ang < 135 else "head-on (>135)"
    p = 0.5 * (a[k] + b[k])
    near = len(corners) and np.min(np.linalg.norm(corners - p, axis=1)) < 8.0
    return kind + (", at a corner (<8 NM)" if near else ", away from corners")


def scenarios(n_eps, seed):
    gen = ScenarioGenerator(n_obstacles=5, n_intruders=0, n_agents=10, agent_speed=150.0,
                            center=(52.0, 4.0))
    rng, _ = seeding.np_random(seed)
    for ep in range(n_eps):
        sc = gen.generate(rng)
        yield ep, sc, tuple(np.mean(np.asarray(sc.sector, float), axis=0))


def diagnose():
    V = json.load(open(VAL))
    meta = V.pop("_meta")
    by_ep = collections.defaultdict(dict)
    for r in V[MODEL]:
        by_ep[r["ep"]][r["agent"]] = r
    allc, hit, secs = collections.Counter(), collections.Counter(), collections.Counter()
    n_route = n_straight = 0
    for ep, sc, ref in scenarios(meta["episodes"], meta["eval_seed"]):
        pl = planner()
        pl.reset([o.vertices for o in sc.obstacles], sc.sector, ref)
        R = routes(pl, sc, ref)
        S = {a.ac_id: track(np.vstack([to_nm([a.start], *ref), to_nm([a.goal], *ref)]))[0]
             for a in sc.agents}
        same_t = collections.defaultdict(list)          # pairs: equal intrusion seconds
        for i, r in by_ep[ep].items():
            if r["intrusion_time"] > 0:
                same_t[r["intrusion_time"]].append(i)
        actual = {tuple(sorted(v)): t for t, v in same_t.items() if len(v) == 2}
        ids = sorted(R)
        for x in range(len(ids)):
            for y in range(x + 1, len(ids)):
                i, j = ids[x], ids[y]
                d, k = closest(R[i][0][0], R[j][0][0])
                n_straight += closest(S[i], S[j])[0].min() < 5
                if d.min() >= 5:
                    continue
                n_route += 1
                corners = np.vstack([R[i][1][1:-1], R[j][1][1:-1]])
                key = geometry(R[i][0][0], R[j][0][0], k, corners)
                allc[key] += 1
                if (i, j) in actual:
                    hit[key] += 1
                    secs[key] += 2 * actual[(i, j)]
    tot = sum(secs.values())
    print(f"{MODEL}, {meta['episodes']} episodes: planned-route conflicts {n_route} "
          f"(straight lines {n_straight}); unresolved {sum(hit.values())}")
    print(f"{'geometry':44s} {'conflicts':>9s} {'unresolved':>10s} {'fail%':>6s} {'intr s share':>12s}")
    for key in sorted(allc, key=lambda k: -secs[k]):
        print(f"{key:44s} {allc[key]:9d} {hit[key]:10d} {100 * hit[key] / allc[key]:5.1f}% "
              f"{100 * secs[key] / max(tot, 1):11.1f}%")


def offset(n_eps=300, d=3.0):
    stats = {0.0: collections.Counter(), d: collections.Counter()}
    worst = np.inf
    for ep, sc, ref in scenarios(n_eps, 7777):
        for off, c in stats.items():
            pl = planner(off)
            pl.reset([o.vertices for o in sc.obstacles], sc.sector, ref)
            R = routes(pl, sc, ref)
            c["routes offset"] += pl.n_offset
            c["least penetration"] += pl.n_least_pen
            c["route NM"] += sum(L for (_, L), _ in R.values())
            if off:
                for a in sc.agents:
                    wps = pl.plan(a.start, a.goal)
                    if pl.last_offset_nm:
                        worst = min(worst, pl.min_clearance(np.vstack(([a.start], wps))))
            T = [R[k][0][0] for k in sorted(R)]
            for x in range(len(T)):
                for y in range(x + 1, len(T)):
                    dd, k = closest(T[x], T[y])
                    if dd.min() < 5:
                        c["conflicts"] += 1
                        va = T[x][min(k + 1, len(T[x]) - 1)] - T[x][max(k - 1, 0)]
                        vb = T[y][min(k + 1, len(T[y]) - 1)] - T[y][max(k - 1, 0)]
                        c["head-on"] += va @ vb < -0.7 * np.linalg.norm(va) * np.linalg.norm(vb)
    for off, c in stats.items():
        print(f"offset {off:g} NM: conflicts {c['conflicts']}, head-on {c['head-on']}, "
              f"route {c['route NM'] / (10 * n_eps):.1f} NM/aircraft, routes offset "
              f"{c['routes offset']}, least penetration {c['least penetration']}")
    print(f"minimum obstacle clearance of offset routes: {worst:.2f} NM")


if __name__ == "__main__":
    {"diagnose": diagnose, "offset": offset}[sys.argv[1] if len(sys.argv) > 1 else "diagnose"]()
