"""How much harder is conflict-timed training traffic? (solution/traffic.py)

Run from bluesky-gym/:   python ../results/analysis/traffic_counts.py

For 300 scenarios from each generator, the agent flies its straight start-goal
line at 150 m/s and every intruder its route at its own speed; counts the
intruders that come within 5 NM (and 10 NM). Result (seed 12345):

    original          1.25 intruders < 5 NM per scenario, 71% of scenarios with one
    conflict-timed    2.55 intruders < 5 NM per scenario, 87% of scenarios with one
                      (27% of intruders re-timed; the rest cannot be without
                       leaving the 120-180 m/s speed range)
"""
import math
import sys

import numpy as np

sys.path.insert(0, ".")
from core.scenario import ScenarioGenerator                    # noqa: E402
from solution.traffic import ConflictTimedScenarioGenerator    # noqa: E402

C = (52.0, 4.0)


def nm(p):
    return np.array([(p[0] - C[0]) * 60.0, (p[1] - C[1]) * 60.0 * math.cos(math.radians(C[0]))])


def flown(pts, v_nm_s, T):
    pts = np.array(pts)
    seg = np.diff(pts, axis=0)
    L = np.linalg.norm(seg, axis=1)
    cum = np.r_[0, np.cumsum(L)]
    s = np.minimum(np.arange(T) * v_nm_s, cum[-1])
    i = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(L) - 1)
    out = pts[i] + seg[i] * ((s - cum[i]) / np.where(L[i] > 0, L[i], 1))[:, None]
    out[np.arange(T) * v_nm_s > cum[-1]] = np.nan              # gone after its exit
    return out


def counts(gen, n=300, seed=12345):
    rng = np.random.default_rng(seed)
    los = near = eps = 0
    for _ in range(n):
        sc = gen.generate(rng)
        a = sc.agents[0]
        s, g = nm(a.start), nm(a.goal)
        T = int(np.linalg.norm(g - s) / (150 / 1852)) + 1
        agent = s + (g - s) * (np.arange(T) / max(T - 1, 1))[:, None]
        hit = False
        for r in sc.intruder_routes:
            d = np.linalg.norm(flown([nm(p) for p in r.waypoints], r.speed / 1852, T) - agent, axis=1)
            d = d[~np.isnan(d)]
            if len(d) and d.min() < 5:
                los += 1
                hit = True
            if len(d) and d.min() < 10:
                near += 1
        eps += hit
    return los / n, near / n, eps / n


if __name__ == "__main__":
    kw = dict(n_obstacles=5, n_intruders=10, agent_speed=150.0, center=C)
    hard = ConflictTimedScenarioGenerator(**kw)
    for name, gen in (("original", ScenarioGenerator(**kw)), ("conflict-timed", hard)):
        l, n10, e = counts(gen)
        print(f"{name:15s} intruders < 5 NM {l:.2f}, < 10 NM {n10:.2f} per scenario; "
              f"scenarios with one {100 * e:.0f}%")
    print(f"re-timed {hard.n_retimed}/{hard.n_routes} intruders")
