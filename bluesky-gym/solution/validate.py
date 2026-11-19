"""Rank models on a validation scenario stream (default seed 7777), in parallel.

The official test is seed 42, so models are chosen on another stream and only
the chosen ones are run on seed 42. Each worker regenerates the same stream,
so every model flies the same episodes and differences are paired (with seed
42 this reproduces the official per-episode results).

A model is "run_id", a run directory or a .zip, optionally "@M" for the
checkpoint nearest M million steps, "+key=value" to override an MDP setting,
and "label=" in front to name it:

    python -m solution.validate --kind sa --episodes 1000 --out val_sa.json         ../submission/sa/ppo_sa.zip sa_2527464 sa_2527464@14
    python -m solution.validate --kind ma --episodes 300 --out val_ma.json         off3=ma_2531101+route_lateral_offset_nm=3
    python -m solution.validate --summarize val_sa.json [more.json ...]
"""
import argparse
import contextlib
import glob
import io
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from multiprocessing import get_context

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEYS = ["intrusion_time", "time_in_restricted_area", "time_outside_sector",
        "flight_time", "waypoint_reached", "intrusion_events",
        "restricted_area_events", "sector_exit_events"]


def _checkpoint(run_dir, millions):
    zips = glob.glob(os.path.join(run_dir, "ppo_*_ck", "ck_*_steps.zip"))
    steps = {int(re.search(r"ck_(\d+)_steps", z).group(1)): z for z in zips}
    if not steps:
        raise SystemExit(f"no checkpoints in {run_dir}")
    best = min(steps, key=lambda s: abs(s - millions * 1e6))
    if abs(best - millions * 1e6) > 0.5e6:
        raise SystemExit(f"no checkpoint near {millions}M in {run_dir}")
    return steps[best]


def _overrides(parts):
    """``["key=value", ...]`` -> dict, values parsed as JSON where they parse."""
    out = {}
    for part in parts:
        key, eq, raw = part.partition("=")
        if not (key and eq):
            raise SystemExit(f"override {part!r}: expected key=value")
        try:
            out[key] = json.loads(raw)
        except ValueError:
            out[key] = raw
    return out


def resolve(spec, results_root):
    """``label=spec[+key=value...]`` -> (label, absolute .zip path, overrides)."""
    spec, *extra = spec.split("+")
    overrides = _overrides(extra)
    label, _, spec = spec.rpartition("=") if "=" in spec else ("", "", spec)
    path, _, ck = spec.partition("@")
    if not os.path.exists(path):
        hits = sorted(h for h in glob.glob(os.path.join(results_root, "**", path), recursive=True)
                      if os.path.isdir(h))
        # the same run may have been copied into two folders
        sizes = {tuple(sorted(os.path.getsize(z) for z in glob.glob(os.path.join(h, "ppo_*.zip"))))
                 for h in hits}
        if len(hits) > 1 and len(sizes) == 1:
            hits = hits[:1]
        if len(hits) != 1:
            raise SystemExit(f"{spec}: expected one run directory named {path} under "
                             f"{results_root}, found {len(hits)}: {hits[:4]}")
        path = hits[0]
    path = os.path.abspath(path)
    if os.path.isdir(path):
        run = path
        if ck:
            path = _checkpoint(run, float(ck))
        else:
            finals = glob.glob(os.path.join(run, "ppo_*.zip"))
            if len(finals) != 1:
                raise SystemExit(f"{run}: expected one final ppo_*.zip, found {finals}")
            path = finals[0]
    elif ck:
        raise SystemExit(f"{spec}: '@' needs a run directory or run id, not a file")
    if not label:
        m = re.search(r"((?:sa|ma)_\d+)", path)
        label = m.group(1) if m else os.path.splitext(os.path.basename(path))[0]
        if "ck_" in os.path.basename(path):
            s = int(re.search(r"ck_(\d+)_steps", path).group(1))
            label += f"@{s / 1e6:.0f}M"
        label += "".join(f"+{k}={v}" for k, v in overrides.items())
    return label, path, overrides


def _work(task):
    kind, label, model, overrides, shard, nsh, n_eps, seed = task
    sys.path.insert(0, REPO)
    os.chdir(REPO)
    import gymnasium as gym
    from gymnasium.utils import seeding
    from stable_baselines3 import PPO
    from core.scenario import ScenarioGenerator
    from solution.evaluate import resolve_mdp_kw
    from solution.envs import ShapedCompetitionEnv, ShapedCompetitionZooEnv
    from solution.wrappers import FlattenObs
    with contextlib.redirect_stdout(io.StringIO()):
        kw = resolve_mdp_kw(model)
    kw.update(overrides)
    pol = PPO.load(model, device="cpu")
    act = lambda o: pol.predict(o, deterministic=True)[0]
    rng, _ = seeding.np_random(seed)
    if kind == "sa":
        env = gym.wrappers.FlattenObservation(ShapedCompetitionEnv(**kw))
        gen = ScenarioGenerator(n_obstacles=5, n_intruders=10, agent_speed=150.0,
                                center=(52.0, 4.0))
        raw = env.unwrapped
    else:
        env = FlattenObs(ShapedCompetitionZooEnv(n_agents=10, **kw))
        gen = ScenarioGenerator(n_obstacles=5, n_intruders=0, n_agents=10,
                                agent_speed=150.0, center=(52.0, 4.0))
        raw = env.env
    out = []
    for ep in range(n_eps):
        sc = gen.generate(rng)
        if ep % nsh != shard:
            continue
        raw._fixed_scenario = sc
        obs, _ = env.reset()
        if kind == "sa":
            done = False
            while not done:
                obs, _, term, trunc, info = env.step(act(obs))
                done = term or trunc
            out.append(dict(ep=ep, **{k: float(v) for k, v in info.items()}))
        else:
            last = {}
            while env.agents:
                obs, _, _, _, infos = env.step({a: act(obs[a]) for a in env.agents})
                last.update(infos)
            for a, info in last.items():
                out.append(dict(ep=ep, agent=a, **{k: float(v) for k, v in info.items()}))
    env.close()
    return label, out


def run(kind, n_eps, seed, models, workers):
    # ~3 tasks per worker keeps the pool busy until the end
    nsh = max(1, min(n_eps, math.ceil(3 * workers / len(models))))
    tasks = [(kind, lab, path, ov, k, nsh, n_eps, seed)
             for lab, path, ov in models for k in range(nsh)]
    res = {lab: [] for lab, _, _ in models}
    t0, done = time.time(), 0
    with get_context("spawn").Pool(min(workers, len(tasks))) as pool:
        for lab, rows in pool.imap_unordered(_work, tasks):
            res[lab] += rows
            done += 1
            if done % max(1, len(tasks) // 10) == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)} shards, {time.time() - t0:.0f} s", flush=True)
    for lab in res:
        res[lab].sort(key=lambda r: (r["ep"], r.get("agent", "")))
    return res


def per_episode(rows):
    """Per-episode arrays; MA rows are first averaged over the agents."""
    by = defaultdict(list)
    for r in rows:
        by[r["ep"]].append(r)
    eps = sorted(by)
    out = {k: np.array([np.mean([r[k] for r in by[e]]) for e in eps]) for k in KEYS}
    out["ALL3"] = (out["intrusion_time"] + out["time_in_restricted_area"]
                   + out["time_outside_sector"])
    out["misses"] = sum(r["waypoint_reached"] < 1 for r in rows)
    # clean flight: goal reached, no intrusion, no restricted entry, no exit
    clean = lambda r: (r["waypoint_reached"] >= 1 and r["intrusion_time"] == 0
                       and r["time_in_restricted_area"] == 0 and r["time_outside_sector"] == 0
                       and r["intrusion_events"] == 0 and r["restricted_area_events"] == 0
                       and r["sector_exit_events"] == 0)
    out["clean"] = np.array([np.mean([clean(r) for r in by[e]]) for e in eps])
    return eps, out


def summarize(res, meta=None, boot=4000):
    m = {lab: per_episode(rows) for lab, rows in res.items()}
    eps0 = next(iter(m.values()))[0]
    if any(v[0] != eps0 for v in m.values()):
        raise SystemExit("models were scored on different episodes; not comparable")
    m = {lab: v[1] for lab, v in m.items()}
    order = sorted(m, key=lambda k: m[k]["ALL3"].mean())
    best = order[0]
    A = np.stack([m[k]["ALL3"] for k in order])
    rng = np.random.default_rng(0)
    wins = np.zeros(len(order))
    for _ in range(boot):
        wins[np.argmin(A[:, rng.integers(0, A.shape[1], A.shape[1])].mean(1))] += 1
    head = f"{len(eps0)} episodes, identical across models"
    if meta:
        head += f" ({meta.get('kind', '?')}, eval seed {meta.get('eval_seed', '?')})"
    lines = [head, "",
             f"{'model':18s} {'ALL3':>7s} {'intr':>6s} {'restr':>6s} {'out':>5s} {'flight':>7s}"
             f" {'goal':>6s} {'clean%':>6s} {'miss':>4s} {'vs best':>14s} {'P(best)':>7s}"]
    for k, w in zip(order, wins):
        x = m[k]
        d = x["ALL3"] - m[best]["ALL3"]
        vs = "" if k == best else f"{d.mean():+6.2f} +- {d.std() / np.sqrt(len(d)):4.2f}"
        lines.append(f"{k:18s} {x['ALL3'].mean():7.2f} {x['intrusion_time'].mean():6.2f}"
                     f" {x['time_in_restricted_area'].mean():6.2f} {x['time_outside_sector'].mean():5.2f}"
                     f" {x['flight_time'].mean():7.1f} {x['waypoint_reached'].mean():6.3f}"
                     f" {100 * x['clean'].mean():6.1f} {x['misses']:4d} {vs:>14s} {w / boot:7.3f}")
    lines += ["", "Ranked by ALL3 (violation seconds). clean% = flights with goal reached and no",
              "intrusion, restricted entry or sector exit (MA: per aircraft).",
              "vs best = paired difference over the same episodes, +- its standard error.",
              "Pick the best; anything within ~2 SE of it is a statistical tie."]
    return "\n".join(lines)


def load(paths):
    res, metas = {}, []
    for p in paths:
        d = json.load(open(p))
        metas.append(d.pop("_meta", {}))
        for lab, rows in d.items():
            if lab in res:
                lab = f"{lab}[{os.path.basename(p)}]"
            res[lab] = rows
    seeds = {(mm.get("kind"), mm.get("eval_seed"), mm.get("episodes")) for mm in metas if mm}
    if len(seeds) > 1:
        raise SystemExit(f"files come from different streams: {seeds}")
    return res, (metas[0] if metas else {})


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("models", nargs="*")
    p.add_argument("--kind", choices=("sa", "ma"))
    p.add_argument("--episodes", type=int)
    p.add_argument("--eval-seed", type=int, default=7777)
    p.add_argument("--out")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--results", default=os.path.join(os.path.dirname(REPO), "results"),
                   help="where run ids are looked up")
    p.add_argument("--summarize", nargs="+", metavar="JSON",
                   help="print the ranking of saved results instead of evaluating")
    args = p.parse_args()

    if args.summarize:
        res, meta = load(args.summarize)
        print(summarize(res, meta))
        return
    if not (args.kind and args.episodes and args.out and args.models):
        p.error("--kind, --episodes, --out and at least one model are required")
    if args.eval_seed == 42:
        print("WARNING: seed 42 is the official test stream, don't pick models on it")
    models = [resolve(s, args.results) for s in args.models]
    labels = [lab for lab, _, _ in models]
    if len(set(labels)) != len(labels):
        p.error(f"duplicate labels: {labels}")
    print(f"validating {len(models)} {args.kind.upper()} models x {args.episodes} episodes, "
          f"eval seed {args.eval_seed}, {args.workers} workers")
    for lab, path, ov in models:
        print(f"  {lab:18s} {path}" + (f"  overrides {ov}" if ov else ""))
    t0 = time.time()
    res = run(args.kind, args.episodes, args.eval_seed, models, args.workers)
    meta = dict(kind=args.kind, eval_seed=args.eval_seed, episodes=args.episodes,
                models={lab: path for lab, path, _ in models},
                overrides={lab: ov for lab, _, ov in models if ov},
                seconds=round(time.time() - t0))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(dict(_meta=meta, **res), open(args.out, "w"))
    print(f"done in {time.time() - t0:.0f} s -> {args.out}\n")
    print(summarize(res, meta))


if __name__ == "__main__":
    main()
