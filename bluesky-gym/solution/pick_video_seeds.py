"""Choose the five video scenarios by a fixed rule instead of by eye.

Of scenario seeds 1..N, keep those where the trained policy's flight is clean
(every aircraft at its goal, no intrusion, restricted-area entry or sector
exit) and take the K where the do-nothing policy has the most unsafe seconds.
Seeds are env.reset(seed=s) seeds, as used by solution.record.

    python -m solution.pick_video_seeds --env sa --model ../submission/sa/ppo_sa --out seeds_sa.json
"""
import argparse
import json
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np  # noqa: E402

KEYS = ("intrusion_time", "time_in_restricted_area", "time_outside_sector")


def _episode(task):
    """One scenario with one policy, no rendering: ``(seed, trained, unsafe_s, clean)``."""
    env_kind, seed, model_path, n_agents = task
    import contextlib
    import io
    import gymnasium as gym
    from solution.envs import ShapedCompetitionEnv, ShapedCompetitionZooEnv
    from solution.record import load_policy, mdp_kwargs
    from solution.wrappers import FlattenObs
    with contextlib.redirect_stdout(io.StringIO()):
        act = load_policy(model_path)
        mdp = mdp_kwargs(model_path) if model_path else {}
        if env_kind == "sa":
            env = gym.wrappers.FlattenObservation(ShapedCompetitionEnv(**mdp))
        else:
            env = FlattenObs(ShapedCompetitionZooEnv(n_agents=n_agents, **mdp))
    zero = np.zeros(2, dtype=np.float32)
    if env_kind == "sa":
        obs, _ = env.reset(seed=seed)
        done = False
        while not done:
            obs, _, term, trunc, info = env.step(zero if act is None else act(obs))
            done = term or trunc
        infos = [info]
    else:
        obs, _ = env.reset(seed=seed)
        last = {}
        while env.agents:
            acts = {a: (zero if act is None else act(obs[a])) for a in env.agents}
            obs, _, _, _, infos = env.step(acts)
            last.update(infos)
        infos = list(last.values())
    env.close()
    unsafe = sum(float(i[k]) for i in infos for k in KEYS)
    clean = all(float(i["waypoint_reached"]) >= 1 and all(float(i[k]) == 0 for k in KEYS)
                for i in infos)
    return seed, model_path is not None, unsafe, bool(clean)


def pick(rows, k):
    """Apply the rule to ``{seed: (trained_unsafe, trained_clean, neutral_unsafe)}``."""
    ok = [s for s, (_, clean, _) in rows.items() if clean]
    return sorted(ok, key=lambda s: (-rows[s][2], s))[:k]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", choices=("sa", "ma"), required=True)
    p.add_argument("--model", required=True, help="trained model (path without .zip is fine)")
    p.add_argument("--n", type=int, default=40, help="scenario seeds 1..n")
    p.add_argument("--k", type=int, default=5, help="scenarios to keep")
    p.add_argument("--n-agents", type=int, default=10)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--out", default=None, help="write the table and the choice as JSON")
    args = p.parse_args()

    from multiprocessing import get_context
    seeds = range(1, args.n + 1)
    tasks = [(args.env, s, m, args.n_agents) for s in seeds for m in (args.model, None)]
    with get_context("spawn").Pool(min(args.workers, len(tasks))) as pool:
        res = pool.map(_episode, tasks)
    trained = {s: (u, c) for s, t, u, c in res if t}
    neutral = {s: u for s, t, u, _ in res if not t}
    rows = {s: (trained[s][0], trained[s][1], neutral[s]) for s in seeds}
    chosen = pick(rows, args.k)

    print(f"{args.env}: {args.model}")
    print(f"{'seed':>4s} {'trained unsafe s':>16s} {'clean':>6s} {'neutral unsafe s':>16s}")
    for s in seeds:
        u, c, n = rows[s]
        print(f"{s:4d} {u:16.0f} {'yes' if c else 'no':>6s} {n:16.0f}" + ("   <- chosen" if s in chosen else ""))
    n_clean = sum(c for _, c, _ in rows.values())
    print(f"rule: of seeds 1..{args.n}, the {args.k} where the trained policy is clean and the neutral "
          f"policy is worst. Trained clean in {n_clean}/{args.n}. Chosen: {' '.join(map(str, chosen))}")
    if len(chosen) < args.k:
        print(f"WARNING: only {len(chosen)} clean scenarios; raise --n")
    if args.out:
        with open(args.out, "w", newline="\n") as fh:
            json.dump(dict(env=args.env, model=args.model, n=args.n, k=args.k, chosen=chosen,
                           trained_clean=n_clean,
                           table={s: dict(trained_unsafe=rows[s][0], trained_clean=rows[s][1],
                                          neutral_unsafe=rows[s][2]) for s in seeds}), fh, indent=1)


if __name__ == "__main__":
    main()
