"""Official evaluation (seed 42) of our policies with the organisers' harness.

Only the two edit points of scripts/evaluate_competition.py (make_env and
load_policy) are replaced; seeding, episode loop, metrics and the summary
table are the organisers' code as shipped.

    python -m solution.evaluate --env sa --episodes 1000 --model ../submission/sa/ppo_sa --out sa.csv
    python -m solution.evaluate --env ma --episodes 1000 --model ../submission/ma/ppo_ma --out ma.csv

Without --model it evaluates the do-nothing policy.
"""

import argparse
import json
import os

import numpy as np

from scripts import evaluate_competition as harness
from solution.envs import ShapedCompetitionEnv, ShapedCompetitionZooEnv
from solution.wrappers import FlattenObs

# settings stored in <model>_meta.json that define the MDP a model was trained in
MDP_META_KEYS = ("route_mode", "route_clearance_nm", "route_budget_frac",
                 "route_budget_hard_frac", "route_least_penetration",
                 "route_lateral_offset_nm", "speed_mode", "speed_min_kt", "turn_guard_deg",
                 "intent", "route_obs", "probe", "share_intent", "action_frequency")

MDP_KW = {}


def resolve_mdp_kw(model_path):
    """Env kwargs for a model, read from the _meta.json next to it."""
    if not model_path:
        return {}
    stem = os.path.splitext(model_path)[0]
    path = stem + "_meta.json"
    if not os.path.exists(path):
        # checkpoints live one level down, in <out>_ck/
        parent = os.path.dirname(os.path.dirname(os.path.abspath(model_path)))
        metas = [f for f in os.listdir(parent) if f.endswith("_meta.json")]
        if len(metas) != 1:
            raise SystemExit(f"no _meta.json found for {model_path}")
        path = os.path.join(parent, metas[0])
    with open(path) as fh:
        meta = json.load(fh)
    if meta.get("obstacle_mode", "exact") != "exact" or meta.get("scan_headings", 0):
        raise SystemExit(f"{model_path} was trained with options this code no longer has")
    return {k: meta[k] for k in MDP_META_KEYS if meta.get(k) is not None}


def make_env(kind, n_agents=harness.N_AGENTS_MA):
    """Our envs; the scored parameters stay at their defaults, as the rules require."""
    if kind == "sa":
        import gymnasium as gym
        return gym.wrappers.FlattenObservation(ShapedCompetitionEnv(**MDP_KW))
    return FlattenObs(ShapedCompetitionZooEnv(n_agents=n_agents, **MDP_KW))


def load_policy(kind, model_path=None):
    if model_path is None:
        return lambda obs: np.zeros(2, dtype=np.float32)
    from stable_baselines3 import PPO
    model = PPO.load(model_path)
    return lambda obs: model.predict(obs, deterministic=True)[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", choices=["sa", "ma", "both"], default="sa")
    parser.add_argument("--episodes", type=int, default=harness.N_EPISODES)
    parser.add_argument("--n-agents", type=int, default=harness.N_AGENTS_MA)
    parser.add_argument("--model", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    MDP_KW.update(resolve_mdp_kw(args.model))
    print("MDP:", MDP_KW or "env defaults (do-nothing baseline)")

    harness.make_env = make_env
    harness.load_policy = load_policy
    kinds = ["sa", "ma"] if args.env == "both" else [args.env]
    for kind in kinds:
        harness.evaluate(kind, args.episodes, args.n_agents, args.model,
                         args.out if args.env != "both" else None)


if __name__ == "__main__":
    main()
