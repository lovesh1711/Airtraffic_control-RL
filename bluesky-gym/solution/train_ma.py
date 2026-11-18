"""Multi-agent PPO training: one shared policy for all 10 aircraft.

Each worker process runs one ParallelEnv (its own BlueSky) and exposes its 10
aircraft as 10 vector-env slots, so --n-procs N gives N * 10 parallel samples.
The workers are built from factories inside the child process; pickling a
constructed env would hand the child an uninitialised BlueSky.

    python -m solution.train_ma --n-procs 4 --timesteps 500000 --out models/ppo_ma ...
"""

import numpy as np
import supersuit as ss
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecMonitor
from supersuit.vector import MakeCPUAsyncConstructor

from solution.encoder import competition_obs_space
from solution.envs import ShapedCompetitionZooEnv
from solution.resume import per_decision_discount
from solution.training import (check_args, common_parser, env_kwargs, learn, make_model,
                               normalize_reward, write_meta)
from solution.wrappers import FlattenObs, SeedableSB3VecEnvWrapper

N_AGENTS = 10
ENT_COEF_FRESH = 0.005


def _ma_env_fn(n_agents, kwargs):
    def _f():
        env = FlattenObs(ShapedCompetitionZooEnv(render_mode=None, n_agents=n_agents, **kwargs))
        env = ss.black_death_v3(env)                 # keep finished aircraft until the episode ends
        return ss.pettingzoo_env_to_vec_env_v1(env)
    return _f


def analytic_obs_space(n_agents, kwargs):
    """Flat observation space, computed in the parent without a simulator.
    Must match what the workers produce."""
    return spaces.flatten_space(competition_obs_space(
        n_intruders=n_agents - 1, intent=kwargs.get("intent", False),
        route_obs=kwargs.get("route_obs", False), probe=kwargs.get("probe", False),
        share_intent=kwargs.get("share_intent", False)))


def build_vec_env(n_agents, n_procs, gamma, **kwargs):
    if n_procs <= 1:
        venv = SeedableSB3VecEnvWrapper(_ma_env_fn(n_agents, kwargs)())
    else:
        fns = [_ma_env_fn(n_agents, kwargs) for _ in range(n_procs)]
        act_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float64)
        venv = SeedableSB3VecEnvWrapper(MakeCPUAsyncConstructor(n_procs)(
            fns, analytic_obs_space(n_agents, kwargs), act_space))
    return normalize_reward(VecMonitor(venv), gamma)


def main():
    p = common_parser(__doc__)
    p.add_argument("--n-procs", type=int, default=1, help="worker processes")
    p.add_argument("--n-agents", type=int, default=N_AGENTS)
    args = p.parse_args()
    check_args(p, args, args.n_procs)

    kw = env_kwargs(args)
    print("env:", kw)
    gamma, _ = per_decision_discount(args.action_frequency, args.gamma)
    venv = build_vec_env(args.n_agents, args.n_procs, gamma, **kw)
    model, venv = make_model(args, venv, n_others=args.n_agents - 1,
                             ent_coef_fresh=ENT_COEF_FRESH)

    callbacks = []
    if args.replay_prob > 0:
        from solution.replay import ReplayStatsCallback
        callbacks.append(ReplayStatsCallback(every=10))

    write_meta(args, model)
    learn(args, model, venv, callbacks, n_parallel=args.n_agents * args.n_procs)


if __name__ == "__main__":
    main()
