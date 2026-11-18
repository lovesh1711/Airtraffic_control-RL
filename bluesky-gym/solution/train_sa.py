"""Single-agent PPO training.

BlueSky is one simulator per process, so rollouts scale with SubprocVecEnv
(one env per worker process). Set OMP_NUM_THREADS=1 with many workers.

    python -m solution.train_sa --n-envs 8 --timesteps 200000 --out models/ppo_sa ...
"""

import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from solution.envs import ShapedCompetitionEnv
from solution.resume import per_decision_discount
from solution.training import (check_args, common_parser, env_kwargs, learn, make_model,
                               normalize_reward, write_meta)

ENT_COEF_FRESH = 0.02


def _make_one(seed, **kwargs):
    def _init():
        env = Monitor(gym.wrappers.FlattenObservation(ShapedCompetitionEnv(render_mode=None, **kwargs)))
        env.reset(seed=seed)
        return env
    return _init


def build_vec_env(n_envs, seed, gamma, **kwargs):
    fns = [_make_one(seed + i, **kwargs) for i in range(n_envs)]
    if n_envs == 1:
        venv = DummyVecEnv(fns)
    else:
        # spawn, not fork: every worker needs its own fresh BlueSky
        venv = SubprocVecEnv(fns, start_method="spawn")
    return normalize_reward(VecMonitor(venv), gamma)


def main():
    p = common_parser(__doc__)
    p.add_argument("--n-envs", type=int, default=4, help="worker processes")
    p.add_argument("--bounded-mean", action="store_true",
                   help="tanh on the policy mean (fresh runs only)")
    p.add_argument("--max-std", type=float, default=None, help="cap on the policy std")
    p.add_argument("--hard-traffic-prob", type=float, default=0.0,
                   help="share of fresh training scenarios with conflict-timed intruders")
    p.add_argument("--hard-traffic-frac", type=float, default=1.0)
    p.add_argument("--hard-traffic-jitter-s", type=float, default=60.0)
    p.set_defaults(n_epochs=4, target_kl=0.02)
    args = p.parse_args()
    check_args(p, args, args.n_envs)
    if args.bounded_mean and args.resume:
        p.error("--bounded-mean changes the policy class; it cannot be added on --resume")

    kw = env_kwargs(args)
    if args.hard_traffic_prob > 0:
        kw.update(hard_traffic_prob=args.hard_traffic_prob,
                  hard_traffic_frac=args.hard_traffic_frac,
                  hard_traffic_jitter_s=args.hard_traffic_jitter_s)
    print("env:", kw)

    gamma, _ = per_decision_discount(args.action_frequency, args.gamma)
    venv = build_vec_env(args.n_envs, args.seed, gamma, **kw)
    policy_cls = "MlpPolicy"
    if args.bounded_mean:
        from solution.policies import BoundedMeanActorCriticPolicy
        policy_cls = BoundedMeanActorCriticPolicy
    model, venv = make_model(args, venv, n_others=10, ent_coef_fresh=ENT_COEF_FRESH,
                             policy_cls=policy_cls)

    from solution.policies import ClampLogStd, describe_policy
    print(describe_policy(model))
    callbacks = []
    if args.replay_prob > 0:
        from solution.replay import ReplayStatsCallback
        callbacks.append(ReplayStatsCallback(every=10))
    if args.max_std is not None:
        callbacks.append(ClampLogStd(max_std=args.max_std))

    write_meta(args, model, bounded_mean=args.bounded_mean, max_std=args.max_std,
               hard_traffic_prob=args.hard_traffic_prob)
    learn(args, model, venv, callbacks, n_parallel=args.n_envs)


if __name__ == "__main__":
    main()
