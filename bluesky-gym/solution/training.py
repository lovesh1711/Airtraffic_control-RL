"""Shared PPO training code for train_sa.py and train_ma.py."""

import argparse
import json
import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecNormalize

from solution.envs import OFFICIAL_EVAL_SEED, official_seed_clash
from solution.resume import (LinearLR, apply_hparam_overrides, per_decision_discount,
                             resize_rollout, rollout_steps, saved_rollout_size, set_discount,
                             set_lr_schedule, vecnorm_candidates)

SHAPING_ARGS = ("w_step", "w_separation", "w_progress", "w_obstacle", "w_boundary",
                "w_action", "w_smooth")


def common_parser(description):
    p = argparse.ArgumentParser(description=description)
    a = p.add_argument
    a("--timesteps", type=int, default=200_000)
    a("--out", required=True, help="output path prefix, e.g. results/run/ppo_sa")
    a("--seed", type=int, default=0)
    a("--resume", default=None, help="model .zip to continue training")
    a("--checkpoint-freq", type=int, default=250_000, help="env steps between checkpoints (0 = off)")
    a("--tensorboard", default=None)

    # PPO
    a("--n-steps", type=int, default=512, help="rollout length per env")
    a("--rollout-size", type=int, default=None,
      help="samples per update over all envs (overrides --n-steps; on resume the "
           "checkpoint's size is kept by default)")
    a("--batch-size", type=int, default=512)
    a("--lr", type=float, default=3e-4)
    a("--lr-final", type=float, default=None, help="decay lr linearly to this over this run")
    a("--n-epochs", type=int, default=None)
    a("--target-kl", type=float, default=None)
    a("--ent-coef", type=float, default=None)
    a("--gamma", type=float, default=None, help="default 0.995 per 10 s decision")
    a("--net-arch", type=int, nargs="+", default=[256, 256])
    a("--embed-dim", type=int, default=64)
    a("--n-heads", type=int, default=4)

    # observation
    a("--intent", action="store_true", help="closest approach along both aircraft's plans")
    a("--route-obs", action="store_true", help="route anchor, next leg, remaining route, time slack")
    a("--probe", action="store_true", help="manoeuvre probe (needs --speed-mode anchored)")
    a("--share-intent", action="store_true", help="neighbours' current commands (multi-agent)")

    # action
    a("--route-mode", default="goal", choices=("goal", "visibility"))
    a("--route-clearance-nm", type=float, default=2.5)
    a("--route-budget-frac", type=float, default=0.8,
      help="longest route as a fraction of the distance flyable in 3000 s")
    a("--route-budget-hard-frac", type=float, default=None,
      help="longer route allowed only when nothing fits --route-budget-frac")
    a("--route-least-penetration", action="store_true",
      help="if no clear route fits, take the one least inside obstacles")
    a("--route-lateral-offset-nm", type=float, default=0.0,
      help="fly routes this far right of the shortest path (separate lanes)")
    a("--speed-mode", default="relative", choices=("relative", "anchored"))
    a("--speed-min-kt", type=float, default=200.0)
    a("--turn-guard-deg", type=float, default=None)
    a("--action-frequency", type=int, default=10, help="seconds between decisions")

    # reward
    for name in SHAPING_ARGS:
        a("--" + name.replace("_", "-"), type=float, default=None)
    a("--w-miss", type=float, default=0.0, help="penalty for timing out without the goal")
    a("--w-intrusion-event", type=float, default=0.0)
    a("--w-restricted-event", type=float, default=0.0)
    a("--w-exit-event", type=float, default=0.0)
    a("--w-sep-barrier", type=float, default=0.0)
    a("--sep-barrier-nm", type=float, default=6.5)
    a("--separation-mode", default="plan", choices=("straight", "plan"),
      help="predict conflicts along straight lines or along both aircraft's plans")

    # training scenarios
    a("--replay-prob", type=float, default=0.0, help="prioritised scenario replay")
    a("--replay-size", type=int, default=256)
    a("--replay-temperature", type=float, default=0.3)
    return p


def check_args(p, args, n_workers):
    if official_seed_clash(args.seed, n_workers):
        p.error(f"--seed {args.seed} with {n_workers} workers would seed one with "
                f"{OFFICIAL_EVAL_SEED}, the official test stream")
    if args.action_frequency < 1:
        p.error("--action-frequency must be at least 1")


def env_kwargs(args):
    """Environment kwargs from the parsed arguments."""
    kw = {k: getattr(args, k) for k in SHAPING_ARGS if getattr(args, k) is not None}
    kw.update(route_mode=args.route_mode, route_clearance_nm=args.route_clearance_nm,
              route_budget_frac=args.route_budget_frac, intent=args.intent,
              route_obs=args.route_obs, probe=args.probe, share_intent=args.share_intent,
              speed_mode=args.speed_mode, speed_min_kt=args.speed_min_kt,
              separation_mode=args.separation_mode)
    optional = dict(turn_guard_deg=args.turn_guard_deg,
                    route_budget_hard_frac=args.route_budget_hard_frac)
    kw.update({k: v for k, v in optional.items() if v is not None})
    if args.route_least_penetration:
        kw["route_least_penetration"] = True
    if args.route_lateral_offset_nm:
        kw["route_lateral_offset_nm"] = args.route_lateral_offset_nm
    if args.action_frequency != 10:
        kw["action_frequency"] = args.action_frequency
    for k in ("w_miss", "w_intrusion_event", "w_restricted_event", "w_exit_event",
              "w_sep_barrier"):
        if getattr(args, k):
            kw[k] = getattr(args, k)
    if args.w_sep_barrier:
        kw["sep_barrier_nm"] = args.sep_barrier_nm
    if args.replay_prob > 0:
        kw.update(replay_prob=args.replay_prob, replay_size=args.replay_size,
                  replay_temperature=args.replay_temperature)
    return kw


def normalize_reward(venv, gamma):
    # observations are already scaled; only the reward is normalised, so the
    # saved policy needs no statistics at evaluation time
    return VecNormalize(venv, norm_obs=False, norm_reward=True, clip_reward=10.0, gamma=gamma)


def make_model(args, venv, n_others, ent_coef_fresh, policy_cls="MlpPolicy"):
    """Fresh PPO model, or the --resume checkpoint with this run's settings.

    Returns (model, venv): on resume the reward normaliser stats are loaded.
    """
    gamma, gae_lambda = per_decision_discount(args.action_frequency, args.gamma)
    os.makedirs(f"{args.out}_ck", exist_ok=True)

    if args.resume:
        for cand in vecnorm_candidates(args.resume, f"{args.out}_ck"):
            if os.path.exists(cand):
                venv = VecNormalize.load(cand, venv.venv)
                venv.training = True
                venv.gamma = gamma
                print(f"loaded VecNormalize stats from {cand}")
                break
        else:
            print("WARNING: no VecNormalize stats found; reward scaling restarts")
        model = PPO.load(args.resume, env=venv, device="cpu")
        print(f"resumed from {args.resume} at {model.num_timesteps} steps")
        changed = apply_hparam_overrides(model, ent_coef=args.ent_coef, n_epochs=args.n_epochs,
                                         target_kl=args.target_kl)
        changed.update(set_discount(model, gamma, gae_lambda))
        for k, (was, now) in changed.items():
            print(f"override on resume: {k} {was} -> {now}")
        target = args.rollout_size or saved_rollout_size(args.resume)
        resize_rollout(model, rollout_steps(target, model.n_envs))
    else:
        from solution.encoder import SetEncoderExtractor, build_extractor_kwargs
        n_steps = rollout_steps(args.rollout_size, venv.num_envs) if args.rollout_size else args.n_steps
        policy_kwargs = dict(
            net_arch=list(args.net_arch),
            features_extractor_class=SetEncoderExtractor,
            features_extractor_kwargs=build_extractor_kwargs(
                n_intruders=n_others, embed_dim=args.embed_dim, n_heads=args.n_heads,
                intent=args.intent, route_obs=args.route_obs, probe=args.probe,
                share_intent=args.share_intent))
        model = PPO(policy_cls, venv, n_steps=n_steps, batch_size=args.batch_size,
                    learning_rate=args.lr,
                    ent_coef=ent_coef_fresh if args.ent_coef is None else args.ent_coef,
                    n_epochs=10 if args.n_epochs is None else args.n_epochs,
                    target_kl=args.target_kl, gamma=gamma, gae_lambda=gae_lambda,
                    policy_kwargs=policy_kwargs, device="cpu", seed=args.seed,
                    tensorboard_log=args.tensorboard, verbose=1)
    print(f"rollout: {model.n_steps} steps x {model.n_envs} envs = "
          f"{model.n_steps * model.n_envs} samples per update; n_epochs {model.n_epochs}, "
          f"target_kl {model.target_kl}, gamma {model.gamma:.5f}")
    return model, venv


def write_meta(args, model, **extra):
    """<out>_meta.json: the MDP the model belongs to (read back by evaluation)."""
    meta = dict(obstacle_mode="exact", route_mode=args.route_mode,
                route_clearance_nm=args.route_clearance_nm,
                route_budget_frac=args.route_budget_frac,
                route_budget_hard_frac=args.route_budget_hard_frac,
                route_least_penetration=args.route_least_penetration,
                route_lateral_offset_nm=args.route_lateral_offset_nm,
                speed_mode=args.speed_mode, speed_min_kt=args.speed_min_kt,
                turn_guard_deg=args.turn_guard_deg, action_frequency=args.action_frequency,
                intent=args.intent, route_obs=args.route_obs, probe=args.probe,
                share_intent=args.share_intent,
                w_miss=args.w_miss, w_intrusion_event=args.w_intrusion_event,
                w_restricted_event=args.w_restricted_event, w_exit_event=args.w_exit_event,
                w_sep_barrier=args.w_sep_barrier, sep_barrier_nm=args.sep_barrier_nm,
                separation_mode=args.separation_mode, replay_prob=args.replay_prob, resumed_from=args.resume, seed=args.seed,
                lr=args.lr, lr_final=args.lr_final, n_epochs=int(model.n_epochs),
                target_kl=model.target_kl, ent_coef=float(model.ent_coef),
                gamma=float(model.gamma), gae_lambda=float(model.gae_lambda), **extra)
    with open(args.out + "_meta.json", "w") as fh:
        json.dump(meta, fh)


def learn(args, model, venv, callbacks, n_parallel):
    """Train, checkpointing every --checkpoint-freq env steps, and save."""
    # written before training so a killed job's checkpoints can still be evaluated
    if args.checkpoint_freq > 0:
        callbacks.append(CheckpointCallback(save_freq=max(1, args.checkpoint_freq // n_parallel),
                                            save_path=f"{args.out}_ck", name_prefix="ck",
                                            save_vecnormalize=True))
    if args.lr_final is not None:
        set_lr_schedule(model, LinearLR(args.lr, args.lr_final, model.num_timesteps,
                                        args.timesteps))
        print(f"learning rate {args.lr:g} -> {args.lr_final:g} over this run")
    model.learn(total_timesteps=args.timesteps, progress_bar=False,
                reset_num_timesteps=not args.resume, callback=callbacks or None)
    model.save(args.out)
    if isinstance(venv, VecNormalize):
        venv.save(os.path.join(f"{args.out}_ck", "vecnormalize.pkl"))
    print(f"saved policy to {args.out}.zip")
    venv.close()
